import Mathlib

namespace school_problem_l3034_303495

/-- Represents a school with a specific number of classes and students. -/
structure School where
  num_classes : Nat
  largest_class : Nat
  difference : Nat
  total_students : Nat

/-- Calculates the total number of students in the school. -/
def calculate_total (s : School) : Nat :=
  let series := List.range s.num_classes
  series.foldr (fun i acc => acc + s.largest_class - i * s.difference) 0

/-- Theorem stating the properties of the school in the problem. -/
theorem school_problem :
  ∃ (s : School),
    s.num_classes = 5 ∧
    s.largest_class = 32 ∧
    s.difference = 2 ∧
    s.total_students = 140 ∧
    calculate_total s = s.total_students :=
  sorry

end school_problem_l3034_303495


namespace negative_square_to_fourth_power_l3034_303479

theorem negative_square_to_fourth_power (a : ℝ) : (-a^2)^4 = a^8 := by
  sorry

end negative_square_to_fourth_power_l3034_303479


namespace min_reciprocal_sum_l3034_303498

theorem min_reciprocal_sum (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : 2*m + 2*n = 1) :
  1/m + 1/n ≥ 8 := by
  sorry

end min_reciprocal_sum_l3034_303498


namespace cubic_root_sum_l3034_303493

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 6*p^2 + 8*p - 3 = 0 →
  q^3 - 6*q^2 + 8*q - 3 = 0 →
  r^3 - 6*r^2 + 8*r - 3 = 0 →
  p / (q*r + 2) + q / (p*r + 2) + r / (p*q + 2) = 6/5 := by
sorry

end cubic_root_sum_l3034_303493


namespace quadratic_coefficient_l3034_303476

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ
  eval : ℝ → ℝ := λ x => (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ)

/-- The theorem statement -/
theorem quadratic_coefficient (f : QuadraticFunction) 
  (h1 : f.eval 1 = 4)
  (h2 : f.eval (-2) = 3)
  (h3 : f.eval (-1) = 2)
  (h4 : ∀ x : ℝ, f.eval x ≤ f.eval (-1)) :
  f.a = 1 := by sorry

end quadratic_coefficient_l3034_303476


namespace rectangle_area_rectangle_area_is_240_l3034_303405

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 5) * circle_radius
  let rectangle_area : ℝ := rectangle_length * rectangle_breadth
  rectangle_area

theorem rectangle_area_is_240 :
  rectangle_area 3600 10 = 240 := by sorry

end rectangle_area_rectangle_area_is_240_l3034_303405


namespace mrs_hilt_pizzas_l3034_303447

/-- The number of slices in each pizza -/
def slices_per_pizza : ℕ := 8

/-- The total number of slices Mrs. Hilt had -/
def total_slices : ℕ := 16

/-- The number of pizzas Mrs. Hilt bought -/
def pizzas_bought : ℕ := total_slices / slices_per_pizza

theorem mrs_hilt_pizzas : pizzas_bought = 2 := by
  sorry

end mrs_hilt_pizzas_l3034_303447


namespace number_of_boys_in_class_l3034_303409

/-- Proves the number of boys in a class given certain height information -/
theorem number_of_boys_in_class 
  (initial_average : ℝ)
  (wrong_height : ℝ)
  (actual_height : ℝ)
  (actual_average : ℝ)
  (h1 : initial_average = 183)
  (h2 : wrong_height = 166)
  (h3 : actual_height = 106)
  (h4 : actual_average = 181) :
  ∃ n : ℕ, n * initial_average - (wrong_height - actual_height) = n * actual_average ∧ n = 30 :=
by sorry

end number_of_boys_in_class_l3034_303409


namespace smallest_n_satisfying_condition_l3034_303467

/-- The probability that no two of three independently chosen real numbers 
    from [0, n] are within 2 units of each other is greater than 1/2 -/
def probability_condition (n : ℕ) : Prop :=
  (n - 4)^3 / n^3 > 1/2

/-- 12 is the smallest positive integer satisfying the probability condition -/
theorem smallest_n_satisfying_condition : 
  (∀ k < 12, ¬ probability_condition k) ∧ probability_condition 12 :=
sorry

end smallest_n_satisfying_condition_l3034_303467


namespace sin_double_angle_with_tan_three_l3034_303433

theorem sin_double_angle_with_tan_three (θ : ℝ) :
  (∃ (x y : ℝ), x > 0 ∧ y = 3 * x ∧ Real.cos θ * x = Real.sin θ * y) →
  Real.sin (2 * θ) = 3 / 5 := by
  sorry

end sin_double_angle_with_tan_three_l3034_303433


namespace check_amount_proof_l3034_303435

theorem check_amount_proof :
  ∃! (x y : ℕ), 
    y ≤ 99 ∧
    (y : ℚ) + (x : ℚ) / 100 - 5 / 100 = 2 * ((x : ℚ) + (y : ℚ) / 100) ∧
    x = 31 ∧ y = 63 := by
  sorry

end check_amount_proof_l3034_303435


namespace p_sufficient_not_necessary_q_l3034_303460

-- Define the conditions p and q
def p (x : ℝ) : Prop := 0 < x ∧ x < 2
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary_q :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) :=
sorry

end p_sufficient_not_necessary_q_l3034_303460


namespace unique_right_triangle_with_2021_leg_l3034_303485

theorem unique_right_triangle_with_2021_leg : 
  ∃! (a b c : ℕ+), (a = 2021 ∨ b = 2021) ∧ a^2 + b^2 = c^2 := by
  sorry

end unique_right_triangle_with_2021_leg_l3034_303485


namespace orchestra_members_count_l3034_303489

theorem orchestra_members_count : ∃! n : ℕ, 
  150 < n ∧ n < 250 ∧ 
  n % 8 = 1 ∧ 
  n % 6 = 2 ∧ 
  n % 9 = 3 := by
  sorry

end orchestra_members_count_l3034_303489


namespace mushroom_pickers_l3034_303402

theorem mushroom_pickers (n : ℕ) (A V S R : ℚ) : 
  (∀ i : Fin n, i.val ≠ 0 ∧ i.val ≠ 1 ∧ i.val ≠ 2 → A / 2 = V + A / 2) →  -- Condition 1
  (S + A = R + V + A) →                                                   -- Condition 2
  (A > 0) →                                                               -- Anya has mushrooms
  (n > 3) →                                                               -- At least 4 children
  (n : ℚ) * (A / 2) = A + V + S + R →                                     -- Total mushrooms
  n = 6 := by
sorry

end mushroom_pickers_l3034_303402


namespace orchestra_only_females_l3034_303423

theorem orchestra_only_females (
  band_females : ℕ) (band_males : ℕ) 
  (orchestra_females : ℕ) (orchestra_males : ℕ)
  (both_females : ℕ) (both_males : ℕ)
  (total_students : ℕ) :
  band_females = 120 →
  band_males = 110 →
  orchestra_females = 100 →
  orchestra_males = 130 →
  both_females = 90 →
  both_males = 80 →
  total_students = 280 →
  total_students = band_females + band_males + orchestra_females + orchestra_males - both_females - both_males →
  orchestra_females - both_females = 10 := by
sorry

end orchestra_only_females_l3034_303423


namespace pascals_theorem_l3034_303430

-- Define a circle
def Circle : Type := {p : ℝ × ℝ // (p.1^2 + p.2^2 = 1)}

-- Define a line
def Line (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {r : ℝ × ℝ | ∃ t : ℝ, r = (1 - t) • p + t • q}

-- Define the intersection of two lines
def intersect (l1 l2 : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  l1 ∩ l2

-- Define collinearity
def collinear (p q r : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, q - p = t₁ • (r - p) ∧ r - p = t₂ • (q - p)

-- State Pascal's Theorem
theorem pascals_theorem 
  (A B C D E F : Circle) 
  (P : intersect (Line A.val B.val) (Line D.val E.val))
  (Q : intersect (Line B.val C.val) (Line E.val F.val))
  (R : intersect (Line C.val D.val) (Line F.val A.val)) :
  collinear P Q R :=
sorry

end pascals_theorem_l3034_303430


namespace stick_swap_theorem_l3034_303452

/-- Represents a set of three sticks --/
structure StickSet where
  stick1 : Real
  stick2 : Real
  stick3 : Real
  sum_is_one : stick1 + stick2 + stick3 = 1
  all_positive : stick1 > 0 ∧ stick2 > 0 ∧ stick3 > 0

/-- Checks if a triangle can be formed from a set of sticks --/
def can_form_triangle (s : StickSet) : Prop :=
  s.stick1 + s.stick2 > s.stick3 ∧
  s.stick1 + s.stick3 > s.stick2 ∧
  s.stick2 + s.stick3 > s.stick1

theorem stick_swap_theorem (vintik_initial shpuntik_initial vintik_final shpuntik_final : StickSet) :
  can_form_triangle vintik_initial →
  can_form_triangle shpuntik_initial →
  ¬can_form_triangle vintik_final →
  (∃ (x y : Real), 
    vintik_final.stick1 = vintik_initial.stick1 ∧
    vintik_final.stick2 = vintik_initial.stick2 ∧
    vintik_final.stick3 = y ∧
    shpuntik_final.stick1 = shpuntik_initial.stick1 ∧
    shpuntik_final.stick2 = shpuntik_initial.stick2 ∧
    shpuntik_final.stick3 = x ∧
    x + y = vintik_initial.stick3 + shpuntik_initial.stick3) →
  can_form_triangle shpuntik_final := by
  sorry

end stick_swap_theorem_l3034_303452


namespace square_cylinder_volume_l3034_303482

/-- A cylinder with a square cross-section and lateral area 4π has volume 2π -/
theorem square_cylinder_volume (h : ℝ) (lateral_area : ℝ) (volume : ℝ) 
  (h_positive : h > 0)
  (lateral_area_eq : lateral_area = 4 * Real.pi)
  (lateral_area_def : lateral_area = h * h * Real.pi)
  (volume_def : volume = h * h * h / 4) : 
  volume = 2 * Real.pi := by
sorry

end square_cylinder_volume_l3034_303482


namespace arnold_jellybean_count_l3034_303432

/-- Given the following conditions about jellybean counts:
  - Tino has 24 more jellybeans than Lee
  - Arnold has half as many jellybeans as Lee
  - Tino has 34 jellybeans
Prove that Arnold has 5 jellybeans. -/
theorem arnold_jellybean_count (tino lee arnold : ℕ) : 
  tino = lee + 24 →
  arnold = lee / 2 →
  tino = 34 →
  arnold = 5 := by
  sorry

end arnold_jellybean_count_l3034_303432


namespace cousins_age_sum_l3034_303417

theorem cousins_age_sum : ∀ (a b c : ℕ),
  a < 10 ∧ b < 10 ∧ c < 10 →  -- single-digit positive integers
  a ≠ b ∧ b ≠ c ∧ a ≠ c →     -- distinct
  (a < c ∧ b < c) →           -- one cousin is older than the other two
  a * b = 18 →                -- product of younger two
  c * min a b = 28 →          -- product of oldest and youngest
  a + b + c = 18 :=           -- sum of all three
by sorry

end cousins_age_sum_l3034_303417


namespace plywood_cut_perimeter_difference_l3034_303446

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Represents a way to cut the plywood --/
structure CutPattern where
  piece : Rectangle
  num_pieces : ℕ

theorem plywood_cut_perimeter_difference :
  ∀ (cuts : List CutPattern),
    (∀ c ∈ cuts, c.num_pieces = 6 ∧ c.piece.length * c.piece.width * 6 = 54) →
    (∃ c ∈ cuts, perimeter c.piece = 20) →
    (∀ c ∈ cuts, perimeter c.piece ≥ 15) →
    (∃ c ∈ cuts, perimeter c.piece = 15) →
    20 - 15 = 5 := by
  sorry

end plywood_cut_perimeter_difference_l3034_303446


namespace random_events_identification_l3034_303403

-- Define the type for events
inductive Event : Type
  | draw_glasses : Event
  | guess_digit : Event
  | electric_charges : Event
  | lottery_win : Event

-- Define what it means for an event to be random
def is_random_event (e : Event) : Prop :=
  ∀ (outcome : Prop), ¬(outcome ∧ ¬outcome)

-- State the theorem
theorem random_events_identification :
  (is_random_event Event.draw_glasses) ∧
  (is_random_event Event.guess_digit) ∧
  (is_random_event Event.lottery_win) ∧
  (¬is_random_event Event.electric_charges) := by
  sorry

end random_events_identification_l3034_303403


namespace circle_polar_rectangular_equivalence_l3034_303462

/-- The polar coordinate equation of a circle is equivalent to its rectangular coordinate equation -/
theorem circle_polar_rectangular_equivalence (x y ρ θ : ℝ) :
  (x^2 + y^2 - 2*x = 0) ↔ (ρ = 2*Real.cos θ ∧ x = ρ*Real.cos θ ∧ y = ρ*Real.sin θ) :=
sorry

end circle_polar_rectangular_equivalence_l3034_303462


namespace no_functions_satisfy_condition_l3034_303414

theorem no_functions_satisfy_condition :
  ¬∃ (f g : ℝ → ℝ), ∀ (x y : ℝ), f x * g y = x + y - 1 := by
  sorry

end no_functions_satisfy_condition_l3034_303414


namespace harrison_elementary_students_l3034_303428

/-- The number of students in Harrison Elementary School -/
def total_students : ℕ := 1060

/-- The fraction of students remaining at Harrison Elementary School -/
def remaining_fraction : ℚ := 3/5

/-- The number of grade levels -/
def grade_levels : ℕ := 3

/-- The number of students in each advanced class -/
def advanced_class_size : ℕ := 20

/-- The number of normal classes per grade level -/
def normal_classes_per_grade : ℕ := 6

/-- The number of students in each normal class -/
def normal_class_size : ℕ := 32

/-- Theorem stating the total number of students in Harrison Elementary School -/
theorem harrison_elementary_students :
  total_students = 
    (grade_levels * advanced_class_size + 
     grade_levels * normal_classes_per_grade * normal_class_size) / remaining_fraction :=
by sorry

end harrison_elementary_students_l3034_303428


namespace special_ap_ratio_l3034_303458

/-- An arithmetic progression with the property that the sum of its first ten terms
    is four times the sum of its first five terms. -/
structure SpecialAP where
  a : ℝ  -- First term
  d : ℝ  -- Common difference
  sum_condition : (10 * a + 45 * d) = 4 * (5 * a + 10 * d)

/-- The ratio of the first term to the common difference in a SpecialAP is 1:2. -/
theorem special_ap_ratio (ap : SpecialAP) : ap.a / ap.d = 1 / 2 := by
  sorry

#check special_ap_ratio

end special_ap_ratio_l3034_303458


namespace natural_number_representation_l3034_303456

theorem natural_number_representation (n : ℕ) : 
  (∃ a b c : ℕ, (a + b + c)^2 = n * a * b * c ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ↔ 
  n ∈ ({1, 2, 3, 4, 5, 6, 8, 9} : Set ℕ) := by
sorry

end natural_number_representation_l3034_303456


namespace two_diggers_two_hours_l3034_303484

/-- The rate at which diggers dig pits -/
def digging_rate (diggers : ℚ) (pits : ℚ) (hours : ℚ) : ℚ :=
  pits / (diggers * hours)

/-- The number of pits dug given a rate, number of diggers, and hours -/
def pits_dug (rate : ℚ) (diggers : ℚ) (hours : ℚ) : ℚ :=
  rate * diggers * hours

theorem two_diggers_two_hours 
  (h : digging_rate (3/2) (3/2) (3/2) = digging_rate 2 x 2) : x = 8/3 := by
  sorry

#check two_diggers_two_hours

end two_diggers_two_hours_l3034_303484


namespace inequality_proof_l3034_303443

theorem inequality_proof (a b c d e f : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d) (h5 : 0 ≤ e) (h6 : 0 ≤ f)
  (h7 : a + b ≤ e) (h8 : c + d ≤ f) : 
  Real.sqrt (a * c) + Real.sqrt (b * d) ≤ Real.sqrt (e * f) := by
  sorry

end inequality_proof_l3034_303443


namespace problem_solution_l3034_303496

theorem problem_solution (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = -3) :
  x + (x^3 / y^2) + (y^3 / x^2) + y = 591 := by
  sorry

end problem_solution_l3034_303496


namespace min_distance_between_ellipses_l3034_303450

/-- The minimum distance between two ellipses -/
theorem min_distance_between_ellipses :
  let ellipse1 := {(x, y) : ℝ × ℝ | x^2 / 4 + y^2 = 1}
  let ellipse2 := {(x, y) : ℝ × ℝ | (x - 1)^2 / 9 + y^2 / 9 = 1}
  (∃ (A B : ℝ × ℝ), A ∈ ellipse1 ∧ B ∈ ellipse2 ∧
    ∀ (C D : ℝ × ℝ), C ∈ ellipse1 → D ∈ ellipse2 →
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≤ Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)) ∧
  (∀ (A B : ℝ × ℝ), A ∈ ellipse1 → B ∈ ellipse2 →
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≥ 2) ∧
  (∃ (A B : ℝ × ℝ), A ∈ ellipse1 ∧ B ∈ ellipse2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2) :=
by sorry

end min_distance_between_ellipses_l3034_303450


namespace linear_function_not_in_third_quadrant_l3034_303401

/-- A linear function passing through a quadrant if there exists a point (x, y) in that quadrant satisfying the function equation -/
def passes_through_quadrant (a b : ℝ) (quad : ℕ) : Prop :=
  ∃ x y : ℝ, 
    (quad = 1 → x > 0 ∧ y > 0) ∧
    (quad = 2 → x < 0 ∧ y > 0) ∧
    (quad = 3 → x < 0 ∧ y < 0) ∧
    (quad = 4 → x > 0 ∧ y < 0) ∧
    y = a * x + b

theorem linear_function_not_in_third_quadrant :
  ¬ passes_through_quadrant (-1/2) 1 3 :=
sorry

end linear_function_not_in_third_quadrant_l3034_303401


namespace smallest_variable_l3034_303477

theorem smallest_variable (p q r s : ℝ) 
  (h : p + 3 = q - 1 ∧ p + 3 = r + 5 ∧ p + 3 = s - 2) : 
  r ≤ p ∧ r ≤ q ∧ r ≤ s := by
  sorry

end smallest_variable_l3034_303477


namespace billy_sam_money_difference_l3034_303457

theorem billy_sam_money_difference (sam_money : ℕ) (total_money : ℕ) (billy_money : ℕ) :
  sam_money = 75 →
  total_money = 200 →
  billy_money = total_money - sam_money →
  billy_money < 2 * sam_money →
  2 * sam_money - billy_money = 25 := by
  sorry

end billy_sam_money_difference_l3034_303457


namespace barycentric_centroid_relation_l3034_303463

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable {A B C X M : V}
variable (α β γ : ℝ)

/-- Given a triangle ABC and a point X with barycentric coordinates (α : β : γ),
    where α + β + γ = 1, and M is the centroid of triangle ABC,
    prove that 3 * vector(XM) = (α - β) * vector(AB) + (β - γ) * vector(BC) + (γ - α) * vector(CA) -/
theorem barycentric_centroid_relation
  (h1 : X = α • A + β • B + γ • C)
  (h2 : α + β + γ = 1)
  (h3 : M = (1/3 : ℝ) • (A + B + C)) :
  3 • (X - M) = (α - β) • (A - B) + (β - γ) • (B - C) + (γ - α) • (C - A) := by
  sorry

end barycentric_centroid_relation_l3034_303463


namespace equation_solution_l3034_303445

theorem equation_solution :
  ∀ a b c : ℤ,
  (∀ x : ℝ, (x - a) * (x - 5) + 2 = (x + b) * (x + c)) ↔
  ((a = 2 ∧ b = -3 ∧ c = -4) ∨ (a = 8 ∧ b = -6 ∧ c = -7)) :=
by sorry

end equation_solution_l3034_303445


namespace equal_area_rectangles_width_l3034_303499

/-- Given two rectangles with equal area, where one rectangle has dimensions 5 inches by W inches,
    and the other has dimensions 8 inches by 15 inches, prove that W equals 24 inches. -/
theorem equal_area_rectangles_width (W : ℝ) : 
  (5 * W = 8 * 15) → W = 24 := by sorry

end equal_area_rectangles_width_l3034_303499


namespace linda_rings_sold_l3034_303483

/-- Proves that Linda sold 8 rings given the conditions of the problem -/
theorem linda_rings_sold :
  let necklaces_sold : ℕ := 4
  let total_sales : ℕ := 80
  let necklace_price : ℕ := 12
  let ring_price : ℕ := 4
  let rings_sold : ℕ := (total_sales - necklaces_sold * necklace_price) / ring_price
  rings_sold = 8 := by
  sorry

end linda_rings_sold_l3034_303483


namespace sphere_radius_in_truncated_cone_l3034_303478

/-- A truncated cone with a tangent sphere -/
structure TruncatedConeWithSphere where
  bottomRadius : ℝ
  topRadius : ℝ
  sphereRadius : ℝ
  isTangent : Bool

/-- The theorem stating the radius of the sphere in the given problem -/
theorem sphere_radius_in_truncated_cone 
  (cone : TruncatedConeWithSphere)
  (h1 : cone.bottomRadius = 24)
  (h2 : cone.topRadius = 6)
  (h3 : cone.isTangent = true) :
  cone.sphereRadius = 12 := by
  sorry

end sphere_radius_in_truncated_cone_l3034_303478


namespace polynomial_divisibility_l3034_303412

theorem polynomial_divisibility (A B : ℝ) : 
  (∀ x : ℂ, x^2 + x + 1 = 0 → x^205 + A*x + B = 0) → 
  A + B = -1 := by
sorry

end polynomial_divisibility_l3034_303412


namespace count_rearranged_even_numbers_l3034_303471

/-- The number of different even numbers that can be formed by rearranging the digits of 124669 -/
def rearrangedEvenNumbers : ℕ := 240

/-- The original number -/
def originalNumber : ℕ := 124669

/-- Theorem stating that the number of different even numbers formed by rearranging the digits of 124669 is 240 -/
theorem count_rearranged_even_numbers :
  rearrangedEvenNumbers = 240 ∧ originalNumber ≠ rearrangedEvenNumbers :=
by sorry

end count_rearranged_even_numbers_l3034_303471


namespace window_probability_l3034_303449

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * Nat.factorial (n - k))

def probability_BIRD : ℚ := 1 / (choose 4 2)
def probability_WINDS : ℚ := 3 / (choose 5 3)
def probability_FLOW : ℚ := 1 / (choose 4 2)

theorem window_probability : 
  probability_BIRD * probability_WINDS * probability_FLOW = 1 / 120 := by
  sorry

end window_probability_l3034_303449


namespace probability_nine_heads_in_twelve_flips_probability_nine_heads_in_twelve_flips_proof_l3034_303453

/-- The probability of getting exactly 9 heads in 12 flips of a fair coin -/
theorem probability_nine_heads_in_twelve_flips : ℚ :=
  55 / 1024

/-- Proof that the probability of getting exactly 9 heads in 12 flips of a fair coin is 55/1024 -/
theorem probability_nine_heads_in_twelve_flips_proof :
  probability_nine_heads_in_twelve_flips = 55 / 1024 := by
  sorry

end probability_nine_heads_in_twelve_flips_probability_nine_heads_in_twelve_flips_proof_l3034_303453


namespace intersection_bounds_l3034_303486

theorem intersection_bounds (m : ℝ) : 
  let A : Set ℝ := {x | -2 < x ∧ x < 8}
  let B : Set ℝ := {x | 2*m - 1 < x ∧ x < m + 3}
  let U : Set ℝ := Set.univ
  ∃ (a b : ℝ), A ∩ B = {x | a < x ∧ x < b} ∧ b - a = 3 → m = -2 ∨ m = 1 :=
by
  sorry

end intersection_bounds_l3034_303486


namespace triangle_properties_l3034_303420

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively. -/
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (positive_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)
  (angle_sum : A + B + C = π)

theorem triangle_properties (t : Triangle) 
  (h1 : Real.sqrt 3 * Real.cos t.C + Real.sin t.C = (Real.sqrt 3 * t.a) / t.b)
  (h2 : t.a + t.c = 5 * Real.sqrt 7)
  (h3 : t.b = 7) :
  t.B = π / 3 ∧ t.a * t.c * Real.cos t.B = -21 :=
by sorry

end triangle_properties_l3034_303420


namespace complex_power_four_l3034_303406

theorem complex_power_four (i : ℂ) : i^2 = -1 → (1 - i)^4 = -4 := by
  sorry

end complex_power_four_l3034_303406


namespace weight_of_b_l3034_303488

-- Define the weights as real numbers
variable (a b c : ℝ)

-- Define the conditions
def average_abc : Prop := (a + b + c) / 3 = 30
def average_ab : Prop := (a + b) / 2 = 25
def average_bc : Prop := (b + c) / 2 = 28

-- Theorem statement
theorem weight_of_b (h1 : average_abc a b c) (h2 : average_ab a b) (h3 : average_bc b c) : b = 16 := by
  sorry

end weight_of_b_l3034_303488


namespace max_product_l3034_303470

def Digits : Finset Nat := {3, 5, 8, 9, 1}

def valid_two_digit (n : Nat) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ (n / 10) ∈ Digits ∧ (n % 10) ∈ Digits ∧ (n / 10) ≠ (n % 10)

def valid_three_digit (n : Nat) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ (n / 100) ∈ Digits ∧ ((n / 10) % 10) ∈ Digits ∧ (n % 10) ∈ Digits ∧
  (n / 100) ≠ ((n / 10) % 10) ∧ (n / 100) ≠ (n % 10) ∧ ((n / 10) % 10) ≠ (n % 10)

def valid_pair (a b : Nat) : Prop :=
  valid_two_digit a ∧ valid_three_digit b ∧
  (∀ d : Nat, d ∈ Digits → (d = (a / 10) ∨ d = (a % 10) ∨ d = (b / 100) ∨ d = ((b / 10) % 10) ∨ d = (b % 10)))

theorem max_product :
  ∀ a b : Nat, valid_pair a b → a * b ≤ 91 * 853 :=
sorry

end max_product_l3034_303470


namespace parallelogram_most_analogous_to_parallelepiped_l3034_303416

-- Define the types for 2D and 3D figures
inductive PlaneFigure
| Triangle
| Trapezoid
| Rectangle
| Parallelogram

inductive SpaceFigure
| Parallelepiped

-- Define the property of being formed by translation
def FormedByTranslation (plane : PlaneFigure) (space : SpaceFigure) : Prop :=
  match plane, space with
  | PlaneFigure.Parallelogram, SpaceFigure.Parallelepiped => True
  | _, _ => False

-- Define the concept of being analogous
def Analogous (plane : PlaneFigure) (space : SpaceFigure) : Prop :=
  FormedByTranslation plane space

-- Theorem statement
theorem parallelogram_most_analogous_to_parallelepiped :
  ∀ (plane : PlaneFigure),
    Analogous plane SpaceFigure.Parallelepiped →
    plane = PlaneFigure.Parallelogram :=
sorry

end parallelogram_most_analogous_to_parallelepiped_l3034_303416


namespace max_value_of_f_l3034_303461

/-- The sum of the first n terms of the sequence a_n -/
def S (n : ℕ) : ℕ := 2^n - 1

/-- The nth term of the sequence a_n -/
def a (n : ℕ) : ℕ :=
  if n = 1 then 1 else 2^(n-1)

/-- The expression to be maximized -/
def f (n : ℕ) : ℚ :=
  (a n : ℚ) / ((a n * S n : ℕ) + a 6 : ℚ)

theorem max_value_of_f :
  ∀ n : ℕ, n ≥ 1 → f n ≤ 1/15 :=
sorry

end max_value_of_f_l3034_303461


namespace cards_given_away_l3034_303497

theorem cards_given_away (brother_sets sister_sets friend_sets : ℕ) 
  (cards_per_set : ℕ) (h1 : brother_sets = 15) (h2 : sister_sets = 8) 
  (h3 : friend_sets = 4) (h4 : cards_per_set = 25) : 
  (brother_sets + sister_sets + friend_sets) * cards_per_set = 675 := by
  sorry

end cards_given_away_l3034_303497


namespace time_to_cut_kids_hair_l3034_303438

/-- Proves that the time to cut a kid's hair is 25 minutes given the specified conditions --/
theorem time_to_cut_kids_hair (
  time_woman : ℕ)
  (time_man : ℕ)
  (num_women : ℕ)
  (num_men : ℕ)
  (num_children : ℕ)
  (total_time : ℕ)
  (h1 : time_woman = 50)
  (h2 : time_man = 15)
  (h3 : num_women = 3)
  (h4 : num_men = 2)
  (h5 : num_children = 3)
  (h6 : total_time = 255)
  (h7 : total_time = time_woman * num_women + time_man * num_men + num_children * (total_time - time_woman * num_women - time_man * num_men) / num_children) :
  (total_time - time_woman * num_women - time_man * num_men) / num_children = 25 := by
  sorry

end time_to_cut_kids_hair_l3034_303438


namespace point_coordinates_theorem_l3034_303421

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between a point and the y-axis -/
def distanceToYAxis (p : Point) : ℝ := |p.x|

/-- The distance between a point and the x-axis -/
def distanceToXAxis (p : Point) : ℝ := |p.y|

/-- Predicate to check if a point is left of the y-axis -/
def isLeftOfYAxis (p : Point) : Prop := p.x < 0

theorem point_coordinates_theorem (B : Point) 
  (h1 : isLeftOfYAxis B)
  (h2 : distanceToXAxis B = 4)
  (h3 : distanceToYAxis B = 5) :
  (B.x = -5 ∧ B.y = 4) ∨ (B.x = -5 ∧ B.y = -4) := by
  sorry

end point_coordinates_theorem_l3034_303421


namespace max_sum_removed_numbers_l3034_303400

theorem max_sum_removed_numbers (n : ℕ) (m k : ℕ) 
  (h1 : n > 2) 
  (h2 : 1 < m ∧ m < n) 
  (h3 : 1 < k ∧ k < n) 
  (h4 : (n * (n + 1) / 2 - m - k) / (n - 2) = 17) :
  m + k ≤ 51 ∧ ∃ (m' k' : ℕ), 1 < m' ∧ m' < n ∧ 1 < k' ∧ k' < n ∧ m' + k' = 51 := by
  sorry

#check max_sum_removed_numbers

end max_sum_removed_numbers_l3034_303400


namespace distance_polar_point_to_circle_center_distance_specific_point_to_specific_circle_l3034_303490

/-- The distance between a point in polar coordinates and the center of a circle defined by a polar equation --/
theorem distance_polar_point_to_circle_center 
  (r : ℝ) (θ : ℝ) (circle_eq : ℝ → ℝ → Prop) : Prop :=
  let p_rect := (r * Real.cos θ, r * Real.sin θ)
  let circle_center := (1, 0)
  Real.sqrt ((p_rect.1 - circle_center.1)^2 + (p_rect.2 - circle_center.2)^2) = Real.sqrt 3

/-- The main theorem to be proved --/
theorem distance_specific_point_to_specific_circle : 
  distance_polar_point_to_circle_center 2 (Real.pi / 3) (fun ρ θ ↦ ρ = 2 * Real.cos θ) :=
sorry

end distance_polar_point_to_circle_center_distance_specific_point_to_specific_circle_l3034_303490


namespace lake_width_correct_l3034_303441

/-- The width of the lake in miles -/
def lake_width : ℝ := 60

/-- The speed of the faster boat in miles per hour -/
def fast_boat_speed : ℝ := 30

/-- The speed of the slower boat in miles per hour -/
def slow_boat_speed : ℝ := 12

/-- The time difference in hours between the arrivals of the two boats -/
def time_difference : ℝ := 3

/-- Theorem stating that the lake width is correct given the boat speeds and time difference -/
theorem lake_width_correct :
  lake_width / slow_boat_speed = lake_width / fast_boat_speed + time_difference :=
by sorry

end lake_width_correct_l3034_303441


namespace floor_sqrt_27_squared_l3034_303442

theorem floor_sqrt_27_squared : ⌊Real.sqrt 27⌋^2 = 25 := by
  sorry

end floor_sqrt_27_squared_l3034_303442


namespace line_through_C_parallel_to_AB_area_of_triangle_OMN_l3034_303404

-- Define the points
def A : ℝ × ℝ := (1, 4)
def B : ℝ × ℝ := (3, 2)
def C : ℝ × ℝ := (1, 1)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + y - 2 = 0

-- Define the perpendicular bisector
def perp_bisector (x y : ℝ) : Prop := x - y + 1 = 0

-- Define points M and N
def M : ℝ × ℝ := (-1, 0)
def N : ℝ × ℝ := (0, 1)

-- Theorem for the line equation
theorem line_through_C_parallel_to_AB :
  line_equation C.1 C.2 ∧
  (B.2 - A.2) * (C.1 - A.1) = (C.2 - A.2) * (B.1 - A.1) := by sorry

-- Theorem for the area of triangle OMN
theorem area_of_triangle_OMN :
  perp_bisector ((A.1 + B.1) / 2) ((A.2 + B.2) / 2) ∧
  M.2 = 0 ∧ N.1 = 0 ∧
  (1 / 2 : ℝ) * abs M.1 * abs N.2 = (1 / 2 : ℝ) := by sorry

end line_through_C_parallel_to_AB_area_of_triangle_OMN_l3034_303404


namespace math_city_intersections_l3034_303434

/-- Represents a city layout with streets and intersections -/
structure CityLayout where
  num_streets : ℕ
  num_nonintersecting_pairs : ℕ

/-- Calculates the number of intersections in a city layout -/
def num_intersections (layout : CityLayout) : ℕ :=
  (layout.num_streets.choose 2) - layout.num_nonintersecting_pairs

/-- Theorem: In a city with 10 streets and 3 non-intersecting pairs, there are 42 intersections -/
theorem math_city_intersections :
  let layout : CityLayout := ⟨10, 3⟩
  num_intersections layout = 42 := by
  sorry

#eval num_intersections ⟨10, 3⟩

end math_city_intersections_l3034_303434


namespace smaller_number_proof_l3034_303472

theorem smaller_number_proof (x y : ℝ) : 
  x + y = 84 ∧ y = x + 12 → x = 36 := by
  sorry

end smaller_number_proof_l3034_303472


namespace intersection_A_B_l3034_303465

def A : Set ℤ := {-3, -1, 2, 6}
def B : Set ℤ := {x | x > 0}

theorem intersection_A_B : A ∩ B = {2, 6} := by sorry

end intersection_A_B_l3034_303465


namespace middle_term_binomial_coefficient_l3034_303481

theorem middle_term_binomial_coefficient 
  (n : ℕ) 
  (x : ℝ) 
  (h1 : x ≠ 0) 
  (h2 : 2^(n-1) = 1024) : 
  Nat.choose n ((n-1)/2) = 462 := by
  sorry

end middle_term_binomial_coefficient_l3034_303481


namespace polygon_properties_l3034_303419

/-- Proves that a polygon with n sides, where the sum of interior angles is 5 times
    the sum of exterior angles, has 12 sides and 54 diagonals. -/
theorem polygon_properties (n : ℕ) : 
  (n - 2) * 180 = 5 * 360 → 
  n = 12 ∧ 
  n * (n - 3) / 2 = 54 := by
sorry

end polygon_properties_l3034_303419


namespace function_range_and_inequality_l3034_303436

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp x * Real.sin x

theorem function_range_and_inequality (e : ℝ) (π : ℝ) :
  (∀ x ∈ Set.Icc 0 (π / 2), f x ∈ Set.Icc 0 1) ∧
  (∃ k : ℝ, k = Real.exp (π / 2) / (π / 2 - 1) ∧
    ∀ x ∈ Set.Icc 0 (π / 2), f x ≥ k * (x - 1) * (1 - Real.sin x) ∧
    ∀ k' > k, ∃ x ∈ Set.Icc 0 (π / 2), f x < k' * (x - 1) * (1 - Real.sin x)) :=
by sorry

end function_range_and_inequality_l3034_303436


namespace probability_red_then_blue_l3034_303475

def red_marbles : ℕ := 4
def white_marbles : ℕ := 5
def blue_marbles : ℕ := 3

def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles

theorem probability_red_then_blue : 
  (red_marbles : ℚ) / total_marbles * blue_marbles / (total_marbles - 1) = 1 / 11 := by
  sorry

end probability_red_then_blue_l3034_303475


namespace bowling_ball_weight_proof_l3034_303494

/-- The weight of a single kayak in pounds -/
def kayak_weight : ℚ := 32

/-- The number of kayaks -/
def num_kayaks : ℕ := 4

/-- The number of bowling balls -/
def num_bowling_balls : ℕ := 9

/-- The weight of a single bowling ball in pounds -/
def bowling_ball_weight : ℚ := 128 / 9

theorem bowling_ball_weight_proof :
  num_bowling_balls * bowling_ball_weight = num_kayaks * kayak_weight :=
by sorry

end bowling_ball_weight_proof_l3034_303494


namespace geometric_sequence_sum_l3034_303487

/-- A geometric sequence with first term a and common ratio q -/
def geometric_sequence (a q : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => geometric_sequence a q n * q

theorem geometric_sequence_sum (a q : ℝ) :
  let seq := geometric_sequence a q
  (seq 0 + seq 1 = 2) →
  (seq 4 + seq 5 = 4) →
  (seq 8 + seq 9 = 8) := by
  sorry

end geometric_sequence_sum_l3034_303487


namespace product_of_sum_and_sum_of_cubes_l3034_303440

theorem product_of_sum_and_sum_of_cubes (a b : ℝ) 
  (h1 : a + b = 5) 
  (h2 : a^3 + b^3 = 35) : 
  a * b = 6 := by
  sorry

end product_of_sum_and_sum_of_cubes_l3034_303440


namespace necessary_condition_range_l3034_303491

theorem necessary_condition_range (a : ℝ) : 
  (∀ x : ℝ, x < a + 2 → x ≤ 2) → a ≤ 0 := by sorry

end necessary_condition_range_l3034_303491


namespace tangent_circle_radius_l3034_303473

/-- A circle tangent to the x-axis, y-axis, and hypotenuse of a 45°-45°-90° triangle --/
structure TangentCircle where
  /-- The radius of the circle --/
  radius : ℝ
  /-- The circle is tangent to the x-axis --/
  tangent_x : True
  /-- The circle is tangent to the y-axis --/
  tangent_y : True
  /-- The circle is tangent to the hypotenuse of a 45°-45°-90° triangle --/
  tangent_hypotenuse : True
  /-- The length of a leg of the 45°-45°-90° triangle is 2 --/
  triangle_leg : ℝ := 2

/-- The radius of the TangentCircle is equal to 2 + √2 --/
theorem tangent_circle_radius (c : TangentCircle) : c.radius = 2 + Real.sqrt 2 := by
  sorry

end tangent_circle_radius_l3034_303473


namespace range_of_a_l3034_303427

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 0}
def B (a : ℝ) : Set ℝ := {1, 3, a}

-- State the theorem
theorem range_of_a (a : ℝ) : A ∩ B a ≠ ∅ → a ∈ A := by
  sorry

end range_of_a_l3034_303427


namespace cube_sum_theorem_l3034_303468

theorem cube_sum_theorem (p q r : ℝ) 
  (h1 : p + q + r = 4)
  (h2 : p * q + q * r + r * p = 6)
  (h3 : p * q * r = -8) :
  p^3 + q^3 + r^3 = 64 := by
  sorry

end cube_sum_theorem_l3034_303468


namespace composite_expressions_l3034_303459

theorem composite_expressions (p : ℕ) (hp : Nat.Prime p) : 
  (¬ Nat.Prime (p^2 + 35)) ∧ (¬ Nat.Prime (p^2 + 55)) := by
  sorry

end composite_expressions_l3034_303459


namespace food_percentage_is_ten_percent_l3034_303407

-- Define the total amount spent
variable (T : ℝ)

-- Define the percentage spent on food
variable (F : ℝ)

-- Define the conditions
axiom clothing_percentage : 0.60 * T = T * 0.60
axiom other_items_percentage : 0.30 * T = T * 0.30
axiom food_percentage : F * T = T - (0.60 * T + 0.30 * T)

axiom tax_clothing : 0.04 * (0.60 * T) = 0.024 * T
axiom tax_other_items : 0.08 * (0.30 * T) = 0.024 * T
axiom total_tax : 0.048 * T = 0.024 * T + 0.024 * T

-- Theorem to prove
theorem food_percentage_is_ten_percent : F = 0.10 := by
  sorry

end food_percentage_is_ten_percent_l3034_303407


namespace min_pairs_for_flashlight_l3034_303448

/-- Represents the minimum number of pairs to test to guarantee finding a working pair of batteries -/
def min_pairs_to_test (total_batteries : ℕ) (working_batteries : ℕ) : ℕ :=
  total_batteries / 2 - working_batteries / 2 + 1

/-- Theorem stating the minimum number of pairs to test for the given problem -/
theorem min_pairs_for_flashlight :
  min_pairs_to_test 8 4 = 4 :=
by sorry

end min_pairs_for_flashlight_l3034_303448


namespace cubic_function_b_value_l3034_303466

/-- A cubic function f(x) = ax³ + bx² + cx + d with specific properties -/
def cubic_function (a b c d : ℝ) : ℝ → ℝ := fun x ↦ a * x^3 + b * x^2 + c * x + d

theorem cubic_function_b_value (a b c d : ℝ) :
  (cubic_function a b c d (-1) = 0) →
  (cubic_function a b c d 1 = 0) →
  (cubic_function a b c d 0 = 2) →
  b = -2 := by
  sorry

end cubic_function_b_value_l3034_303466


namespace algae_growth_theorem_l3034_303474

/-- The time (in hours) for an algae population to grow from 200 to 145,800 cells, tripling every 3 hours. -/
def algae_growth_time : ℕ :=
  18

theorem algae_growth_theorem (initial_population : ℕ) (final_population : ℕ) (growth_factor : ℕ) (growth_interval : ℕ) :
  initial_population = 200 →
  final_population = 145800 →
  growth_factor = 3 →
  growth_interval = 3 →
  (growth_factor ^ (algae_growth_time / growth_interval)) * initial_population = final_population :=
by
  sorry

#check algae_growth_theorem

end algae_growth_theorem_l3034_303474


namespace card_probability_l3034_303451

theorem card_probability (diamonds hearts : ℕ) (a : ℕ) :
  diamonds = 3 →
  hearts = 2 →
  (a : ℚ) / (a + diamonds + hearts) = 1 / 2 →
  a = 5 := by
  sorry

end card_probability_l3034_303451


namespace complex_fraction_equals_seven_plus_i_l3034_303425

theorem complex_fraction_equals_seven_plus_i :
  let i : ℂ := Complex.I
  (1 + i) * (3 + 4*i) / i = 7 + i := by sorry

end complex_fraction_equals_seven_plus_i_l3034_303425


namespace greatest_prime_factor_of_sum_l3034_303408

def double_factorial (n : ℕ) : ℕ := 
  if n ≤ 1 then 1 else n * double_factorial (n - 2)

theorem greatest_prime_factor_of_sum (n : ℕ) : 
  ∃ p : ℕ, Nat.Prime p ∧ 
    p = Nat.gcd (double_factorial 22 + double_factorial 20) p ∧ 
    ∀ q : ℕ, Nat.Prime q → q ∣ (double_factorial 22 + double_factorial 20) → q ≤ p :=
by
  -- The proof goes here
  sorry

#check greatest_prime_factor_of_sum

end greatest_prime_factor_of_sum_l3034_303408


namespace amara_clothing_proof_l3034_303411

def initial_clothing (donated_first : ℕ) (donated_second : ℕ) (thrown_away : ℕ) (remaining : ℕ) : ℕ :=
  remaining + donated_first + donated_second + thrown_away

theorem amara_clothing_proof :
  let donated_first := 5
  let donated_second := 3 * donated_first
  let thrown_away := 15
  let remaining := 65
  initial_clothing donated_first donated_second thrown_away remaining = 100 := by
  sorry

end amara_clothing_proof_l3034_303411


namespace josh_marbles_l3034_303480

/-- The number of marbles Josh has -/
def total_marbles (blue red yellow : ℕ) : ℕ := blue + red + yellow

/-- The problem statement -/
theorem josh_marbles : 
  ∀ (blue red yellow : ℕ),
  blue = 3 * red →
  red = 14 →
  yellow = 29 →
  total_marbles blue red yellow = 85 := by
sorry

end josh_marbles_l3034_303480


namespace sum_of_products_l3034_303415

theorem sum_of_products (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x * y = 20) (hxz : x * z = 60) (hyz : y * z = 90) :
  x + y + z = 11 * Real.sqrt 30 / 3 := by
sorry

end sum_of_products_l3034_303415


namespace platform_length_calculation_l3034_303492

/-- Calculates the length of a platform given train length and crossing times -/
theorem platform_length_calculation (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) :
  train_length = 300 →
  platform_time = 33 →
  pole_time = 18 →
  ∃ (platform_length : ℝ),
    platform_length = platform_time * (train_length / pole_time) - train_length ∧
    platform_length = 250 := by
  sorry

end platform_length_calculation_l3034_303492


namespace dannys_collection_l3034_303418

theorem dannys_collection (initial_wrappers initial_caps found_wrappers found_caps : ℕ) 
  (h1 : initial_wrappers = 67)
  (h2 : initial_caps = 35)
  (h3 : found_wrappers = 18)
  (h4 : found_caps = 15) :
  (initial_wrappers + found_wrappers) - (initial_caps + found_caps) = 35 := by
  sorry

end dannys_collection_l3034_303418


namespace small_triangles_to_large_triangle_area_ratio_l3034_303413

theorem small_triangles_to_large_triangle_area_ratio :
  let small_side_length : ℝ := 2
  let small_triangle_count : ℕ := 8
  let small_triangle_perimeter : ℝ := 3 * small_side_length
  let large_triangle_perimeter : ℝ := small_triangle_count * small_triangle_perimeter
  let large_side_length : ℝ := large_triangle_perimeter / 3
  let triangle_area (side : ℝ) : ℝ := (Real.sqrt 3 / 4) * side^2
  let small_triangle_area : ℝ := triangle_area small_side_length
  let large_triangle_area : ℝ := triangle_area large_side_length
  (small_triangle_count * small_triangle_area) / large_triangle_area = 1 / 8 := by
  sorry

#check small_triangles_to_large_triangle_area_ratio

end small_triangles_to_large_triangle_area_ratio_l3034_303413


namespace door_blocked_time_l3034_303410

/-- Represents a clock with a door near its center -/
structure Clock :=
  (door_blocked_by_minute_hand : ℕ → Bool)
  (door_blocked_by_hour_hand : ℕ → Bool)

/-- The duration of a day in minutes -/
def day_minutes : ℕ := 24 * 60

/-- Checks if the door is blocked at a given minute -/
def is_door_blocked (clock : Clock) (minute : ℕ) : Bool :=
  clock.door_blocked_by_minute_hand minute ∨ clock.door_blocked_by_hour_hand minute

/-- Counts the number of minutes the door is blocked in a day -/
def blocked_minutes (clock : Clock) : ℕ :=
  (List.range day_minutes).filter (is_door_blocked clock) |>.length

/-- The theorem stating that the door is blocked for 498 minutes per day -/
theorem door_blocked_time (clock : Clock) 
  (h1 : ∀ (hour : ℕ) (minute : ℕ), hour < 24 → minute < 60 → 
    clock.door_blocked_by_minute_hand (hour * 60 + minute) = (9 ≤ minute ∧ minute < 21))
  (h2 : ∀ (minute : ℕ), minute < day_minutes → 
    clock.door_blocked_by_hour_hand minute = 
      ((108 ≤ minute % 720 ∧ minute % 720 < 252) ∨ 
       (828 ≤ minute % 720 ∧ minute % 720 < 972))) :
  blocked_minutes clock = 498 := by
  sorry


end door_blocked_time_l3034_303410


namespace joe_game_buying_duration_l3034_303426

/-- Calculates the number of months before running out of money given initial amount, monthly spending, and monthly income. -/
def monthsBeforeBroke (initialAmount : ℕ) (monthlySpending : ℕ) (monthlyIncome : ℕ) : ℕ :=
  initialAmount / (monthlySpending - monthlyIncome)

/-- Theorem stating that given the specific conditions, Joe can buy and sell games for 12 months before running out of money. -/
theorem joe_game_buying_duration :
  monthsBeforeBroke 240 50 30 = 12 := by
  sorry

end joe_game_buying_duration_l3034_303426


namespace expand_product_l3034_303424

theorem expand_product (x : ℝ) (h : x ≠ 0) :
  (3 / 7) * ((7 / x^2) + 6 * x^3 - 2) = 3 / x^2 + (18 * x^3) / 7 - 6 / 7 := by
  sorry

end expand_product_l3034_303424


namespace largest_less_than_point_seven_l3034_303455

def numbers : Set ℚ := {8/10, 1/2, 9/10, 1/3}

theorem largest_less_than_point_seven :
  (∃ (x : ℚ), x ∈ numbers ∧ x < 7/10 ∧ ∀ (y : ℚ), y ∈ numbers ∧ y < 7/10 → y ≤ x) ∧
  (∀ (z : ℚ), z ∈ numbers ∧ z < 7/10 → z ≤ 1/2) :=
by sorry

end largest_less_than_point_seven_l3034_303455


namespace five_workers_completion_time_l3034_303444

/-- The productivity rates of five workers -/
structure WorkerRates where
  x₁ : ℝ
  x₂ : ℝ
  x₃ : ℝ
  x₄ : ℝ
  x₅ : ℝ

/-- The total amount of work to be done -/
def total_work : ℝ → ℝ := id

theorem five_workers_completion_time 
  (rates : WorkerRates) 
  (y : ℝ) 
  (h₁ : rates.x₁ + rates.x₂ + rates.x₃ = y / 327.5)
  (h₂ : rates.x₁ + rates.x₃ + rates.x₅ = y / 5)
  (h₃ : rates.x₁ + rates.x₃ + rates.x₄ = y / 6)
  (h₄ : rates.x₂ + rates.x₄ + rates.x₅ = y / 4) :
  y / (rates.x₁ + rates.x₂ + rates.x₃ + rates.x₄ + rates.x₅) = 3 := by
  sorry

end five_workers_completion_time_l3034_303444


namespace beka_miles_l3034_303469

/-- The number of miles Jackson flew -/
def jackson_miles : ℕ := 563

/-- The additional miles Beka flew compared to Jackson -/
def additional_miles : ℕ := 310

/-- Theorem: Given the conditions, Beka flew 873 miles -/
theorem beka_miles : jackson_miles + additional_miles = 873 := by
  sorry

end beka_miles_l3034_303469


namespace paper_towel_cost_l3034_303437

theorem paper_towel_cost (case_price : ℝ) (num_rolls : ℕ) (savings_percent : ℝ) :
  case_price = 9 →
  num_rolls = 12 →
  savings_percent = 25 →
  ∃ (individual_price : ℝ),
    case_price = (1 - savings_percent / 100) * (num_rolls * individual_price) ∧
    individual_price = 1 := by
  sorry

end paper_towel_cost_l3034_303437


namespace vector_inequality_iff_positive_dot_product_l3034_303431

variable (n : ℕ)
variable (a b : Fin n → ℝ)

theorem vector_inequality_iff_positive_dot_product :
  ‖a + b‖ > ‖a - b‖ ↔ a • b > 0 := by sorry

end vector_inequality_iff_positive_dot_product_l3034_303431


namespace railway_distances_l3034_303439

theorem railway_distances (total_distance : ℝ) 
  (moscow_mozhaysk_ratio : ℝ) (mozhaysk_vyazma_ratio : ℝ) 
  (vyazma_smolensk_ratio : ℝ) :
  total_distance = 415 ∧ 
  moscow_mozhaysk_ratio = 7/9 ∧ 
  mozhaysk_vyazma_ratio = 27/35 →
  ∃ (moscow_mozhaysk vyazma_smolensk mozhaysk_vyazma : ℝ),
    moscow_mozhaysk = 105 ∧
    mozhaysk_vyazma = 135 ∧
    vyazma_smolensk = 175 ∧
    moscow_mozhaysk + mozhaysk_vyazma + vyazma_smolensk = total_distance ∧
    moscow_mozhaysk = moscow_mozhaysk_ratio * mozhaysk_vyazma ∧
    mozhaysk_vyazma = mozhaysk_vyazma_ratio * vyazma_smolensk :=
by sorry

end railway_distances_l3034_303439


namespace mangoes_in_basket_B_mangoes_in_basket_B_is_30_l3034_303464

theorem mangoes_in_basket_B (total_baskets : ℕ) (avg_fruits : ℕ) 
  (apples_A : ℕ) (peaches_C : ℕ) (pears_D : ℕ) (bananas_E : ℕ) : ℕ :=
  let total_fruits := total_baskets * avg_fruits
  let accounted_fruits := apples_A + peaches_C + pears_D + bananas_E
  total_fruits - accounted_fruits

#check mangoes_in_basket_B 5 25 15 20 25 35 = 30

theorem mangoes_in_basket_B_is_30 :
  mangoes_in_basket_B 5 25 15 20 25 35 = 30 := by
  sorry

end mangoes_in_basket_B_mangoes_in_basket_B_is_30_l3034_303464


namespace two_tangent_lines_l3034_303454

/-- The number of tangent lines to a circle passing through a point -/
def num_tangent_lines (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : ℕ :=
  sorry

/-- The theorem stating that there are exactly two tangent lines from (2,3) to x^2 + y^2 = 4 -/
theorem two_tangent_lines : num_tangent_lines (0, 0) 2 (2, 3) = 2 := by
  sorry

end two_tangent_lines_l3034_303454


namespace profit_at_10_yuan_increase_profit_maximum_at_5_yuan_increase_l3034_303429

/-- Represents the product pricing model -/
structure PricingModel where
  currentPrice : ℝ
  weeklySales : ℝ
  salesDecrease : ℝ
  costPrice : ℝ

/-- Calculates the profit for a given price increase -/
def profit (model : PricingModel) (priceIncrease : ℝ) : ℝ :=
  (model.currentPrice + priceIncrease - model.costPrice) *
  (model.weeklySales - model.salesDecrease * priceIncrease)

/-- The pricing model for the given problem -/
def givenModel : PricingModel :=
  { currentPrice := 60
    weeklySales := 300
    salesDecrease := 10
    costPrice := 40 }

/-- Theorem: A price increase of 10 yuan results in a weekly profit of 6000 yuan -/
theorem profit_at_10_yuan_increase (ε : ℝ) :
  |profit givenModel 10 - 6000| < ε := by sorry

/-- Theorem: A price increase of 5 yuan maximizes the weekly profit -/
theorem profit_maximum_at_5_yuan_increase :
  ∀ x, profit givenModel 5 ≥ profit givenModel x := by sorry

end profit_at_10_yuan_increase_profit_maximum_at_5_yuan_increase_l3034_303429


namespace mean_of_pencil_sharpening_counts_l3034_303422

def pencil_sharpening_counts : List ℕ := [13, 8, 13, 21, 7, 23]

theorem mean_of_pencil_sharpening_counts :
  (pencil_sharpening_counts.sum : ℚ) / pencil_sharpening_counts.length = 85/6 := by
  sorry

end mean_of_pencil_sharpening_counts_l3034_303422
