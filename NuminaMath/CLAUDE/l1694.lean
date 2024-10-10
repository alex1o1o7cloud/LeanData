import Mathlib

namespace undeclared_major_fraction_l1694_169436

/-- The fraction of students who have not declared a major among second- and third-year students -/
theorem undeclared_major_fraction :
  let total_students : ℚ := 1
  let first_year_students : ℚ := 1/3
  let second_year_students : ℚ := 1/3
  let third_year_students : ℚ := 1/3
  let first_year_undeclared : ℚ := 4/5 * first_year_students
  let second_year_declared : ℚ := 1/2 * (first_year_students - first_year_undeclared)
  let second_year_undeclared : ℚ := second_year_students - second_year_declared
  let third_year_undeclared : ℚ := 1/4 * third_year_students
  (second_year_undeclared + third_year_undeclared) / total_students = 23/60 := by
  sorry

end undeclared_major_fraction_l1694_169436


namespace paper_distribution_l1694_169453

theorem paper_distribution (total_students : ℕ) (total_sheets : ℕ) (leftover_sheets : ℕ)
  (h1 : total_students = 24)
  (h2 : total_sheets = 50)
  (h3 : leftover_sheets = 2)
  (h4 : ∃ (girls : ℕ), girls * 3 = total_students) :
  ∃ (girls : ℕ), girls * 3 = total_students ∧ 
    (total_sheets - leftover_sheets) / girls = 6 := by
  sorry

end paper_distribution_l1694_169453


namespace rational_inequality_solution_l1694_169414

theorem rational_inequality_solution (x : ℝ) :
  (x ≠ -1 ∧ x ≠ 2) →
  ((x^2 + 3*x - 4) / (x^2 - x - 2) > 0 ↔ x > 2 ∨ x < -4) :=
by sorry

end rational_inequality_solution_l1694_169414


namespace estimate_sqrt_expression_l1694_169430

theorem estimate_sqrt_expression :
  6 < (Real.sqrt 54 + 2 * Real.sqrt 3) * Real.sqrt (1/3) ∧
  (Real.sqrt 54 + 2 * Real.sqrt 3) * Real.sqrt (1/3) < 7 := by
  sorry

end estimate_sqrt_expression_l1694_169430


namespace parallel_vectors_x_value_l1694_169438

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (4, -2)
  let b : ℝ × ℝ := (x, 5)
  parallel a b → x = -10 :=
by
  sorry

end parallel_vectors_x_value_l1694_169438


namespace max_substitutions_is_fifty_l1694_169495

/-- A type representing a fifth-degree polynomial -/
def FifthDegreePolynomial := ℕ → ℕ

/-- Given a list of ten fifth-degree polynomials, returns the maximum number of consecutive
    natural numbers that can be substituted to produce an arithmetic progression -/
def max_consecutive_substitutions (polynomials : List FifthDegreePolynomial) : ℕ :=
  sorry

/-- The main theorem stating that the maximum number of consecutive substitutions is 50 -/
theorem max_substitutions_is_fifty :
  ∀ (polynomials : List FifthDegreePolynomial),
    polynomials.length = 10 →
    max_consecutive_substitutions polynomials = 50 :=
  sorry

end max_substitutions_is_fifty_l1694_169495


namespace sufficient_not_necessary_condition_l1694_169469

theorem sufficient_not_necessary_condition (x : ℝ) :
  (∀ x, 0 < x ∧ x < 1 → 0 < x^2 ∧ x^2 < 1) ∧
  (∃ x, 0 < x^2 ∧ x^2 < 1 ∧ ¬(0 < x ∧ x < 1)) := by
  sorry

end sufficient_not_necessary_condition_l1694_169469


namespace triangle_area_l1694_169434

/-- Given a triangle with perimeter 36 and inradius 2.5, prove its area is 45 -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) 
    (h1 : perimeter = 36) 
    (h2 : inradius = 2.5) 
    (h3 : area = inradius * (perimeter / 2)) : 
  area = 45 := by
  sorry

end triangle_area_l1694_169434


namespace factorial_fraction_equals_one_l1694_169468

theorem factorial_fraction_equals_one : (4 * Nat.factorial 7 + 28 * Nat.factorial 6) / Nat.factorial 8 = 1 := by
  sorry

end factorial_fraction_equals_one_l1694_169468


namespace smallest_integer_bound_l1694_169418

theorem smallest_integer_bound (a b c d : ℤ) : 
  a < b ∧ b < c ∧ c < d  -- Four different integers
  → d = 90  -- Largest integer is 90
  → (a + b + c + d) / 4 = 68  -- Average is 68
  → a ≥ 5  -- Smallest integer is at least 5
:= by sorry

end smallest_integer_bound_l1694_169418


namespace quadratic_root_range_l1694_169402

theorem quadratic_root_range (a : ℝ) (x₁ x₂ : ℝ) : 
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ 
    a * x₁^2 + (a + 2) * x₁ + 9 * a = 0 ∧
    a * x₂^2 + (a + 2) * x₂ + 9 * a = 0 ∧
    x₁ < 2 ∧ 2 < x₂) →
  -4/15 < a ∧ a < 0 :=
by sorry

end quadratic_root_range_l1694_169402


namespace sum_g_11_and_neg_11_l1694_169480

/-- Given a function g(x) = px^8 + qx^6 - rx^4 + sx^2 + 5, 
    if g(11) = 7, then g(11) + g(-11) = 14 -/
theorem sum_g_11_and_neg_11 (p q r s : ℝ) : 
  let g : ℝ → ℝ := λ x => p * x^8 + q * x^6 - r * x^4 + s * x^2 + 5
  g 11 = 7 → g 11 + g (-11) = 14 := by
  sorry

end sum_g_11_and_neg_11_l1694_169480


namespace bike_distance_proof_l1694_169424

theorem bike_distance_proof (x t : ℝ) 
  (h1 : (x + 1) * (3 * t / 4) = x * t)
  (h2 : (x - 1) * (t + 3) = x * t) :
  x * t = 36 := by
  sorry

end bike_distance_proof_l1694_169424


namespace triangle_area_is_four_thirds_l1694_169498

-- Define the line m: 3x - y + 2 = 0
def line_m (x y : ℝ) : Prop := 3 * x - y + 2 = 0

-- Define the symmetric line l with respect to the x-axis
def line_l (x y : ℝ) : Prop := 3 * x + y + 2 = 0

-- Define the y-axis
def y_axis (x : ℝ) : Prop := x = 0

-- Theorem statement
theorem triangle_area_is_four_thirds :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    line_m x₁ y₁ ∧ y_axis x₁ ∧
    line_m x₂ y₂ ∧ x₂ = -2/3 ∧ y₂ = 0 ∧
    line_l x₃ y₃ ∧ y_axis x₃ ∧
    (1/2 * abs (x₂ * (y₁ - y₃))) = 4/3 :=
sorry

end triangle_area_is_four_thirds_l1694_169498


namespace system_solution_l1694_169435

theorem system_solution (a b c x y z : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : x * y = a) (h2 : y * z = b) (h3 : z * x = c) :
  (x = Real.sqrt (a * c / b) ∨ x = -Real.sqrt (a * c / b)) ∧
  (y = Real.sqrt (a * b / c) ∨ y = -Real.sqrt (a * b / c)) ∧
  (z = Real.sqrt (b * c / a) ∨ z = -Real.sqrt (b * c / a)) := by
  sorry

end system_solution_l1694_169435


namespace track_meet_adults_l1694_169459

theorem track_meet_adults (children : ℕ) (total_seats : ℕ) (empty_seats : ℕ) 
  (h1 : children = 52)
  (h2 : total_seats = 95)
  (h3 : empty_seats = 14) :
  total_seats - empty_seats - children = 29 := by
  sorry

end track_meet_adults_l1694_169459


namespace graph_single_point_implies_d_eq_39_l1694_169429

/-- The equation of the graph -/
def graph_equation (x y d : ℝ) : Prop :=
  3 * x^2 + y^2 + 6 * x - 12 * y + d = 0

/-- The graph consists of a single point -/
def single_point (d : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, graph_equation p.1 p.2 d

/-- If the graph of 3x^2 + y^2 + 6x - 12y + d = 0 consists of a single point, then d = 39 -/
theorem graph_single_point_implies_d_eq_39 : ∀ d : ℝ, single_point d → d = 39 := by
  sorry

end graph_single_point_implies_d_eq_39_l1694_169429


namespace union_of_M_and_N_l1694_169421

def M : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def N : Set ℝ := {x | 2 < x ∧ x ≤ 5}

theorem union_of_M_and_N :
  M ∪ N = {x | -1 ≤ x ∧ x ≤ 5} := by sorry

end union_of_M_and_N_l1694_169421


namespace equalize_foma_ierema_l1694_169405

/-- Represents the amount of gold coins each merchant has -/
structure MerchantWealth where
  foma : ℕ
  ierema : ℕ
  yuliy : ℕ

/-- The conditions of the problem -/
def problem_conditions (w : MerchantWealth) : Prop :=
  (w.foma - 70 = w.ierema + 70) ∧ 
  (w.foma - 40 = w.yuliy)

/-- The theorem to be proved -/
theorem equalize_foma_ierema (w : MerchantWealth) 
  (h : problem_conditions w) : 
  ∃ (x : ℕ), w.foma - x = w.ierema + x ∧ x = 55 := by
  sorry

end equalize_foma_ierema_l1694_169405


namespace intersection_count_is_two_l1694_169456

/-- The number of intersection points between two circles -/
def intersection_count (c1 c2 : ℝ × ℝ → Prop) : ℕ :=
  sorry

/-- First circle: (x - 2.5)² + y² = 6.25 -/
def circle1 (p : ℝ × ℝ) : Prop :=
  (p.1 - 2.5)^2 + p.2^2 = 6.25

/-- Second circle: x² + (y - 5)² = 25 -/
def circle2 (p : ℝ × ℝ) : Prop :=
  p.1^2 + (p.2 - 5)^2 = 25

/-- Theorem stating that the number of intersection points between the two circles is 2 -/
theorem intersection_count_is_two :
  intersection_count circle1 circle2 = 2 := by sorry

end intersection_count_is_two_l1694_169456


namespace marble_bag_problem_l1694_169465

theorem marble_bag_problem :
  ∀ (r b : ℕ),
  r + b > 0 →
  r = (r + b) / 3 →
  (r - 3) = (r + b - 3) / 4 →
  r = (r + b - 2) / 3 →
  r + b = 19 := by
sorry

end marble_bag_problem_l1694_169465


namespace neither_question_correct_percentage_l1694_169422

theorem neither_question_correct_percentage
  (p_first : ℝ)
  (p_second : ℝ)
  (p_both : ℝ)
  (h1 : p_first = 0.63)
  (h2 : p_second = 0.50)
  (h3 : p_both = 0.33)
  : 1 - (p_first + p_second - p_both) = 0.20 := by
  sorry

end neither_question_correct_percentage_l1694_169422


namespace product_of_sums_l1694_169462

theorem product_of_sums (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b + a + b = 3) (hbc : b * c + b + c = 3) (hac : a * c + a + c = 3) :
  (a + 1) * (b + 1) * (c + 1) = 8 := by
sorry

end product_of_sums_l1694_169462


namespace solution_set_when_a_is_3_range_of_a_for_inequality_l1694_169481

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x + a| + |x - 1|

-- Part 1
theorem solution_set_when_a_is_3 :
  {x : ℝ | f 3 x < 6} = Set.Ioo (-8/3) (4/3) := by sorry

-- Part 2
theorem range_of_a_for_inequality :
  ∀ a : ℝ, (∀ x : ℝ, f a x + f a (-x) ≥ 5) ↔ 
  a ∈ Set.Iic (-3/2) ∪ Set.Ici (3/2) := by sorry

end solution_set_when_a_is_3_range_of_a_for_inequality_l1694_169481


namespace domain_of_f_l1694_169428

-- Define the function f
def f (x : ℝ) : ℝ := (x - 5) ^ (1/3) + (x - 7) ^ (1/4)

-- Define the domain of f
def domain_f : Set ℝ := {x : ℝ | x ≥ 7}

-- Theorem statement
theorem domain_of_f : 
  ∀ x : ℝ, x ∈ domain_f ↔ ∃ y : ℝ, f x = y :=
sorry

end domain_of_f_l1694_169428


namespace distance_P_to_xoy_is_3_l1694_169464

/-- The distance from a point to the xOy plane in 3D Cartesian coordinates --/
def distance_to_xoy_plane (p : ℝ × ℝ × ℝ) : ℝ :=
  |p.2.2|

/-- The point P with coordinates (1, -2, 3) --/
def P : ℝ × ℝ × ℝ := (1, -2, 3)

/-- Theorem: The distance from point P(1,-2,3) to the xOy plane is 3 --/
theorem distance_P_to_xoy_is_3 : distance_to_xoy_plane P = 3 := by
  sorry

end distance_P_to_xoy_is_3_l1694_169464


namespace unique_pair_divisibility_l1694_169489

theorem unique_pair_divisibility (a b : ℕ) :
  a > 0 ∧ b > 0 ∧ (b^a ∣ a^b - 1) ↔ a = 3 ∧ b = 2 := by
  sorry

end unique_pair_divisibility_l1694_169489


namespace percentage_problem_l1694_169442

theorem percentage_problem : 
  ∃ x : ℝ, (120 / 100) * x = 1800 → (20 / 100) * x = 300 := by
sorry

end percentage_problem_l1694_169442


namespace door_unlock_problem_l1694_169497

-- Define the number of buttons and the number of buttons to press
def total_buttons : ℕ := 10
def buttons_to_press : ℕ := 3

-- Define the time for each attempt
def time_per_attempt : ℕ := 2

-- Calculate the total number of combinations
def total_combinations : ℕ := Nat.choose total_buttons buttons_to_press

-- Define the maximum time needed (in seconds)
def max_time : ℕ := total_combinations * time_per_attempt

-- Define the average time needed (in seconds)
def avg_time : ℚ := (1 + total_combinations : ℚ) / 2 * time_per_attempt

-- Define the maximum number of attempts in 60 seconds
def max_attempts_in_minute : ℕ := 60 / time_per_attempt

theorem door_unlock_problem :
  (max_time = 240) ∧
  (avg_time = 121) ∧
  (max_attempts_in_minute = 30) ∧
  ((max_attempts_in_minute - 1 : ℚ) / total_combinations = 29 / 120) := by
  sorry

end door_unlock_problem_l1694_169497


namespace smallest_integers_difference_difference_is_27720_l1694_169493

theorem smallest_integers_difference : ℕ → Prop :=
  fun d =>
    ∃ n₁ n₂ : ℕ,
      n₁ > 1 ∧ n₂ > 1 ∧
      n₂ > n₁ ∧
      (∀ k : ℕ, 2 ≤ k → k ≤ 11 → n₁ % k = 1) ∧
      (∀ k : ℕ, 2 ≤ k → k ≤ 11 → n₂ % k = 1) ∧
      (∀ m : ℕ, m > 1 → (∀ k : ℕ, 2 ≤ k → k ≤ 11 → m % k = 1) → m ≥ n₁) ∧
      d = n₂ - n₁

theorem difference_is_27720 : smallest_integers_difference 27720 := by sorry

end smallest_integers_difference_difference_is_27720_l1694_169493


namespace order_of_numbers_l1694_169439

def Ψ : ℤ := -1006

def Ω : ℤ := -1007

def Θ : ℤ := -1008

theorem order_of_numbers : Θ < Ω ∧ Ω < Ψ := by
  sorry

end order_of_numbers_l1694_169439


namespace ratio_problem_l1694_169404

theorem ratio_problem (a b c x y : ℚ) 
  (h1 : a / b = 5 / 4)
  (h2 : b / c = 3 / x)
  (h3 : a / c = y / 4)
  (h4 : x = 1) : y = 15 := by
  sorry

end ratio_problem_l1694_169404


namespace point_on_x_axis_l1694_169471

/-- If a point P with coordinates (m-3, 2+m) lies on the x-axis, then its coordinates are (-5, 0). -/
theorem point_on_x_axis (m : ℝ) :
  (∃ P : ℝ × ℝ, P = (m - 3, 2 + m) ∧ P.2 = 0) →
  (∃ P : ℝ × ℝ, P = (m - 3, 2 + m) ∧ P = (-5, 0)) :=
by sorry

end point_on_x_axis_l1694_169471


namespace sandals_sold_l1694_169482

theorem sandals_sold (sneakers boots total : ℕ) 
  (h1 : sneakers = 2)
  (h2 : boots = 11)
  (h3 : total = 17)
  (h4 : ∃ sandals : ℕ, total = sneakers + sandals + boots) :
  ∃ sandals : ℕ, sandals = 4 ∧ total = sneakers + sandals + boots :=
by
  sorry

end sandals_sold_l1694_169482


namespace units_digit_of_7_to_1000_l1694_169400

theorem units_digit_of_7_to_1000 : (7^1000 : ℕ) % 10 = 1 := by
  sorry

end units_digit_of_7_to_1000_l1694_169400


namespace zebra_crossing_distance_l1694_169458

/-- Given a boulevard with zebra crossing, calculate the distance between stripes --/
theorem zebra_crossing_distance (boulevard_width : ℝ) (stripe_length : ℝ) (gate_distance : ℝ)
  (h1 : boulevard_width = 60)
  (h2 : stripe_length = 65)
  (h3 : gate_distance = 22) :
  (boulevard_width * gate_distance) / stripe_length = 20.31 := by
  sorry

end zebra_crossing_distance_l1694_169458


namespace max_product_constrained_l1694_169463

theorem max_product_constrained (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (hsum : x/3 + y/4 = 1) : 
  x * y ≤ 3 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀/3 + y₀/4 = 1 ∧ x₀ * y₀ = 3 := by
  sorry

end max_product_constrained_l1694_169463


namespace inscribed_quadrilateral_is_rectangle_l1694_169403

-- Define a circle
def Circle : Type := Unit

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define the property of being inscribed in a circle
def inscribed_in_circle (q : Quadrilateral) (c : Circle) : Prop := sorry

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop := sorry

-- Theorem statement
theorem inscribed_quadrilateral_is_rectangle 
  (q : Quadrilateral) (c : Circle) : 
  inscribed_in_circle q c → is_rectangle q := by sorry

end inscribed_quadrilateral_is_rectangle_l1694_169403


namespace pen_collection_problem_l1694_169446

/-- Represents the pen collection problem --/
theorem pen_collection_problem (initial_pens : ℕ) (final_pens : ℕ) (sharon_pens : ℕ) 
  (h1 : initial_pens = 25)
  (h2 : final_pens = 75)
  (h3 : sharon_pens = 19) :
  ∃ (mike_pens : ℕ), 2 * (initial_pens + mike_pens) - sharon_pens = final_pens ∧ mike_pens = 22 := by
  sorry

end pen_collection_problem_l1694_169446


namespace marley_samantha_apple_ratio_l1694_169494

/-- Proves that the ratio of Marley's apples to Samantha's apples is 3:1 -/
theorem marley_samantha_apple_ratio :
  let louis_oranges : ℕ := 5
  let louis_apples : ℕ := 3
  let samantha_oranges : ℕ := 8
  let samantha_apples : ℕ := 7
  let marley_oranges : ℕ := 2 * louis_oranges
  let marley_total_fruits : ℕ := 31
  let marley_apples : ℕ := marley_total_fruits - marley_oranges
  (marley_apples : ℚ) / samantha_apples = 3 / 1 := by
  sorry


end marley_samantha_apple_ratio_l1694_169494


namespace solve_equations_l1694_169415

theorem solve_equations :
  (∀ x : ℝ, 4 * x = 20 → x = 5) ∧
  (∀ x : ℝ, x - 18 = 40 → x = 58) ∧
  (∀ x : ℝ, x / 7 = 12 → x = 84) ∧
  (∀ n : ℝ, 8 * n / 2 = 15 → n = 15 / 4) :=
by sorry

end solve_equations_l1694_169415


namespace video_difference_l1694_169478

/-- The number of videos watched by three friends -/
def total_videos : ℕ := 411

/-- The number of videos watched by Kelsey -/
def kelsey_videos : ℕ := 160

/-- The number of videos watched by Ekon -/
def ekon_videos : ℕ := kelsey_videos - 43

/-- The number of videos watched by Uma -/
def uma_videos : ℕ := total_videos - kelsey_videos - ekon_videos

/-- Ekon watched fewer videos than Uma -/
axiom ekon_less_than_uma : ekon_videos < uma_videos

theorem video_difference : uma_videos - ekon_videos = 17 := by
  sorry

end video_difference_l1694_169478


namespace card_sum_theorem_l1694_169486

theorem card_sum_theorem (a b c d e f g h : ℕ) : 
  (a + b) * (c + d) * (e + f) * (g + h) = 330 → 
  a + b + c + d + e + f + g + h = 21 := by
sorry

end card_sum_theorem_l1694_169486


namespace solve_equation_l1694_169425

theorem solve_equation (x : ℝ) : 2*x - 3*x + 4*x = 150 → x = 50 := by
  sorry

end solve_equation_l1694_169425


namespace cube_root_of_negative_eight_is_negative_two_l1694_169433

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Theorem statement
theorem cube_root_of_negative_eight_is_negative_two :
  cubeRoot (-8) = -2 := by sorry

end cube_root_of_negative_eight_is_negative_two_l1694_169433


namespace initial_bananas_per_child_l1694_169479

/-- Proves that the initial number of bananas per child is 2 -/
theorem initial_bananas_per_child (total_children : ℕ) (absent_children : ℕ) 
  (extra_bananas : ℕ) (h1 : total_children = 610) (h2 : absent_children = 305) 
  (h3 : extra_bananas = 2) : 
  (total_children : ℚ) * (total_children - absent_children) = 
  (total_children - absent_children) * ((total_children - absent_children) + extra_bananas) :=
by sorry

#check initial_bananas_per_child

end initial_bananas_per_child_l1694_169479


namespace max_value_implies_ratio_l1694_169491

/-- Given a function f(x) = 3sin(x) + 4cos(x) that reaches its maximum value at x = θ,
    prove that (sin(2θ) + cos²(θ) + 1) / cos(2θ) = 15/7 -/
theorem max_value_implies_ratio (θ : ℝ) 
  (h : ∀ x, 3 * Real.sin x + 4 * Real.cos x ≤ 3 * Real.sin θ + 4 * Real.cos θ) :
  (Real.sin (2 * θ) + Real.cos θ ^ 2 + 1) / Real.cos (2 * θ) = 15 / 7 := by
  sorry

end max_value_implies_ratio_l1694_169491


namespace parabola_intersection_probability_l1694_169476

/-- Represents the outcome of rolling a fair six-sided die -/
inductive SixSidedDie : Type
  | one | two | three | four | five | six

/-- Represents the outcome of rolling a fair four-sided die (2 to 5) -/
inductive FourSidedDie : Type
  | two | three | four | five

/-- Represents a parabola of the form y = x^2 + ax + b -/
structure Parabola1 where
  a : SixSidedDie
  b : SixSidedDie

/-- Represents a parabola of the form y = x^2 + px^2 + cx + d -/
structure Parabola2 where
  p : FourSidedDie
  c : SixSidedDie
  d : SixSidedDie

/-- Returns true if two parabolas intersect -/
def intersect (p1 : Parabola1) (p2 : Parabola2) : Bool :=
  sorry

/-- Probability that two randomly chosen parabolas intersect -/
def intersection_probability : ℚ :=
  sorry

theorem parabola_intersection_probability :
  intersection_probability = 209 / 216 :=
sorry

end parabola_intersection_probability_l1694_169476


namespace remaining_distance_l1694_169461

theorem remaining_distance (total : ℕ) (monday : ℕ) (tuesday : ℕ) 
  (h1 : total = 8205)
  (h2 : monday = 907)
  (h3 : tuesday = 582) :
  total - (monday + tuesday) = 6716 := by
  sorry

end remaining_distance_l1694_169461


namespace quadratic_factorization_l1694_169499

theorem quadratic_factorization (y : ℝ) : 16 * y^2 - 40 * y + 25 = (4 * y - 5)^2 := by
  sorry

end quadratic_factorization_l1694_169499


namespace hamburger_sales_solution_l1694_169452

/-- Represents the hamburger sales problem. -/
def HamburgerSales (total_goal : ℕ) (price : ℕ) (first_group : ℕ) (remaining : ℕ) : Prop :=
  let total_hamburgers := total_goal / price
  let accounted_for := first_group + remaining
  total_hamburgers - accounted_for = 2

/-- Theorem stating the solution to the hamburger sales problem. -/
theorem hamburger_sales_solution :
  HamburgerSales 50 5 4 4 := by
  sorry

end hamburger_sales_solution_l1694_169452


namespace factorization_sum_l1694_169441

theorem factorization_sum (a b c d e f g h j k : ℤ) :
  (∀ x y : ℝ, 8 * x^4 - 125 * y^4 = (a*x + b*y) * (c*x^2 + d*x*y + e*y^2) * (f*x + g*y) * (h*x^2 + j*x*y + k*y^2)) →
  a + b + c + d + e + f + g + h + j + k = 4 := by
  sorry

end factorization_sum_l1694_169441


namespace profit_reached_l1694_169485

/-- The number of disks in a buying pack -/
def buying_pack : ℕ := 5

/-- The cost of a buying pack in dollars -/
def buying_cost : ℚ := 8

/-- The number of disks in a selling pack -/
def selling_pack : ℕ := 4

/-- The price of a selling pack in dollars -/
def selling_price : ℚ := 10

/-- The target profit in dollars -/
def target_profit : ℚ := 120

/-- The minimum number of disks that must be sold to reach the target profit -/
def disks_to_sell : ℕ := 134

theorem profit_reached :
  let cost_per_disk : ℚ := buying_cost / buying_pack
  let price_per_disk : ℚ := selling_price / selling_pack
  let profit_per_disk : ℚ := price_per_disk - cost_per_disk
  (disks_to_sell : ℚ) * profit_per_disk ≥ target_profit ∧
  ∀ n : ℕ, (n : ℚ) * profit_per_disk ≥ target_profit → n ≥ disks_to_sell :=
by sorry

end profit_reached_l1694_169485


namespace quadratic_roots_properties_l1694_169483

theorem quadratic_roots_properties (a b : ℝ) : 
  a^2 + 5*a + 2 = 0 → 
  b^2 + 5*b + 2 = 0 → 
  a ≠ b →
  (1/a + 1/b = -5/2) ∧ ((a^2 + 7*a) * (b^2 + 7*b) = 32) := by
  sorry

end quadratic_roots_properties_l1694_169483


namespace common_root_divisibility_l1694_169440

theorem common_root_divisibility (a b c : ℤ) (h1 : c ≠ b) 
  (h2 : ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ (c - b) * x^2 + (c - a) * x + (a + b) = 0) : 
  3 ∣ (a + b + 2*c) := by
sorry

end common_root_divisibility_l1694_169440


namespace prism_diagonals_l1694_169410

/-- A rectangular prism with given dimensions -/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The number of diagonals in a rectangular prism -/
def num_diagonals (p : RectangularPrism) : ℕ :=
  12 + 4  -- 12 face diagonals + 4 space diagonals

/-- Theorem: A rectangular prism with dimensions 4, 3, and 5 has 16 diagonals -/
theorem prism_diagonals :
  let p : RectangularPrism := ⟨4, 3, 5⟩
  num_diagonals p = 16 := by
  sorry

end prism_diagonals_l1694_169410


namespace unique_prime_triple_l1694_169450

theorem unique_prime_triple : 
  ∃! (x y z : ℕ), 
    (Nat.Prime x ∧ Nat.Prime y ∧ Nat.Prime z) ∧ 
    (x > y ∧ y > z) ∧
    (Nat.Prime (x - y) ∧ Nat.Prime (y - z) ∧ Nat.Prime (x - z)) ∧
    (x = 7 ∧ y = 5 ∧ z = 2) := by
  sorry

end unique_prime_triple_l1694_169450


namespace baker_problem_l1694_169444

/-- The number of cakes that can be made given the available ingredients and recipe requirements. -/
def num_cakes : ℕ := 49

/-- The number of loaves of bread that can be made given the available ingredients and recipe requirements. -/
def num_bread : ℕ := 30

/-- The amount of flour available (in cups). -/
def flour_available : ℕ := 188

/-- The amount of sugar available (in cups). -/
def sugar_available : ℕ := 113

/-- The amount of flour required for one loaf of bread (in cups). -/
def flour_per_bread : ℕ := 3

/-- The amount of sugar required for one loaf of bread (in cups). -/
def sugar_per_bread : ℚ := 1/2

/-- The amount of flour required for one cake (in cups). -/
def flour_per_cake : ℕ := 2

/-- The amount of sugar required for one cake (in cups). -/
def sugar_per_cake : ℕ := 2

theorem baker_problem :
  (num_bread * flour_per_bread + num_cakes * flour_per_cake = flour_available) ∧
  (num_bread * sugar_per_bread + num_cakes * sugar_per_cake = sugar_available) :=
by sorry

end baker_problem_l1694_169444


namespace binomial_factorial_l1694_169408

theorem binomial_factorial : Nat.factorial (Nat.choose 8 5) = Nat.factorial 56 := by
  sorry

end binomial_factorial_l1694_169408


namespace opposite_of_negative_three_sevenths_l1694_169412

theorem opposite_of_negative_three_sevenths :
  let x : ℚ := -3/7
  let y : ℚ := 3/7
  (∀ a b : ℚ, (a + b = 0 ↔ b = -a)) →
  y = -x :=
by sorry

end opposite_of_negative_three_sevenths_l1694_169412


namespace shaded_region_perimeter_l1694_169477

/-- The perimeter of a region formed by three identical touching circles -/
theorem shaded_region_perimeter (c : ℝ) (θ : ℝ) : 
  c > 0 → θ > 0 → θ < 2 * Real.pi →
  let r := c / (2 * Real.pi)
  let arc_length := θ / (2 * Real.pi) * c
  3 * arc_length = c →
  c = 48 → θ = 2 * Real.pi / 3 →
  3 * arc_length = 48 := by
  sorry

end shaded_region_perimeter_l1694_169477


namespace amy_yard_area_l1694_169409

theorem amy_yard_area :
  ∀ (short_posts long_posts : ℕ) 
    (post_distance : ℝ) 
    (total_posts : ℕ),
  short_posts > 1 →
  long_posts > 1 →
  post_distance > 0 →
  total_posts = 24 →
  long_posts = (3 * short_posts) / 2 →
  total_posts = 2 * short_posts + 2 * long_posts - 4 →
  post_distance = 3 →
  (short_posts - 1 : ℝ) * post_distance * ((long_posts - 1 : ℝ) * post_distance) = 189 :=
by sorry

end amy_yard_area_l1694_169409


namespace find_B_l1694_169457

theorem find_B (x y A : ℕ) (hx : x > 1) (hy : y > 1) (hxy : x > y) 
  (heq : x * y = x + y + A) : x / y = 12 := by
  sorry

end find_B_l1694_169457


namespace bus_problem_l1694_169416

theorem bus_problem (initial : ℕ) (first_on : ℕ) (second_off : ℕ) (third_off : ℕ) (third_on : ℕ) (final : ℕ) :
  initial = 18 →
  first_on = 5 →
  second_off = 4 →
  third_off = 3 →
  third_on = 5 →
  final = 25 →
  ∃ (second_on : ℕ), 
    final = initial + first_on + second_on - second_off - third_off + third_on ∧
    second_on = 4 := by
  sorry

end bus_problem_l1694_169416


namespace select_two_from_nine_l1694_169431

theorem select_two_from_nine (n : ℕ) (k : ℕ) : n = 9 ∧ k = 2 → Nat.choose n k = 36 := by
  sorry

end select_two_from_nine_l1694_169431


namespace equation_solution_l1694_169484

theorem equation_solution (y : ℝ) : 
  (y / 6) / 3 = 6 / (y / 3) → y = 18 ∨ y = -18 := by
  sorry

end equation_solution_l1694_169484


namespace tasty_pair_iff_isogonal_conjugate_exists_tasty_pair_for_both_triangles_l1694_169474

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the properties of the triangle
def isAcute (t : Triangle) : Prop := sorry

def isScalene (t : Triangle) : Prop := sorry

-- Define the tasty pair property
def isTastyPair (t : Triangle) (P Q : Point) : Prop := sorry

-- Define isogonal conjugates
def isIsogonalConjugate (t : Triangle) (P Q : Point) : Prop := sorry

-- Define the reflection of a triangle about its circumcenter
def reflectTriangle (t : Triangle) : Triangle := sorry

-- Main theorem
theorem tasty_pair_iff_isogonal_conjugate (t : Triangle) (h1 : isAcute t) (h2 : isScalene t) :
  ∀ P Q : Point, isTastyPair t P Q ↔ isIsogonalConjugate t P Q :=
sorry

-- Additional theorem
theorem exists_tasty_pair_for_both_triangles (t : Triangle) (h1 : isAcute t) (h2 : isScalene t) :
  ∃ P Q : Point, isTastyPair t P Q ∧ isTastyPair (reflectTriangle t) P Q :=
sorry

end tasty_pair_iff_isogonal_conjugate_exists_tasty_pair_for_both_triangles_l1694_169474


namespace min_value_quadratic_l1694_169443

theorem min_value_quadratic (x y : ℝ) : 
  y = x^2 + 16*x + 10 → (∀ z, y ≤ z → z = x^2 + 16*x + 10) → y = -54 :=
by
  sorry

end min_value_quadratic_l1694_169443


namespace max_d_value_l1694_169467

def is_valid_number (d f : ℕ) : Prop :=
  d < 10 ∧ f < 10 ∧ (636330 + 100000 * d + f) % 33 = 0

theorem max_d_value :
  (∃ d f : ℕ, is_valid_number d f) →
  (∀ d f : ℕ, is_valid_number d f → d ≤ 9) ∧
  (∃ f : ℕ, is_valid_number 9 f) :=
sorry

end max_d_value_l1694_169467


namespace boat_men_count_l1694_169445

/-- The number of men in the boat -/
def n : ℕ := 8

/-- The weight of the man being replaced -/
def old_weight : ℕ := 60

/-- The weight of the new man -/
def new_weight : ℕ := 68

/-- The increase in average weight after replacement -/
def avg_increase : ℕ := 1

theorem boat_men_count :
  ∀ W : ℕ,
  (W + (new_weight - old_weight)) / n = W / n + avg_increase →
  n = 8 :=
sorry

end boat_men_count_l1694_169445


namespace triangular_sum_iff_squares_sum_l1694_169420

/-- A triangular number is a positive integer of the form n * (n + 1) / 2 -/
def IsTriangular (k : ℕ) : Prop :=
  ∃ n : ℕ, k = n * (n + 1) / 2

/-- m is a sum of two triangular numbers -/
def IsSumOfTwoTriangular (m : ℕ) : Prop :=
  ∃ a b : ℕ, IsTriangular a ∧ IsTriangular b ∧ m = a + b

/-- n is a sum of two squares -/
def IsSumOfTwoSquares (n : ℕ) : Prop :=
  ∃ x y : ℤ, n = x^2 + y^2

/-- Main theorem: m is a sum of two triangular numbers if and only if 4m + 1 is a sum of two squares -/
theorem triangular_sum_iff_squares_sum (m : ℕ) :
  IsSumOfTwoTriangular m ↔ IsSumOfTwoSquares (4 * m + 1) :=
sorry

end triangular_sum_iff_squares_sum_l1694_169420


namespace complex_division_example_l1694_169448

theorem complex_division_example : (1 - 3*I) / (1 + I) = -1 - 2*I := by
  sorry

end complex_division_example_l1694_169448


namespace incorrect_derivation_l1694_169437

theorem incorrect_derivation : ¬ (∀ (a b c : ℝ), c > 0 → c / a > c / b → a < b) := by
  sorry

end incorrect_derivation_l1694_169437


namespace new_players_joined_new_players_joined_game_l1694_169419

theorem new_players_joined (initial_players : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) : ℕ :=
  let new_players := (total_lives - initial_players * lives_per_player) / lives_per_player
  new_players

theorem new_players_joined_game : new_players_joined 8 6 60 = 2 := by
  sorry

end new_players_joined_new_players_joined_game_l1694_169419


namespace one_quadrilateral_is_rhombus_l1694_169466

-- Define the properties of quadrilaterals
structure QuadrilateralProperties where
  opposite_sides_equal : Bool
  opposite_sides_parallel : Bool
  adjacent_sides_equal : Bool
  diagonals_perpendicular_and_bisected : Bool

-- Define a function to check if a quadrilateral with given properties is a rhombus
def is_rhombus (props : QuadrilateralProperties) : Bool :=
  (props.opposite_sides_equal && props.adjacent_sides_equal) ||
  (props.adjacent_sides_equal && props.diagonals_perpendicular_and_bisected) ||
  (props.opposite_sides_equal && props.diagonals_perpendicular_and_bisected)

-- Theorem statement
theorem one_quadrilateral_is_rhombus 
  (quad1 props1 quad2 props2 : QuadrilateralProperties) 
  (h1 : props1.opposite_sides_equal + props1.opposite_sides_parallel + 
        props1.adjacent_sides_equal + props1.diagonals_perpendicular_and_bisected = 2)
  (h2 : props2.opposite_sides_equal + props2.opposite_sides_parallel + 
        props2.adjacent_sides_equal + props2.diagonals_perpendicular_and_bisected = 2)
  (h3 : props1.opposite_sides_equal + props2.opposite_sides_equal = 1)
  (h4 : props1.opposite_sides_parallel + props2.opposite_sides_parallel = 1)
  (h5 : props1.adjacent_sides_equal + props2.adjacent_sides_equal = 1)
  (h6 : props1.diagonals_perpendicular_and_bisected + props2.diagonals_perpendicular_and_bisected = 1) :
  is_rhombus props1 ∨ is_rhombus props2 :=
sorry

end one_quadrilateral_is_rhombus_l1694_169466


namespace arithmetic_sequence_average_l1694_169454

/-- Given an arithmetic sequence with 5 terms, first term 8, and common difference 8,
    prove that the average (mean) of the sequence is 24. -/
theorem arithmetic_sequence_average (a : Fin 5 → ℕ) 
  (h1 : a 0 = 8)
  (h2 : ∀ i : Fin 4, a (i + 1) = a i + 8) :
  (Finset.sum Finset.univ a) / 5 = 24 := by
  sorry

end arithmetic_sequence_average_l1694_169454


namespace pencil_weight_l1694_169447

/-- Given that 5 pencils weigh 141.5 grams, prove that one pencil weighs 28.3 grams. -/
theorem pencil_weight (total_weight : ℝ) (num_pencils : ℕ) (h1 : total_weight = 141.5) (h2 : num_pencils = 5) :
  total_weight / num_pencils = 28.3 := by
  sorry

end pencil_weight_l1694_169447


namespace amoeba_growth_30_minutes_l1694_169413

/-- The number of amoebas after a given time interval, given an initial population and growth rate. -/
def amoeba_population (initial : ℕ) (growth_factor : ℕ) (intervals : ℕ) : ℕ :=
  initial * growth_factor ^ intervals

/-- Theorem stating that given the initial conditions, the final amoeba population after 30 minutes is 36450. -/
theorem amoeba_growth_30_minutes :
  let initial_population : ℕ := 50
  let growth_factor : ℕ := 3
  let interval_duration : ℕ := 5
  let total_duration : ℕ := 30
  let num_intervals : ℕ := total_duration / interval_duration
  amoeba_population initial_population growth_factor num_intervals = 36450 := by
  sorry

#eval amoeba_population 50 3 6

end amoeba_growth_30_minutes_l1694_169413


namespace carrot_weight_problem_l1694_169451

/-- Given 20 carrots weighing 3.64 kg, if 4 carrots are removed and the average weight
    of the remaining 16 carrots is 180 grams, then the average weight of the 4 removed
    carrots is 190 grams. -/
theorem carrot_weight_problem (total_weight : Real) (remaining_avg : Real) :
  total_weight = 3.64 →
  remaining_avg = 180 →
  let removed := 4
  let remaining := 20 - removed
  let removed_weight := total_weight * 1000 - remaining * remaining_avg
  removed_weight / removed = 190 := by
  sorry

end carrot_weight_problem_l1694_169451


namespace sum_of_powers_of_fifth_root_of_unity_l1694_169417

theorem sum_of_powers_of_fifth_root_of_unity (ω : ℂ) (h1 : ω^5 = 1) (h2 : ω ≠ 1) :
  ω^15 + ω^18 + ω^21 + ω^24 + ω^27 + ω^30 + ω^33 + ω^36 + ω^39 + ω^42 + ω^45 = 0 := by
  sorry

end sum_of_powers_of_fifth_root_of_unity_l1694_169417


namespace math_competition_problem_l1694_169455

theorem math_competition_problem (a b : ℝ) 
  (ha : 4 / a^4 - 2 / a^2 - 3 = 0) 
  (hb : b^4 + b^2 - 3 = 0) : 
  (a^4 * b^4 + 4) / a^4 = 7 := by
  sorry

end math_competition_problem_l1694_169455


namespace specific_pyramid_surface_area_l1694_169449

/-- A right rectangular pyramid with square bases -/
structure RightRectangularPyramid where
  upperBaseEdge : ℝ
  lowerBaseEdge : ℝ
  sideEdge : ℝ

/-- Calculate the surface area of a right rectangular pyramid -/
def surfaceArea (p : RightRectangularPyramid) : ℝ :=
  -- Surface area calculation
  sorry

/-- The theorem stating the surface area of the specific pyramid -/
theorem specific_pyramid_surface_area :
  let p : RightRectangularPyramid := {
    upperBaseEdge := 2,
    lowerBaseEdge := 4,
    sideEdge := 2
  }
  surfaceArea p = 10 * Real.sqrt 3 + 20 := by
  sorry

end specific_pyramid_surface_area_l1694_169449


namespace rabbit_count_prove_rabbit_count_l1694_169401

theorem rabbit_count : ℕ → ℕ → Prop :=
  fun total_white total_gray =>
    (∃ (caged_white : ℕ), caged_white = 6 ∧ total_white = caged_white + 9) ∧
    (∃ (caged_gray : ℕ), caged_gray = 4 ∧ total_gray = caged_gray) ∧
    (∃ (caged_white : ℕ), caged_white = 9 ∧ total_white = caged_white) ∧
    (∃ (caged_gray : ℕ), caged_gray = 4 ∧ total_gray = caged_gray + 16) →
    total_white + total_gray = 159

theorem prove_rabbit_count : ∃ (total_white total_gray : ℕ), rabbit_count total_white total_gray :=
  sorry

end rabbit_count_prove_rabbit_count_l1694_169401


namespace height_percentage_difference_l1694_169406

theorem height_percentage_difference (a b : ℝ) (h : b = 1.25 * a) :
  (b - a) / b * 100 = 20 := by sorry

end height_percentage_difference_l1694_169406


namespace valid_team_combinations_l1694_169488

/-- The number of ways to select a team of size k from n people -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of guests -/
def total_guests : ℕ := 5

/-- The number of male guests -/
def male_guests : ℕ := 3

/-- The number of female guests -/
def female_guests : ℕ := 2

/-- The required team size -/
def team_size : ℕ := 3

/-- The number of valid team combinations -/
def valid_combinations : ℕ := 
  choose male_guests 1 * choose female_guests 2 + 
  choose male_guests 2 * choose female_guests 1

theorem valid_team_combinations : valid_combinations = 9 := by sorry

end valid_team_combinations_l1694_169488


namespace four_digit_sum_3333_l1694_169411

/-- Represents a four-digit number as a tuple of its digits -/
def FourDigitNumber := (Nat × Nat × Nat × Nat)

/-- Converts a FourDigitNumber to its numerical value -/
def toNumber (n : FourDigitNumber) : Nat :=
  1000 * n.1 + 100 * n.2.1 + 10 * n.2.2.1 + n.2.2.2

/-- Rearranges a FourDigitNumber by moving the last digit to the front -/
def rearrange (n : FourDigitNumber) : FourDigitNumber :=
  (n.2.2.2, n.1, n.2.1, n.2.2.1)

/-- Checks if a FourDigitNumber contains zero -/
def containsZero (n : FourDigitNumber) : Bool :=
  n.1 = 0 || n.2.1 = 0 || n.2.2.1 = 0 || n.2.2.2 = 0

theorem four_digit_sum_3333 (n : FourDigitNumber) :
  ¬containsZero n →
  toNumber n + toNumber (rearrange n) = 3333 →
  n = (1, 2, 1, 2) ∨ n = (2, 1, 2, 1) := by
  sorry

end four_digit_sum_3333_l1694_169411


namespace min_sum_with_log_condition_l1694_169496

theorem min_sum_with_log_condition (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h_log : Real.log a + Real.log b = Real.log (a + b)) :
  ∀ x y : ℝ, x > 0 → y > 0 → Real.log x + Real.log y = Real.log (x + y) → a + b ≤ x + y ∧ a + b = 4 :=
by sorry

end min_sum_with_log_condition_l1694_169496


namespace complex_equation_solution_l1694_169432

theorem complex_equation_solution (z : ℂ) :
  (2 - 5 * Complex.I) * z = 29 → z = 2 + 5 * Complex.I := by
  sorry

end complex_equation_solution_l1694_169432


namespace football_game_spectators_l1694_169472

/-- Represents the number of spectators at a football game --/
structure Spectators :=
  (adults : ℕ)
  (children : ℕ)
  (vips : ℕ)

/-- Conditions of the football game spectator problem --/
def football_game_conditions (s : Spectators) : Prop :=
  s.vips = 20 ∧
  s.children = s.adults / 2 ∧
  2 * s.adults + 2 * s.children + 2 * s.vips = 310

/-- Theorem stating the correct number of spectators --/
theorem football_game_spectators :
  ∃ (s : Spectators), football_game_conditions s ∧
    s.adults = 90 ∧ s.children = 45 ∧ s.vips = 20 ∧
    s.adults + s.children + s.vips = 155 :=
sorry


end football_game_spectators_l1694_169472


namespace cubic_inequality_l1694_169426

theorem cubic_inequality (x : ℝ) : x^3 - 12*x^2 > -36*x ↔ x ∈ Set.Ioo 0 6 ∪ Set.Ioi 6 := by
  sorry

end cubic_inequality_l1694_169426


namespace simplify_fraction_l1694_169423

theorem simplify_fraction (a : ℝ) (h : a ≠ 1) : 
  (a^2 / (a - 1)) - ((1 - 2*a) / (1 - a)) = a - 1 := by
  sorry

end simplify_fraction_l1694_169423


namespace only_B_and_C_valid_l1694_169473

-- Define the set of individuals
inductive Person : Type
  | A : Person
  | B : Person
  | C : Person
  | D : Person

-- Define a type for the selection of individuals
def Selection := Person → Prop

-- Define the conditions
def condition1 (s : Selection) : Prop := s Person.A → s Person.B
def condition2 (s : Selection) : Prop := ¬(s Person.C) → ¬(s Person.B)
def condition3 (s : Selection) : Prop := s Person.C → ¬(s Person.D)

-- Define that exactly two individuals are selected
def exactlyTwo (s : Selection) : Prop :=
  (∃ (p1 p2 : Person), p1 ≠ p2 ∧ s p1 ∧ s p2 ∧ ∀ (p : Person), s p → (p = p1 ∨ p = p2))

-- State the theorem
theorem only_B_and_C_valid :
  ∀ (s : Selection),
    condition1 s →
    condition2 s →
    condition3 s →
    exactlyTwo s →
    s Person.B ∧ s Person.C ∧ ¬(s Person.A) ∧ ¬(s Person.D) :=
by
  sorry


end only_B_and_C_valid_l1694_169473


namespace triangle_problem_l1694_169470

/-- Given a triangle ABC with sides a, b, c and angles A, B, C, prove the angle B and area of the triangle under specific conditions. -/
theorem triangle_problem (a b c A B C : ℝ) : 
  (a + b) / Real.sin (A + B) = (a - c) / (Real.sin A - Real.sin B) →
  b = 3 →
  Real.sin A = Real.sqrt 3 / 3 →
  B = π / 3 ∧ 
  (1/2 * a * b * Real.sin C = (Real.sqrt 3 + 3 * Real.sqrt 2) / 2) := by
  sorry

end triangle_problem_l1694_169470


namespace inverse_proportion_y_relationship_l1694_169492

theorem inverse_proportion_y_relationship :
  ∀ (y₁ y₂ y₃ : ℝ),
  (y₁ = -3 / (-3)) →
  (y₂ = -3 / (-1)) →
  (y₃ = -3 / (1/3)) →
  (y₃ < y₁) ∧ (y₁ < y₂) := by
  sorry

end inverse_proportion_y_relationship_l1694_169492


namespace det_dilation_matrix_3d_l1694_169407

/-- A matrix representing a dilation centered at the origin with scale factor 4 -/
def dilation_matrix (n : ℕ) (k : ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  Matrix.diagonal (λ _ => k)

theorem det_dilation_matrix_3d :
  let E := dilation_matrix 3 4
  Matrix.det E = 64 := by sorry

end det_dilation_matrix_3d_l1694_169407


namespace complex_number_existence_l1694_169475

theorem complex_number_existence : ∃ z : ℂ, 
  (∃ r : ℝ, z + 5 / z = r) ∧ 
  (Complex.re (z + 3) = -Complex.im (z + 3)) ∧
  ((z = -1 - 2*Complex.I) ∨ (z = -2 - Complex.I)) := by
  sorry

end complex_number_existence_l1694_169475


namespace b_speed_is_13_l1694_169487

-- Define the walking scenario
def walking_scenario (speed_A speed_B initial_distance meeting_time : ℝ) : Prop :=
  speed_A > 0 ∧ speed_B > 0 ∧ initial_distance > 0 ∧ meeting_time > 0 ∧
  speed_A * meeting_time + speed_B * meeting_time = initial_distance

-- Theorem statement
theorem b_speed_is_13 :
  ∀ (speed_B : ℝ),
    walking_scenario 12 speed_B 25 1 →
    speed_B = 13 := by
  sorry

end b_speed_is_13_l1694_169487


namespace largest_of_five_consecutive_even_l1694_169427

/-- The sum of the first n positive even integers -/
def sum_first_n_even (n : ℕ) : ℕ := n * (n + 1)

/-- Sum of five consecutive even integers -/
def sum_five_consecutive_even (m : ℕ) : ℕ := 5 * m - 20

theorem largest_of_five_consecutive_even : 
  ∃ m : ℕ, sum_first_n_even 30 = sum_five_consecutive_even m ∧ m = 190 := by
  sorry

end largest_of_five_consecutive_even_l1694_169427


namespace katy_brownies_theorem_l1694_169460

/-- The number of brownies Katy made -/
def total_brownies : ℕ := 15

/-- The number of brownies Katy ate on Monday -/
def monday_brownies : ℕ := 5

/-- The number of brownies Katy ate on Tuesday -/
def tuesday_brownies : ℕ := 2 * monday_brownies

theorem katy_brownies_theorem :
  total_brownies = monday_brownies + tuesday_brownies :=
by sorry

end katy_brownies_theorem_l1694_169460


namespace arithmetic_calculation_l1694_169490

theorem arithmetic_calculation : 14 - (-12) + (-25) - 17 = -16 := by
  sorry

end arithmetic_calculation_l1694_169490
