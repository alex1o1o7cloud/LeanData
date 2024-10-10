import Mathlib

namespace complex_magnitude_example_l3584_358483

theorem complex_magnitude_example : Complex.abs (12 - 5*Complex.I) = 13 := by
  sorry

end complex_magnitude_example_l3584_358483


namespace x_squared_in_set_l3584_358408

theorem x_squared_in_set (x : ℝ) : x^2 ∈ ({0, -1, x} : Set ℝ) → x = 1 := by
  sorry

end x_squared_in_set_l3584_358408


namespace probability_x_plus_2y_leq_6_l3584_358488

-- Define the region
def region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 5}

-- Define the condition x + 2y ≤ 6
def condition (p : ℝ × ℝ) : Prop :=
  p.1 + 2 * p.2 ≤ 6

-- Define the probability measure on the region
noncomputable def prob : MeasureTheory.Measure (ℝ × ℝ) :=
  sorry

-- State the theorem
theorem probability_x_plus_2y_leq_6 :
  prob {p ∈ region | condition p} / prob region = 3 / 10 := by
  sorry

end probability_x_plus_2y_leq_6_l3584_358488


namespace exists_perpendicular_k_line_intersects_circle_chord_length_when_k_neg_one_l3584_358495

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y + 2 * k = 0

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 9

-- Define the line l₀
def line_l0 (x y : ℝ) : Prop := x - 2 * y + 2 = 0

-- Statement 1: Perpendicularity condition
theorem exists_perpendicular_k : ∃ k : ℝ, ∀ x y : ℝ, 
  line_l k x y → line_l0 x y → k * (1/2) = -1 :=
sorry

-- Statement 2: Intersection of line l and circle O
theorem line_intersects_circle : ∀ k : ℝ, ∃ x y : ℝ, 
  line_l k x y ∧ circle_O x y :=
sorry

-- Statement 3: Chord length when k = -1
theorem chord_length_when_k_neg_one : 
  ∃ x₁ y₁ x₂ y₂ : ℝ, 
    line_l (-1) x₁ y₁ ∧ line_l (-1) x₂ y₂ ∧ 
    circle_O x₁ y₁ ∧ circle_O x₂ y₂ ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = 28 :=
sorry

end exists_perpendicular_k_line_intersects_circle_chord_length_when_k_neg_one_l3584_358495


namespace product_of_specific_numbers_l3584_358482

theorem product_of_specific_numbers : 469157 * 9999 = 4690872843 := by
  sorry

end product_of_specific_numbers_l3584_358482


namespace function_f_properties_l3584_358450

/-- A function satisfying the given conditions -/
def FunctionF (f : ℝ → ℝ) : Prop :=
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ ≠ f x₂) ∧
  (∀ x y, f (x + y) = f x * f y)

/-- Theorem stating the properties of the function f -/
theorem function_f_properties (f : ℝ → ℝ) (hf : FunctionF f) :
  f 0 = 1 ∧ ∀ x, f x > 0 := by
  sorry

end function_f_properties_l3584_358450


namespace quadratic_always_has_real_roots_min_a_for_positive_integer_roots_min_a_is_zero_l3584_358491

/-- The quadratic equation x^2 - (a+2)x + (a+1) = 0 -/
def quadratic (a : ℝ) (x : ℝ) : ℝ := x^2 - (a+2)*x + (a+1)

/-- The discriminant of the quadratic equation -/
def discriminant (a : ℝ) : ℝ := (a+2)^2 - 4*(a+1)

theorem quadratic_always_has_real_roots (a : ℝ) :
  discriminant a ≥ 0 := by sorry

theorem min_a_for_positive_integer_roots :
  ∀ a : ℕ, (∃ x y : ℕ, x ≠ y ∧ quadratic a x = 0 ∧ quadratic a y = 0) →
  a ≥ 0 := by sorry

theorem min_a_is_zero :
  ∃ a : ℕ, a = 0 ∧
  (∃ x y : ℕ, x ≠ y ∧ quadratic a x = 0 ∧ quadratic a y = 0) ∧
  ∀ b : ℕ, b < a →
  ¬(∃ x y : ℕ, x ≠ y ∧ quadratic b x = 0 ∧ quadratic b y = 0) := by sorry

end quadratic_always_has_real_roots_min_a_for_positive_integer_roots_min_a_is_zero_l3584_358491


namespace parabola_c_value_l3584_358424

/-- Given a parabola y = ax^2 + bx + c with vertex (3, -5) passing through (1, -3),
    prove that c = -0.5 -/
theorem parabola_c_value (a b c : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →   -- Parabola equation
  (3, -5) = (3, a * 3^2 + b * 3 + c) →     -- Vertex condition
  -3 = a * 1^2 + b * 1 + c →               -- Point condition
  c = -0.5 := by
sorry


end parabola_c_value_l3584_358424


namespace percentage_of_indian_children_l3584_358441

theorem percentage_of_indian_children 
  (total_men : ℕ) 
  (total_women : ℕ) 
  (total_children : ℕ) 
  (percent_indian_men : ℚ) 
  (percent_indian_women : ℚ) 
  (percent_non_indian : ℚ) :
  total_men = 500 →
  total_women = 300 →
  total_children = 500 →
  percent_indian_men = 10 / 100 →
  percent_indian_women = 60 / 100 →
  percent_non_indian = 55.38461538461539 / 100 →
  (↑(total_men * 10 + total_women * 60 + total_children * 70) / ↑(total_men + total_women + total_children) : ℚ) = 1 - percent_non_indian :=
by sorry

end percentage_of_indian_children_l3584_358441


namespace modified_cube_properties_l3584_358437

/-- Represents a cube with removals as described in the problem -/
structure ModifiedCube where
  side_length : ℕ
  small_cube_size : ℕ
  center_removal_size : ℕ
  unit_removal : Bool

/-- Calculates the remaining volume after removals -/
def remaining_volume (c : ModifiedCube) : ℕ := sorry

/-- Calculates the surface area after removals -/
def surface_area (c : ModifiedCube) : ℕ := sorry

/-- The main theorem stating the properties of the modified cube -/
theorem modified_cube_properties :
  let c : ModifiedCube := {
    side_length := 12,
    small_cube_size := 2,
    center_removal_size := 2,
    unit_removal := true
  }
  remaining_volume c = 1463 ∧ surface_area c = 4598 := by sorry

end modified_cube_properties_l3584_358437


namespace tallest_giraffe_height_is_96_l3584_358443

/-- The height of the shortest giraffe in inches -/
def shortest_giraffe_height : ℕ := 68

/-- The height difference between the tallest and shortest giraffes in inches -/
def height_difference : ℕ := 28

/-- The number of adult giraffes at the zoo -/
def num_giraffes : ℕ := 14

/-- The height of the tallest giraffe in inches -/
def tallest_giraffe_height : ℕ := shortest_giraffe_height + height_difference

theorem tallest_giraffe_height_is_96 : tallest_giraffe_height = 96 := by
  sorry

end tallest_giraffe_height_is_96_l3584_358443


namespace pentagon_area_l3584_358496

/-- Given points on a coordinate plane, prove the area of the pentagon formed by these points and their intersection. -/
theorem pentagon_area (A B D E C : ℝ × ℝ) : 
  A = (9, 1) →
  B = (2, 0) →
  D = (1, 5) →
  E = (9, 7) →
  (C.1 - A.1) / (D.1 - A.1) = (C.2 - A.2) / (D.2 - A.2) →
  (C.1 - B.1) / (E.1 - B.1) = (C.2 - B.2) / (E.2 - B.2) →
  abs ((A.1 * B.2 + B.1 * C.2 + C.1 * D.2 + D.1 * E.2 + E.1 * A.2) -
       (A.2 * B.1 + B.2 * C.1 + C.2 * D.1 + D.2 * E.1 + E.2 * A.1)) / 2 = 33 :=
by sorry

end pentagon_area_l3584_358496


namespace triangle_db_length_l3584_358462

-- Define the triangle ABC and point D
structure Triangle :=
  (A B C D : ℝ × ℝ)
  (right_angle_ABC : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0)
  (right_angle_ADB : (B.1 - A.1) * (D.1 - A.1) + (B.2 - A.2) * (D.2 - A.2) = 0)
  (AC_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 19)
  (AD_length : Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) = 4)

-- Theorem statement
theorem triangle_db_length (t : Triangle) : 
  Real.sqrt ((t.B.1 - t.D.1)^2 + (t.B.2 - t.D.2)^2) = 2 * Real.sqrt 15 := by
  sorry

end triangle_db_length_l3584_358462


namespace no_solution_for_system_l3584_358489

theorem no_solution_for_system :
  ¬ ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    x^(1/3) - y^(1/3) - z^(1/3) = 64 ∧
    x^(1/4) - y^(1/4) - z^(1/4) = 32 ∧
    x^(1/6) - y^(1/6) - z^(1/6) = 8 :=
by sorry

end no_solution_for_system_l3584_358489


namespace arithmetic_geometric_sequence_l3584_358415

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = a n + 2) →  -- arithmetic sequence with common difference 2
  (a 3)^2 = a 1 * a 4 →  -- a_1, a_3, and a_4 form a geometric sequence
  a 1 = -8 := by
sorry

end arithmetic_geometric_sequence_l3584_358415


namespace circle_relationship_l3584_358494

-- Define the circles and point
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 = 9
def circle_O2 (a b x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = 1
def point_on_O1 (x y : ℝ) : Prop := circle_O1 x y

-- Define the condition for P and Q
def condition_PQ (x₁ y₁ a b : ℝ) : Prop := (a - x₁)^2 + (b - y₁)^2 = 1

-- Define the possible relationships
inductive CircleRelationship
  | ExternallyTangent
  | Intersecting
  | InternallyTangent

-- Theorem statement
theorem circle_relationship 
  (x₁ y₁ a b : ℝ) 
  (h1 : point_on_O1 x₁ y₁) 
  (h2 : condition_PQ x₁ y₁ a b) : 
  ∃ r : CircleRelationship, r = CircleRelationship.ExternallyTangent ∨ 
                            r = CircleRelationship.Intersecting ∨ 
                            r = CircleRelationship.InternallyTangent :=
sorry

end circle_relationship_l3584_358494


namespace three_heads_one_tail_probability_three_heads_one_tail_probability_proof_l3584_358493

/-- The probability of getting exactly three heads and one tail when four fair coins are tossed simultaneously -/
theorem three_heads_one_tail_probability : ℝ :=
  1 / 4

/-- Proof that the probability of getting exactly three heads and one tail when four fair coins are tossed simultaneously is 1/4 -/
theorem three_heads_one_tail_probability_proof :
  three_heads_one_tail_probability = 1 / 4 := by
  sorry

end three_heads_one_tail_probability_three_heads_one_tail_probability_proof_l3584_358493


namespace contrapositive_equivalence_l3584_358481

theorem contrapositive_equivalence (m : ℕ+) : 
  (¬(∃ x : ℝ, x^2 + x - m.val = 0) → m.val ≤ 0) ↔ 
  (m.val > 0 → ∃ x : ℝ, x^2 + x - m.val = 0) :=
by sorry

end contrapositive_equivalence_l3584_358481


namespace factorial_division_l3584_358446

theorem factorial_division : (Nat.factorial 8) / (Nat.factorial (8 - 2)) = 56 := by
  sorry

end factorial_division_l3584_358446


namespace seating_arrangements_mod_1000_l3584_358422

/-- Represents a seating arrangement of ambassadors and advisors. -/
structure SeatingArrangement where
  ambassador_seats : Finset (Fin 6)
  advisor_seats : Finset (Fin 12)

/-- The set of all valid seating arrangements. -/
def validArrangements : Finset SeatingArrangement :=
  sorry

/-- The number of valid seating arrangements. -/
def N : ℕ := Finset.card validArrangements

/-- Theorem stating that the number of valid seating arrangements
    is congruent to 520 modulo 1000. -/
theorem seating_arrangements_mod_1000 :
  N % 1000 = 520 := by sorry

end seating_arrangements_mod_1000_l3584_358422


namespace A_sufficient_for_B_l3584_358436

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 2}

theorem A_sufficient_for_B : ∀ x : ℝ, x ∈ A → x ∈ B := by
  sorry

end A_sufficient_for_B_l3584_358436


namespace distance_sum_inequality_l3584_358400

theorem distance_sum_inequality (a : ℝ) (ha : a > 0) :
  (∃ x : ℝ, |x - 5| + |x - 1| < a) ↔ a > 4 := by sorry

end distance_sum_inequality_l3584_358400


namespace base_8_sum_4321_l3584_358449

def base_8_sum (n : ℕ) : ℕ :=
  (n.digits 8).sum

theorem base_8_sum_4321 : base_8_sum 4321 = 9 := by sorry

end base_8_sum_4321_l3584_358449


namespace root_in_interval_l3584_358439

theorem root_in_interval : ∃ x : ℝ, 2 < x ∧ x < 3 ∧ Real.log x + x - 4 = 0 := by
  sorry

end root_in_interval_l3584_358439


namespace no_bounded_function_satisfying_inequality_l3584_358409

theorem no_bounded_function_satisfying_inequality :
  ¬ ∃ f : ℝ → ℝ, (∀ x : ℝ, ∃ M : ℝ, |f x| ≤ M) ∧ 
    (f 1 > 0) ∧ 
    (∀ x y : ℝ, (f (x + y))^2 ≥ (f x)^2 + 2 * f (x * y) + (f y)^2) :=
by
  sorry

end no_bounded_function_satisfying_inequality_l3584_358409


namespace gifted_subscribers_l3584_358428

/-- Calculates the number of gifted subscribers for a Twitch streamer --/
theorem gifted_subscribers
  (initial_subscribers : ℕ)
  (income_per_subscriber : ℕ)
  (current_monthly_income : ℕ)
  (h1 : initial_subscribers = 150)
  (h2 : income_per_subscriber = 9)
  (h3 : current_monthly_income = 1800) :
  current_monthly_income / income_per_subscriber - initial_subscribers = 50 :=
by sorry

end gifted_subscribers_l3584_358428


namespace smallest_x_absolute_value_equation_l3584_358440

theorem smallest_x_absolute_value_equation :
  ∃ (x : ℝ), x * |x| = 3 * x + 2 ∧
  ∀ (y : ℝ), y * |y| = 3 * y + 2 → x ≤ y :=
by sorry

end smallest_x_absolute_value_equation_l3584_358440


namespace haley_necklace_count_l3584_358458

/-- The number of necklaces Haley, Jason, and Josh have satisfy the given conditions -/
def NecklaceProblem (h j q : ℕ) : Prop :=
  (h = j + 5) ∧ (q = j / 2) ∧ (h = q + 15)

/-- Theorem: If the necklace counts satisfy the given conditions, then Haley has 25 necklaces -/
theorem haley_necklace_count
  (h j q : ℕ) (hcond : NecklaceProblem h j q) : h = 25 := by
  sorry

end haley_necklace_count_l3584_358458


namespace expression_equals_two_l3584_358403

theorem expression_equals_two (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 2) :
  (2 * x^2 - x) / ((x + 1) * (x - 2)) - (4 + x) / ((x + 1) * (x - 2)) = 2 :=
by sorry

end expression_equals_two_l3584_358403


namespace subset_condition_l3584_358463

theorem subset_condition (A B : Set ℕ) (m : ℕ) : 
  A = {0, 1, 2} → 
  B = {1, m} → 
  B ⊆ A → 
  m = 0 ∨ m = 2 := by
sorry

end subset_condition_l3584_358463


namespace gcd_of_four_numbers_l3584_358411

theorem gcd_of_four_numbers : Nat.gcd 546 (Nat.gcd 1288 (Nat.gcd 3042 5535)) = 1 := by
  sorry

end gcd_of_four_numbers_l3584_358411


namespace xy_negative_implies_abs_sum_less_abs_diff_l3584_358499

theorem xy_negative_implies_abs_sum_less_abs_diff (x y : ℝ) 
  (h1 : x * y < 0) : 
  |x + y| < |x - y| := by sorry

end xy_negative_implies_abs_sum_less_abs_diff_l3584_358499


namespace original_amount_is_48_l3584_358444

/-- Proves that the original amount is 48 rupees given the described transactions --/
theorem original_amount_is_48 (x : ℚ) : 
  ((2/3 * ((2/3 * x + 10) + 20)) = x) → x = 48 := by
  sorry

end original_amount_is_48_l3584_358444


namespace inequality_preservation_l3584_358497

theorem inequality_preservation (a b : ℝ) (h : a > b) : a + 1 > b + 1 := by
  sorry

end inequality_preservation_l3584_358497


namespace gcf_of_180_250_300_l3584_358474

theorem gcf_of_180_250_300 : Nat.gcd 180 (Nat.gcd 250 300) = 10 := by
  sorry

end gcf_of_180_250_300_l3584_358474


namespace complement_to_set_l3584_358480

def U : Finset Nat := {1,2,3,4,5,6,7,8}

theorem complement_to_set (B : Finset Nat) :
  (U \ B = {1,3}) → (B = {2,4,5,6,7,8}) := by
  sorry

end complement_to_set_l3584_358480


namespace larger_tv_diagonal_l3584_358465

theorem larger_tv_diagonal (d : ℝ) : d > 0 →
  (d^2 / 2) = (17^2 / 2) + 143.5 →
  d = 24 := by
sorry

end larger_tv_diagonal_l3584_358465


namespace quadratic_roots_distinct_l3584_358413

theorem quadratic_roots_distinct (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  (x₁^2 - (k+3)*x₁ + k = 0) ∧ 
  (x₂^2 - (k+3)*x₂ + k = 0) :=
by
  sorry

end quadratic_roots_distinct_l3584_358413


namespace no_integer_solutions_binomial_power_l3584_358484

theorem no_integer_solutions_binomial_power (n k m t : ℕ) (l : ℕ) (h1 : l ≥ 2) (h2 : 4 ≤ k) (h3 : k ≤ n - 4) :
  Nat.choose n k ≠ m ^ t := by
  sorry

end no_integer_solutions_binomial_power_l3584_358484


namespace log_inequality_l3584_358425

theorem log_inequality (x : ℝ) (h : x > 0) : Real.log (1 + x^2) < x^2 / (1 + x^2) := by
  sorry

end log_inequality_l3584_358425


namespace polynomial_no_x_x2_terms_l3584_358404

theorem polynomial_no_x_x2_terms (m n : ℚ) : 
  (∀ x, 3 * (x^3 + 1/3 * x^2 + n * x) - (m * x^2 - 6 * x - 1) = 
        3 * x^3 + 1) → 
  m + n = -1 := by
sorry

end polynomial_no_x_x2_terms_l3584_358404


namespace max_gcd_consecutive_b_l3584_358410

def b (n : ℕ) : ℕ := n.factorial + 3 * n

theorem max_gcd_consecutive_b : (∃ n : ℕ, Nat.gcd (b n) (b (n + 1)) = 14) ∧ 
  (∀ n : ℕ, Nat.gcd (b n) (b (n + 1)) ≤ 14) :=
sorry

end max_gcd_consecutive_b_l3584_358410


namespace parallel_vectors_angle_l3584_358445

theorem parallel_vectors_angle (α : Real) : 
  let a : Fin 2 → Real := ![1 - Real.cos α, Real.sqrt 3]
  let b : Fin 2 → Real := ![Real.sin α, 3]
  (∀ (i j : Fin 2), a i * b j = a j * b i) →  -- parallel condition
  0 < α → α < Real.pi / 2 →                   -- acute angle condition
  α = Real.pi / 6 := by
sorry

end parallel_vectors_angle_l3584_358445


namespace family_event_handshakes_l3584_358492

/-- The number of sets of twins at the family event -/
def twin_sets : ℕ := 12

/-- The number of sets of triplets at the family event -/
def triplet_sets : ℕ := 4

/-- The total number of twins at the family event -/
def total_twins : ℕ := twin_sets * 2

/-- The total number of triplets at the family event -/
def total_triplets : ℕ := triplet_sets * 3

/-- The fraction of triplets each twin shakes hands with -/
def twin_triplet_fraction : ℚ := 1 / 3

/-- The fraction of twins each triplet shakes hands with -/
def triplet_twin_fraction : ℚ := 2 / 3

/-- The total number of unique handshakes at the family event -/
def total_handshakes : ℕ := 462

theorem family_event_handshakes :
  (total_twins * (total_twins - 2) / 2) +
  (total_triplets * (total_triplets - 3) / 2) +
  ((total_twins * (total_triplets * twin_triplet_fraction).floor +
    total_triplets * (total_twins * triplet_twin_fraction).floor) / 2) =
  total_handshakes := by sorry

end family_event_handshakes_l3584_358492


namespace tens_digit_of_19_power_2023_l3584_358418

theorem tens_digit_of_19_power_2023 : ∃ n : ℕ, 19^2023 ≡ 50 + n [ZMOD 100] :=
sorry

end tens_digit_of_19_power_2023_l3584_358418


namespace x_plus_y_value_l3584_358486

theorem x_plus_y_value (x y : ℝ) (hx : |x| = 3) (hy : |y| = 5) (hxy : x * y < 0) :
  x + y = 2 ∨ x + y = -2 := by
sorry

end x_plus_y_value_l3584_358486


namespace reflection_of_P_is_correct_l3584_358421

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectXAxis (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem reflection_of_P_is_correct : 
  let P : Point := { x := 2, y := -3 }
  reflectXAxis P = { x := 2, y := 3 } := by
  sorry

end reflection_of_P_is_correct_l3584_358421


namespace arithmetic_square_root_of_sqrt_16_l3584_358419

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end arithmetic_square_root_of_sqrt_16_l3584_358419


namespace weekly_rate_is_190_l3584_358478

/-- Represents the car rental problem --/
structure CarRental where
  dailyRate : ℕ
  totalDays : ℕ
  totalCost : ℕ
  weeklyRate : ℕ

/-- The car rental agency's pricing policy --/
def rentalPolicy (r : CarRental) : Prop :=
  r.dailyRate = 30 ∧
  r.totalDays = 11 ∧
  r.totalCost = 310 ∧
  r.weeklyRate = r.totalCost - (r.totalDays - 7) * r.dailyRate

/-- Theorem stating that the weekly rate is $190 --/
theorem weekly_rate_is_190 (r : CarRental) :
  rentalPolicy r → r.weeklyRate = 190 := by
  sorry

#check weekly_rate_is_190

end weekly_rate_is_190_l3584_358478


namespace eight_steps_result_l3584_358423

def alternate_divide_multiply (n : ℕ) : ℕ → ℕ
  | 0 => n
  | i + 1 => if i % 2 = 0 then (alternate_divide_multiply n i) / 2 else (alternate_divide_multiply n i) * 3

theorem eight_steps_result :
  alternate_divide_multiply 10000000 8 = 2^3 * 3^4 * 5^7 := by
  sorry

end eight_steps_result_l3584_358423


namespace existence_of_abc_l3584_358466

theorem existence_of_abc (n : ℕ) (A : Finset ℕ) :
  A ⊆ Finset.range (5^n + 1) →
  A.card = 4*n + 2 →
  ∃ a b c : ℕ, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a < b ∧ b < c ∧ c + 2*a > 3*b :=
sorry

end existence_of_abc_l3584_358466


namespace grid_lines_formula_l3584_358452

/-- The number of straight lines needed to draw an n × n square grid -/
def grid_lines (n : ℕ) : ℕ := 2 * (n + 1)

/-- Theorem stating that the number of straight lines needed to draw an n × n square grid is 2(n + 1) -/
theorem grid_lines_formula (n : ℕ) : grid_lines n = 2 * (n + 1) := by
  sorry

end grid_lines_formula_l3584_358452


namespace last_two_digits_of_7_power_10_l3584_358471

theorem last_two_digits_of_7_power_10 : 7^10 ≡ 49 [ZMOD 100] := by
  sorry

end last_two_digits_of_7_power_10_l3584_358471


namespace yogurt_refund_l3584_358454

theorem yogurt_refund (total_packs : ℕ) (expired_percentage : ℚ) (cost_per_pack : ℕ) : 
  total_packs = 80 → 
  expired_percentage = 40 / 100 → 
  cost_per_pack = 12 → 
  (total_packs : ℚ) * expired_percentage * cost_per_pack = 384 := by
  sorry

end yogurt_refund_l3584_358454


namespace zero_not_in_range_of_f_l3584_358453

-- Define the function f
noncomputable def f : ℝ → ℤ
| x => if x > -1 then Int.ceil (1 / (x + 1))
       else if x < -1 then Int.floor (1 / (x + 1))
       else 0  -- This value doesn't matter as f is undefined at x = -1

-- Theorem statement
theorem zero_not_in_range_of_f :
  ∀ x : ℝ, x ≠ -1 → f x ≠ 0 := by sorry

end zero_not_in_range_of_f_l3584_358453


namespace fly_probabilities_l3584_358448

def fly_move (x y : ℕ) : Prop := x ≤ 8 ∧ y ≤ 10

def prob_reach (x y : ℕ) : ℚ := (Nat.choose (x + y) x : ℚ) / 2^(x + y)

def prob_through (x1 y1 x2 y2 x3 y3 : ℕ) : ℚ :=
  (Nat.choose (x1 + y1) x1 * Nat.choose (x3 - x2 + y3 - y2) (x3 - x2) : ℚ) / 2^(x3 + y3)

def inside_circle (x y cx cy r : ℝ) : Prop :=
  (x - cx)^2 + (y - cy)^2 ≤ r^2

theorem fly_probabilities :
  let p1 := prob_reach 8 10
  let p2 := prob_through 5 6 6 6 8 10
  let p3 := (2 * Nat.choose 9 2 * Nat.choose 9 6 + 2 * Nat.choose 9 3 * Nat.choose 9 5 + Nat.choose 9 4 ^ 2 : ℚ) / 2^18
  (p1 = (Nat.choose 18 8 : ℚ) / 2^18) ∧
  (p2 = (Nat.choose 11 5 * Nat.choose 6 2 : ℚ) / 2^18) ∧
  (∀ x y, fly_move x y → inside_circle x y 4 5 3 → prob_reach x y ≤ p3) := by
  sorry

end fly_probabilities_l3584_358448


namespace marked_line_points_l3584_358406

/-- Represents a line with marked points -/
structure MarkedLine where
  points : ℕ  -- Total number of points
  a_inside : ℕ  -- Number of segments A is inside
  b_inside : ℕ  -- Number of segments B is inside

/-- Theorem stating the number of points on the line -/
theorem marked_line_points (l : MarkedLine) 
  (ha : l.a_inside = 50) 
  (hb : l.b_inside = 56) : 
  l.points = 16 := by
  sorry

#check marked_line_points

end marked_line_points_l3584_358406


namespace apartment_cost_l3584_358438

/-- The cost of a room on the first floor of Krystiana's apartment building. -/
def first_floor_cost : ℝ := 8.75

/-- The number of rooms on each floor. -/
def rooms_per_floor : ℕ := 3

/-- The additional cost for a room on the second floor compared to the first floor. -/
def second_floor_additional_cost : ℝ := 20

theorem apartment_cost (total_earnings : ℝ) 
  (h_total : total_earnings = 165) :
  first_floor_cost * rooms_per_floor + 
  (first_floor_cost + second_floor_additional_cost) * rooms_per_floor + 
  (2 * first_floor_cost) * rooms_per_floor = total_earnings := by
  sorry

end apartment_cost_l3584_358438


namespace square_division_area_l3584_358456

theorem square_division_area : ∃ (x y : ℕ), 
  x > 0 ∧ y > 0 ∧ y ≠ 1 ∧ 
  x^2 = 24 + y^2 ∧
  x^2 = 49 := by
  sorry

end square_division_area_l3584_358456


namespace book_price_change_book_price_problem_l3584_358412

theorem book_price_change (initial_price : ℝ) 
  (decrease_percent : ℝ) (increase_percent : ℝ) : ℝ :=
  let price_after_decrease := initial_price * (1 - decrease_percent)
  let final_price := price_after_decrease * (1 + increase_percent)
  final_price

theorem book_price_problem : 
  book_price_change 400 0.15 0.40 = 476 := by
  sorry

end book_price_change_book_price_problem_l3584_358412


namespace xyz_equation_solutions_l3584_358470

theorem xyz_equation_solutions (n : ℕ+) (p : ℕ) (h_prime : Nat.Prime p) (h_p_gt_3 : p > 3) :
  ∃! k : ℕ, k = 3 * (n + 1) ∧
  ∃ S : Finset (ℕ × ℕ × ℕ),
    S.card = k ∧
    ∀ (x y z : ℕ), (x, y, z) ∈ S ↔ 
      x > 0 ∧ y > 0 ∧ z > 0 ∧ 
      x * y * z = p ^ (n : ℕ) * (x + y + z) :=
sorry

end xyz_equation_solutions_l3584_358470


namespace modulus_of_z_l3584_358461

theorem modulus_of_z (z : ℂ) (h : z * (2 - 3*I) = 6 + 4*I) : Complex.abs z = 2 := by
  sorry

end modulus_of_z_l3584_358461


namespace smallest_root_of_quadratic_l3584_358442

theorem smallest_root_of_quadratic (x : ℝ) :
  (12 * x^2 - 44 * x + 40 = 0) → (x ≥ 5/3) :=
by sorry

end smallest_root_of_quadratic_l3584_358442


namespace triangle_proof_l3584_358429

theorem triangle_proof 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h1 : A < π / 2) 
  (h2 : Real.sin (A - π / 4) = Real.sqrt 2 / 10) 
  (h3 : (1 / 2) * b * c * Real.sin A = 24) 
  (h4 : b = 10) : 
  Real.sin A = 4 / 5 ∧ a = 8 := by
  sorry

end triangle_proof_l3584_358429


namespace f_neg_one_equals_two_l3584_358417

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define g in terms of f
def g (x : ℝ) := f x + 4

-- State the theorem
theorem f_neg_one_equals_two
  (h_odd : ∀ x, f (-x) = -f x)  -- f is an odd function
  (h_g_one : g 1 = 2)           -- g(1) = 2
  : f (-1) = 2 := by
  sorry


end f_neg_one_equals_two_l3584_358417


namespace perpendicular_planes_from_perpendicular_lines_l3584_358434

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines
variable (perpendicular_lines : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_from_perpendicular_lines 
  (m n : Line) (α β : Plane) 
  (h_distinct_lines : m ≠ n) 
  (h_distinct_planes : α ≠ β) 
  (h_m_perp_n : perpendicular_lines m n) 
  (h_m_perp_α : perpendicular_line_plane m α) 
  (h_n_perp_β : perpendicular_line_plane n β) : 
  perpendicular_planes α β :=
sorry

end perpendicular_planes_from_perpendicular_lines_l3584_358434


namespace sum_of_digits_B_is_seven_l3584_358479

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A is the sum of digits of 4444^4444 -/
def A : ℕ := sumOfDigits (4444^4444)

/-- B is the sum of digits of A -/
def B : ℕ := sumOfDigits A

/-- Theorem: The sum of digits of B is 7 -/
theorem sum_of_digits_B_is_seven : sumOfDigits B = 7 := by sorry

end sum_of_digits_B_is_seven_l3584_358479


namespace min_boys_is_two_l3584_358477

/-- Represents the number of apples collected by a boy -/
inductive AppleCount
  | fixed : AppleCount  -- Represents 20 apples
  | percentage : AppleCount  -- Represents 20% of the total

/-- Represents a group of boys collecting apples -/
structure AppleCollection where
  boys : ℕ  -- Number of boys
  fixed_count : ℕ  -- Number of boys collecting fixed amount
  percentage_count : ℕ  -- Number of boys collecting percentage
  total_apples : ℕ  -- Total number of apples collected

/-- Checks if an AppleCollection is valid according to the problem conditions -/
def is_valid_collection (c : AppleCollection) : Prop :=
  c.boys = c.fixed_count + c.percentage_count ∧
  c.fixed_count > 0 ∧
  c.percentage_count > 0 ∧
  c.total_apples = 20 * c.fixed_count + (c.total_apples / 5) * c.percentage_count

/-- The main theorem stating that 2 is the minimum number of boys -/
theorem min_boys_is_two :
  ∀ c : AppleCollection, is_valid_collection c → c.boys ≥ 2 :=
sorry

end min_boys_is_two_l3584_358477


namespace consecutive_integer_roots_l3584_358430

theorem consecutive_integer_roots (p q : ℤ) : 
  (∃ x y : ℤ, x^2 - p*x + q = 0 ∧ y^2 - p*y + q = 0 ∧ y = x + 1) →
  Prime q →
  (p = 3 ∨ p = -3) ∧ q = 2 := by
sorry

end consecutive_integer_roots_l3584_358430


namespace hannah_bought_three_sweatshirts_l3584_358459

/-- Represents the purchase of sweatshirts and T-shirts by Hannah -/
structure Purchase where
  sweatshirts : ℕ
  tshirts : ℕ
  sweatshirt_cost : ℕ
  tshirt_cost : ℕ
  total_spent : ℕ

/-- Hannah's specific purchase -/
def hannahs_purchase : Purchase where
  sweatshirts := 0  -- We'll prove this should be 3
  tshirts := 2
  sweatshirt_cost := 15
  tshirt_cost := 10
  total_spent := 65

/-- The theorem stating that Hannah bought 3 sweatshirts -/
theorem hannah_bought_three_sweatshirts :
  ∃ (p : Purchase), p.tshirts = 2 ∧ p.sweatshirt_cost = 15 ∧ p.tshirt_cost = 10 ∧ p.total_spent = 65 ∧ p.sweatshirts = 3 :=
by
  sorry


end hannah_bought_three_sweatshirts_l3584_358459


namespace fred_likes_twelve_pairs_l3584_358455

theorem fred_likes_twelve_pairs : 
  (Finset.filter (fun n : Fin 100 => n.val % 8 = 0) Finset.univ).card = 12 := by
  sorry

end fred_likes_twelve_pairs_l3584_358455


namespace angela_figures_theorem_l3584_358407

def calculate_remaining_figures (initial : ℕ) : ℕ :=
  let increased := initial + (initial * 15 / 100)
  let after_selling := increased - (increased / 4)
  let after_giving_to_daughter := after_selling - (after_selling / 3)
  let final := after_giving_to_daughter - (after_giving_to_daughter * 20 / 100)
  final

theorem angela_figures_theorem :
  calculate_remaining_figures 24 = 12 := by
  sorry

end angela_figures_theorem_l3584_358407


namespace adam_has_23_tattoos_l3584_358460

/-- Calculates the number of tattoos Adam has given Jason's tattoo configuration -/
def adam_tattoos (jason_arm_tattoos jason_leg_tattoos jason_arms jason_legs : ℕ) : ℕ :=
  2 * (jason_arm_tattoos * jason_arms + jason_leg_tattoos * jason_legs) + 3

/-- Proves that Adam has 23 tattoos given Jason's tattoo configuration -/
theorem adam_has_23_tattoos :
  adam_tattoos 2 3 2 2 = 23 := by
  sorry

end adam_has_23_tattoos_l3584_358460


namespace unique_grid_solution_l3584_358468

-- Define the grid
def Grid := Fin 3 → Fin 3 → Option Char

-- Define adjacency
def adjacent (i j k l : Fin 3) : Prop :=
  (i = k ∧ j.val + 1 = l.val) ∨
  (i = k ∧ j.val = l.val + 1) ∨
  (i.val + 1 = k.val ∧ j = l) ∨
  (i.val = k.val + 1 ∧ j = l) ∨
  (i.val + 1 = k.val ∧ j.val + 1 = l.val) ∨
  (i.val + 1 = k.val ∧ j.val = l.val + 1) ∨
  (i.val = k.val + 1 ∧ j.val + 1 = l.val) ∨
  (i.val = k.val + 1 ∧ j.val = l.val + 1)

-- Define the constraints
def valid_grid (g : Grid) : Prop :=
  (∀ i j, g i j ∈ [none, some 'A', some 'B', some 'C']) ∧
  (∀ i, ∃! j, g i j = some 'A') ∧
  (∀ i, ∃! j, g i j = some 'B') ∧
  (∀ i, ∃! j, g i j = some 'C') ∧
  (∀ j, ∃! i, g i j = some 'A') ∧
  (∀ j, ∃! i, g i j = some 'B') ∧
  (∀ j, ∃! i, g i j = some 'C') ∧
  (∀ i j k l, adjacent i j k l → g i j ≠ g k l) ∧
  (g 0 1 = none ∧ g 1 0 = none)

-- Define the diagonal string
def diagonal_string (g : Grid) : String :=
  String.mk [
    (g 0 0).getD 'X',
    (g 1 1).getD 'X',
    (g 2 2).getD 'X'
  ]

-- The theorem to prove
theorem unique_grid_solution :
  ∀ g : Grid, valid_grid g → diagonal_string g = "XXC" := by
  sorry

end unique_grid_solution_l3584_358468


namespace oblique_line_plane_angle_range_l3584_358451

-- Define the angle between an oblique line and a plane
def angle_oblique_line_plane (θ : Real) : Prop := 
  θ > 0 ∧ θ < Real.pi / 2

-- Theorem statement
theorem oblique_line_plane_angle_range :
  ∀ θ : Real, angle_oblique_line_plane θ ↔ 0 < θ ∧ θ < Real.pi / 2 :=
by sorry

end oblique_line_plane_angle_range_l3584_358451


namespace cube_equals_nine_times_implies_fifth_power_l3584_358405

theorem cube_equals_nine_times_implies_fifth_power (w : ℕ+) 
  (h : w.val ^ 3 = 9 * w.val) : w.val ^ 5 = 243 := by
  sorry

end cube_equals_nine_times_implies_fifth_power_l3584_358405


namespace expression_value_l3584_358420

theorem expression_value : (5^8 - 3^7) * (1^6 + (-1)^5)^11 = 0 := by
  sorry

end expression_value_l3584_358420


namespace square_sum_value_l3584_358431

theorem square_sum_value (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 12) : a^2 + b^2 = 33 := by
  sorry

end square_sum_value_l3584_358431


namespace quadratic_inequality_solution_set_l3584_358402

-- Define the quadratic function types
def QuadraticFunction (a b c : ℝ) := λ x : ℝ => a * x^2 + b * x + c

-- Define the solution set type
def SolutionSet := Set ℝ

-- State the theorem
theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h1 : SolutionSet) 
  (h2 : h1 = {x : ℝ | x < (1/3) ∨ x > (1/2)}) 
  (h3 : h1 = {x : ℝ | QuadraticFunction a b c x < 0}) :
  {x : ℝ | QuadraticFunction c (-b) a x > 0} = Set.Ioo (-3) (-2) := by
  sorry

end quadratic_inequality_solution_set_l3584_358402


namespace trigonometric_factorization_l3584_358475

theorem trigonometric_factorization (x : Real) :
  1 - Real.sin x ^ 5 - Real.cos x ^ 5 =
  (1 - Real.sin x) * (1 - Real.cos x) *
  (3 + 2 * (Real.sin x + Real.cos x) + 2 * Real.sin x * Real.cos x +
   Real.sin x * Real.cos x * (Real.sin x + Real.cos x)) := by
  sorry

end trigonometric_factorization_l3584_358475


namespace relationship_abc_l3584_358498

theorem relationship_abc : 
  let a : ℝ := 1.1 * Real.log 1.1
  let b : ℝ := 0.1 * Real.exp 0.1
  let c : ℝ := 1 / 9
  a < b ∧ b < c := by sorry

end relationship_abc_l3584_358498


namespace commodity_tax_consumption_l3584_358416

theorem commodity_tax_consumption (T C : ℝ) (h1 : T > 0) (h2 : C > 0) : 
  let new_tax := 0.75 * T
  let new_revenue := 0.825 * T * C
  let new_consumption := C * (1 + 10 / 100)
  new_tax * new_consumption = new_revenue := by sorry

end commodity_tax_consumption_l3584_358416


namespace symmetrical_lines_and_ellipse_intersection_l3584_358476

/-- Given two lines symmetrical about y = x + 1, prove their slopes multiply to 1 and their intersection points with an ellipse form a line passing through a fixed point. -/
theorem symmetrical_lines_and_ellipse_intersection
  (k : ℝ) (h_k_pos : k > 0) (h_k_neq_one : k ≠ 1)
  (l : Set (ℝ × ℝ)) (l_eq : l = {(x, y) | y = k * x + 1})
  (l₁ : Set (ℝ × ℝ)) (k₁ : ℝ) (l₁_eq : l₁ = {(x, y) | y = k₁ * x + 1})
  (h_symmetry : ∀ (x y : ℝ), (x, y) ∈ l ↔ (y - 1, x + 1) ∈ l₁)
  (E : Set (ℝ × ℝ)) (E_eq : E = {(x, y) | x^2 / 4 + y^2 = 1})
  (A M : ℝ × ℝ) (h_AM : A ∈ E ∧ M ∈ E ∧ A ∈ l ∧ M ∈ l ∧ A ≠ M)
  (N : ℝ × ℝ) (h_AN : A ∈ E ∧ N ∈ E ∧ A ∈ l₁ ∧ N ∈ l₁ ∧ A ≠ N) :
  (k * k₁ = 1) ∧
  (∃ (m b : ℝ), ∀ (x : ℝ), M.2 - N.2 = m * (M.1 - N.1) ∧ N.2 = m * N.1 + b ∧ b = -5/3) :=
by sorry

end symmetrical_lines_and_ellipse_intersection_l3584_358476


namespace simplify_expression_l3584_358487

theorem simplify_expression (y : ℝ) : 7 * y - 3 * y + 9 + 15 = 4 * y + 24 := by
  sorry

end simplify_expression_l3584_358487


namespace root_sum_fraction_l3584_358432

theorem root_sum_fraction (p q r : ℝ) : 
  p^3 - 6*p^2 + 11*p - 6 = 0 →
  q^3 - 6*q^2 + 11*q - 6 = 0 →
  r^3 - 6*r^2 + 11*r - 6 = 0 →
  (p / (p*q + 2)) + (q / (p*r + 2)) + (r / (q*p + 2)) = 3/4 := by
sorry

end root_sum_fraction_l3584_358432


namespace evaluate_expression_l3584_358427

theorem evaluate_expression (x y z : ℚ) : 
  x = 1/4 → y = 1/3 → z = 12 → x^3 * y^4 * z = 1/432 := by
  sorry

end evaluate_expression_l3584_358427


namespace race_solution_l3584_358464

/-- Race between A and B from M to N and back -/
structure Race where
  distance : ℝ  -- Distance between M and N
  time_A : ℝ    -- Time taken by A
  time_B : ℝ    -- Time taken by B

/-- Conditions of the race -/
def race_conditions (r : Race) : Prop :=
  -- A reaches N sooner than B
  r.time_A < r.time_B
  -- A meets B 100 meters before N on the way back
  ∧ ∃ t : ℝ, t < r.time_A ∧ t * (r.distance / r.time_A) = (2 * r.distance - 100)
  -- A arrives at M 4 minutes earlier than B
  ∧ r.time_B = r.time_A + 4
  -- If A turns around at M, they meet B at 1/5 of the M to N distance
  ∧ ∃ t : ℝ, t < r.time_A ∧ t * (r.distance / r.time_A) = (1/5) * r.distance

/-- The theorem to be proved -/
theorem race_solution :
  ∃ r : Race, race_conditions r ∧ r.distance = 1000 ∧ r.time_A = 18 ∧ r.time_B = 22 := by
  sorry

end race_solution_l3584_358464


namespace ferris_wheel_rides_l3584_358435

theorem ferris_wheel_rides (rollercoaster_rides catapult_rides : ℕ) 
  (rollercoaster_cost catapult_cost ferris_wheel_cost total_tickets : ℕ) :
  rollercoaster_rides = 3 →
  catapult_rides = 2 →
  rollercoaster_cost = 4 →
  catapult_cost = 4 →
  ferris_wheel_cost = 1 →
  total_tickets = 21 →
  (total_tickets - (rollercoaster_rides * rollercoaster_cost + catapult_rides * catapult_cost)) / ferris_wheel_cost = 1 :=
by sorry

end ferris_wheel_rides_l3584_358435


namespace solution_implies_a_equals_six_l3584_358457

theorem solution_implies_a_equals_six :
  ∀ a : ℝ, (2 * 1 + 5 = 1 + a) → a = 6 := by
  sorry

end solution_implies_a_equals_six_l3584_358457


namespace initial_value_proof_l3584_358447

theorem initial_value_proof (increase_rate : ℝ) (final_value : ℝ) (years : ℕ) : 
  increase_rate = 1/8 →
  years = 2 →
  final_value = 8100 →
  final_value = 6400 * (1 + increase_rate)^years →
  6400 = 6400 := by sorry

end initial_value_proof_l3584_358447


namespace bankers_discount_calculation_l3584_358467

/-- Banker's discount calculation -/
theorem bankers_discount_calculation
  (true_discount : ℝ)
  (sum_due : ℝ)
  (h1 : true_discount = 60)
  (h2 : sum_due = 360) :
  true_discount + (true_discount^2 / sum_due) = 70 :=
by sorry

end bankers_discount_calculation_l3584_358467


namespace prob_diff_games_l3584_358469

/-- Probability of getting heads on a single toss of the biased coin -/
def p_heads : ℚ := 3/4

/-- Probability of getting tails on a single toss of the biased coin -/
def p_tails : ℚ := 1/4

/-- Probability of winning Game A -/
def p_win_game_a : ℚ := 
  4 * (p_heads^3 * p_tails) + p_heads^4

/-- Probability of winning Game B -/
def p_win_game_b : ℚ := 
  (p_heads^2 + p_tails^2)^2

/-- The difference in probabilities between winning Game A and Game B -/
theorem prob_diff_games : p_win_game_a - p_win_game_b = 89/256 := by
  sorry

end prob_diff_games_l3584_358469


namespace at_most_one_greater_than_one_l3584_358433

theorem at_most_one_greater_than_one (x y : ℝ) (h : x + y < 2) :
  ¬(x > 1 ∧ y > 1) := by
  sorry

end at_most_one_greater_than_one_l3584_358433


namespace complement_of_A_in_U_l3584_358472

def U : Finset ℕ := {2, 0, 1, 5}
def A : Finset ℕ := {0, 2}

theorem complement_of_A_in_U :
  (U \ A) = {1, 5} := by sorry

end complement_of_A_in_U_l3584_358472


namespace factorization_equality_l3584_358490

theorem factorization_equality (m n : ℝ) : 2*m*n^2 - 12*m*n + 18*m = 2*m*(n-3)^2 := by
  sorry

end factorization_equality_l3584_358490


namespace class_average_problem_l3584_358473

theorem class_average_problem (total_students : ℕ) 
  (high_score_students : ℕ) (zero_score_students : ℕ) 
  (high_score : ℕ) (class_average : ℕ) :
  total_students = 25 →
  high_score_students = 3 →
  zero_score_students = 5 →
  high_score = 95 →
  class_average = 42 →
  let remaining_students := total_students - (high_score_students + zero_score_students)
  let total_score := total_students * class_average
  let high_score_total := high_score_students * high_score
  let remaining_score := total_score - high_score_total
  remaining_score / remaining_students = 45 := by
sorry

end class_average_problem_l3584_358473


namespace data_analysis_l3584_358485

def data : List ℝ := [11, 10, 11, 13, 11, 13, 15]

def mode (l : List ℝ) : ℝ := sorry

def mean (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

def median (l : List ℝ) : ℝ := sorry

theorem data_analysis :
  mode data = 11 ∧
  mean data = 12 ∧
  variance data = 18/7 ∧
  median data = 11 := by sorry

end data_analysis_l3584_358485


namespace square_of_99_9_l3584_358401

theorem square_of_99_9 : (99.9 : ℝ)^2 = 10000 - 20 + 0.01 := by
  sorry

end square_of_99_9_l3584_358401


namespace discount_problem_l3584_358426

/-- Calculate the original price given the discounted price and discount rate -/
def originalPrice (discountedPrice : ℚ) (discountRate : ℚ) : ℚ :=
  discountedPrice / (1 - discountRate)

/-- The problem statement -/
theorem discount_problem (item1_discounted : ℚ) (item1_rate : ℚ)
                         (item2_discounted : ℚ) (item2_rate : ℚ)
                         (item3_discounted : ℚ) (item3_rate : ℚ) :
  item1_discounted = 4400 →
  item1_rate = 56 / 100 →
  item2_discounted = 3900 →
  item2_rate = 35 / 100 →
  item3_discounted = 2400 →
  item3_rate = 20 / 100 →
  originalPrice item1_discounted item1_rate +
  originalPrice item2_discounted item2_rate +
  originalPrice item3_discounted item3_rate = 19000 := by
  sorry

end discount_problem_l3584_358426


namespace tangent_sum_simplification_l3584_358414

theorem tangent_sum_simplification :
  (Real.tan (30 * π / 180) + Real.tan (40 * π / 180) + Real.tan (50 * π / 180) + Real.tan (60 * π / 180)) / Real.cos (20 * π / 180) = 8 * Real.sqrt 3 / 3 := by
  sorry

end tangent_sum_simplification_l3584_358414
