import Mathlib

namespace system_one_solution_system_two_solution_l2634_263407

-- System 1
theorem system_one_solution (x y : ℝ) : 
  2 * x - y = 1 ∧ 7 * x - 3 * y = 4 → x = 1 ∧ y = 1 := by sorry

-- System 2
theorem system_two_solution (x y : ℝ) : 
  x / 2 + y / 3 = 6 ∧ x - y = -3 → x = 6 ∧ y = 9 := by sorry

end system_one_solution_system_two_solution_l2634_263407


namespace expand_product_l2634_263487

theorem expand_product (x : ℝ) : (3*x - 4) * (2*x + 7) = 6*x^2 + 13*x - 28 := by
  sorry

end expand_product_l2634_263487


namespace geometric_sequence_ratio_l2634_263465

/-- Given a geometric sequence {a_n} with common ratio q = 2 and S_n as the sum of the first n terms,
    prove that S_4 / a_2 = 15/2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- Definition of geometric sequence
  (∀ n, S n = a 1 * (1 - q^n) / (1 - q)) →  -- Definition of sum of geometric sequence
  q = 2 →  -- Given condition
  S 4 / a 2 = 15 / 2 := by
sorry

end geometric_sequence_ratio_l2634_263465


namespace not_all_squares_congruent_l2634_263429

-- Define a square
structure Square where
  side_length : ℝ
  angle_measure : ℝ
  is_rectangle : Prop
  similar_to_all_squares : Prop

-- Define properties of squares
axiom square_angles (s : Square) : s.angle_measure = 90

axiom square_sides_equal (s : Square) : s.side_length > 0

axiom square_is_rectangle (s : Square) : s.is_rectangle = true

axiom squares_similar (s1 s2 : Square) : s1.similar_to_all_squares ∧ s2.similar_to_all_squares

-- Theorem to prove
theorem not_all_squares_congruent : ¬∀ (s1 s2 : Square), s1 = s2 := by
  sorry


end not_all_squares_congruent_l2634_263429


namespace peach_crate_pigeonhole_l2634_263474

/-- The number of crates of peaches -/
def total_crates : ℕ := 154

/-- The minimum number of peaches in a crate -/
def min_peaches : ℕ := 130

/-- The maximum number of peaches in a crate -/
def max_peaches : ℕ := 160

/-- The number of possible peach counts per crate -/
def possible_counts : ℕ := max_peaches - min_peaches + 1

theorem peach_crate_pigeonhole :
  ∃ (n : ℕ), n = 4 ∧
  (∀ (m : ℕ), m > n →
    ∃ (distribution : Fin total_crates → ℕ),
      (∀ i, min_peaches ≤ distribution i ∧ distribution i ≤ max_peaches) ∧
      (∀ k, ¬(∃ (S : Finset (Fin total_crates)), S.card = m ∧ (∀ i ∈ S, distribution i = k)))) ∧
  (∃ (distribution : Fin total_crates → ℕ),
    (∀ i, min_peaches ≤ distribution i ∧ distribution i ≤ max_peaches) →
    ∃ (k : ℕ) (S : Finset (Fin total_crates)), S.card = n ∧ (∀ i ∈ S, distribution i = k)) := by
  sorry


end peach_crate_pigeonhole_l2634_263474


namespace ball_probability_l2634_263411

theorem ball_probability (m : ℕ) : 
  (3 : ℝ) / ((m : ℝ) + 3) = (1 : ℝ) / 4 → m = 9 := by
  sorry

end ball_probability_l2634_263411


namespace equality_of_fractions_l2634_263498

theorem equality_of_fractions (x y z k : ℝ) 
  (h : 9 / (x + y) = k / (y + z) ∧ k / (y + z) = 15 / (x - z)) : k = 0 := by
  sorry

end equality_of_fractions_l2634_263498


namespace total_revenue_is_2176_l2634_263434

def kitten_price : ℕ := 80
def puppy_price : ℕ := 150
def rabbit_price : ℕ := 45
def guinea_pig_price : ℕ := 30

def kitten_count : ℕ := 10
def puppy_count : ℕ := 8
def rabbit_count : ℕ := 4
def guinea_pig_count : ℕ := 6

def discount_rate : ℚ := 1/10

def total_revenue : ℚ := 
  (kitten_count * kitten_price + 
   puppy_count * puppy_price + 
   rabbit_count * rabbit_price + 
   guinea_pig_count * guinea_pig_price : ℚ) - 
  (min kitten_count puppy_count * discount_rate * (kitten_price + puppy_price))

theorem total_revenue_is_2176 : total_revenue = 2176 := by
  sorry

end total_revenue_is_2176_l2634_263434


namespace solution_set_equality_l2634_263495

-- Define the set of real numbers x that satisfy the inequality
def solution_set : Set ℝ := {x : ℝ | (x + 3) / (4 - x) ≥ 0 ∧ x ≠ 4}

-- Theorem stating that the solution set is equal to the interval [-3, 4)
theorem solution_set_equality : solution_set = Set.Icc (-3) 4 \ {4} :=
sorry

end solution_set_equality_l2634_263495


namespace absolute_value_non_negative_l2634_263415

theorem absolute_value_non_negative (a : ℝ) : ¬(|a| < 0) := by
  sorry

end absolute_value_non_negative_l2634_263415


namespace valid_arrangements_l2634_263451

/-- The number of letters to be arranged -/
def n : ℕ := 8

/-- The number of pairs of repeated letters -/
def k : ℕ := 3

/-- The total number of unrestricted arrangements -/
def total_arrangements : ℕ := n.factorial / (2^k)

/-- The number of arrangements with one pair of identical letters together -/
def arrangements_one_pair : ℕ := k * ((n-1).factorial / (2^(k-1)))

/-- The number of arrangements with two pairs of identical letters together -/
def arrangements_two_pairs : ℕ := (k.choose 2) * ((n-2).factorial / (2^(k-2)))

/-- The number of arrangements with three pairs of identical letters together -/
def arrangements_three_pairs : ℕ := (n-3).factorial

/-- The theorem stating the number of valid arrangements -/
theorem valid_arrangements :
  total_arrangements - arrangements_one_pair + arrangements_two_pairs - arrangements_three_pairs = 2220 :=
sorry

end valid_arrangements_l2634_263451


namespace improved_running_distance_l2634_263406

/-- Proves that a runner who can cover 40 yards in 5 seconds and improves their speed by 40% will cover 112 yards in 10 seconds -/
theorem improved_running_distance 
  (initial_distance : ℝ) 
  (initial_time : ℝ) 
  (improvement_percentage : ℝ) 
  (new_time : ℝ) :
  initial_distance = 40 ∧ 
  initial_time = 5 ∧ 
  improvement_percentage = 40 ∧ 
  new_time = 10 → 
  (initial_distance * (1 + improvement_percentage / 100) * (new_time / initial_time)) = 112 :=
by sorry

end improved_running_distance_l2634_263406


namespace quadratic_root_transformation_l2634_263464

/-- Given a quadratic equation 2Ax^2 + 3Bx + 4C = 0 with roots r and s,
    prove that the value of p in the equation x^2 + px + q = 0 with roots r^2 and s^2
    is equal to (16AC - 9B^2) / (4A^2) -/
theorem quadratic_root_transformation (A B C : ℝ) (r s : ℝ) :
  (2 * A * r ^ 2 + 3 * B * r + 4 * C = 0) →
  (2 * A * s ^ 2 + 3 * B * s + 4 * C = 0) →
  ∃ q : ℝ, r ^ 2 ^ 2 + ((16 * A * C - 9 * B ^ 2) / (4 * A ^ 2)) * r ^ 2 + q = 0 ∧
           s ^ 2 ^ 2 + ((16 * A * C - 9 * B ^ 2) / (4 * A ^ 2)) * s ^ 2 + q = 0 :=
by sorry

end quadratic_root_transformation_l2634_263464


namespace problem_one_problem_two_l2634_263450

-- Problem 1
theorem problem_one : -1^4 - 7 / (2 - (-3)^2) = 0 := by sorry

-- Problem 2
-- Define a custom type for degrees and minutes
structure DegreeMinute where
  degrees : Int
  minutes : Int

-- Define addition for DegreeMinute
def add_degree_minute (a b : DegreeMinute) : DegreeMinute :=
  let total_minutes := a.minutes + b.minutes
  let extra_degrees := total_minutes / 60
  let remaining_minutes := total_minutes % 60
  ⟨a.degrees + b.degrees + extra_degrees, remaining_minutes⟩

-- Define subtraction for DegreeMinute
def sub_degree_minute (a b : DegreeMinute) : DegreeMinute :=
  let total_minutes_a := a.degrees * 60 + a.minutes
  let total_minutes_b := b.degrees * 60 + b.minutes
  let diff_minutes := total_minutes_a - total_minutes_b
  ⟨diff_minutes / 60, diff_minutes % 60⟩

-- Define multiplication of DegreeMinute by Int
def mul_degree_minute (a : DegreeMinute) (n : Int) : DegreeMinute :=
  let total_minutes := (a.degrees * 60 + a.minutes) * n
  ⟨total_minutes / 60, total_minutes % 60⟩

theorem problem_two :
  sub_degree_minute
    (add_degree_minute ⟨56, 17⟩ ⟨12, 45⟩)
    (mul_degree_minute ⟨16, 21⟩ 4) = ⟨3, 38⟩ := by sorry

end problem_one_problem_two_l2634_263450


namespace two_thousandth_digit_sum_l2634_263438

def is_valid_sequence (seq : List Nat) : Prop :=
  seq.length = 2000 ∧
  seq.head? = some 3 ∧
  ∀ i, i < 1999 → (seq.get? i).isSome ∧ (seq.get? (i+1)).isSome →
    (17 ∣ (seq.get! i * 10 + seq.get! (i+1))) ∨ (23 ∣ (seq.get! i * 10 + seq.get! (i+1)))

theorem two_thousandth_digit_sum (seq : List Nat) (a b : Nat) :
  is_valid_sequence seq →
  (seq.get? 1999 = some a ∨ seq.get? 1999 = some b) →
  a + b = 7 := by
  sorry

end two_thousandth_digit_sum_l2634_263438


namespace sqrt_inequality_and_floor_l2634_263481

theorem sqrt_inequality_and_floor (n : ℕ) : 
  (Real.sqrt (n + 1) + 2 * Real.sqrt n < Real.sqrt (9 * n + 3)) ∧
  ¬∃ n : ℕ, ⌊Real.sqrt (n + 1) + 2 * Real.sqrt n⌋ < ⌊Real.sqrt (9 * n + 3)⌋ := by
  sorry

end sqrt_inequality_and_floor_l2634_263481


namespace charity_donation_l2634_263491

def cassandra_pennies : ℕ := 5000
def james_difference : ℕ := 276

def total_donation : ℕ := cassandra_pennies + (cassandra_pennies - james_difference)

theorem charity_donation :
  total_donation = 9724 :=
sorry

end charity_donation_l2634_263491


namespace minimum_point_implies_b_greater_than_one_l2634_263455

theorem minimum_point_implies_b_greater_than_one (a b : ℝ) (hb : b ≠ 0) :
  let f := fun x : ℝ ↦ (x - b) * (x^2 + a*x + b)
  (∀ x, f b ≤ f x) →
  b > 1 := by
sorry

end minimum_point_implies_b_greater_than_one_l2634_263455


namespace larger_number_in_ratio_l2634_263403

theorem larger_number_in_ratio (a b : ℚ) : 
  a / b = 8 / 3 → a + b = 143 → max a b = 104 := by
  sorry

end larger_number_in_ratio_l2634_263403


namespace multiply_squared_terms_l2634_263440

theorem multiply_squared_terms (a : ℝ) : 3 * a^2 * (2 * a^2) = 6 * a^4 := by
  sorry

end multiply_squared_terms_l2634_263440


namespace triangle_properties_l2634_263448

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : abc.a = 3)
  (h2 : abc.c = 2)
  (h3 : Real.sin abc.A = Real.cos (π/2 - abc.B)) : 
  Real.cos abc.C = 7/9 ∧ 
  (1/2 * abc.a * abc.b * Real.sin abc.C) = 2 * Real.sqrt 2 := by
  sorry

end triangle_properties_l2634_263448


namespace quadratic_form_equivalence_l2634_263445

theorem quadratic_form_equivalence (b : ℝ) (m : ℝ) : 
  b > 0 → 
  (∀ x, x^2 + b*x + 44 = (x + m)^2 + 8) → 
  b = 12 := by
sorry

end quadratic_form_equivalence_l2634_263445


namespace derivative_of_f_l2634_263497

-- Define the function f
def f (x : ℝ) : ℝ := 2016 * x^2

-- State the theorem
theorem derivative_of_f (x : ℝ) :
  deriv f x = 4032 * x := by sorry

-- Note: The 'deriv' function in Lean represents the derivative.

end derivative_of_f_l2634_263497


namespace units_digit_of_product_l2634_263472

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- A number is composite if it has a factor other than 1 and itself -/
def isComposite (n : ℕ) : Prop := ∃ k m : ℕ, 1 < k ∧ k < n ∧ n = k * m

theorem units_digit_of_product :
  isComposite 9 ∧ isComposite 10 ∧ isComposite 12 →
  unitsDigit (9 * 10 * 12) = 0 := by
  sorry

end units_digit_of_product_l2634_263472


namespace weight_replacement_l2634_263467

theorem weight_replacement (n : ℕ) (avg_increase : ℝ) (new_weight : ℝ) :
  n = 10 →
  avg_increase = 6.3 →
  new_weight = 128 →
  ∃ (old_weight : ℝ),
    old_weight = new_weight - n * avg_increase ∧
    old_weight = 65 :=
by sorry

end weight_replacement_l2634_263467


namespace square_area_from_perimeter_l2634_263425

-- Define the square
def Square (perimeter : ℝ) : Type :=
  { side : ℝ // perimeter = 4 * side }

-- Theorem statement
theorem square_area_from_perimeter (perimeter : ℝ) (h : perimeter = 80) :
  ∃ (s : Square perimeter), (s.val)^2 = 400 := by
  sorry

end square_area_from_perimeter_l2634_263425


namespace parabola_y_relationship_l2634_263463

/-- A parabola defined by y = 2(x-1)² + c passing through three points -/
structure Parabola where
  c : ℝ
  y₁ : ℝ
  y₂ : ℝ
  y₃ : ℝ
  eq_y₁ : y₁ = 2 * (-2 - 1)^2 + c
  eq_y₂ : y₂ = 2 * (0 - 1)^2 + c
  eq_y₃ : y₃ = 2 * (5/3 - 1)^2 + c

/-- Theorem stating the relationship between y₁, y₂, and y₃ for the given parabola -/
theorem parabola_y_relationship (p : Parabola) : p.y₁ > p.y₂ ∧ p.y₂ > p.y₃ := by
  sorry

end parabola_y_relationship_l2634_263463


namespace equidistant_complex_function_l2634_263444

theorem equidistant_complex_function (a b : ℝ) :
  (∀ z : ℂ, ‖(a + b * I) * z^2 - z^2‖ = ‖(a + b * I) * z^2‖) →
  ‖(a + b * I)‖ = 10 →
  b^2 = 99.75 := by sorry

end equidistant_complex_function_l2634_263444


namespace alpha_plus_beta_equals_118_l2634_263437

theorem alpha_plus_beta_equals_118 :
  ∀ α β : ℝ,
  (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 96*x + 2209) / (x^2 + 63*x - 3969)) →
  α + β = 118 := by
sorry

end alpha_plus_beta_equals_118_l2634_263437


namespace total_animals_l2634_263478

theorem total_animals (a b c : ℕ) (ha : a = 6) (hb : b = 8) (hc : c = 4) :
  a + b + c = 18 := by
  sorry

end total_animals_l2634_263478


namespace total_baseball_cards_l2634_263430

theorem total_baseball_cards (rob_doubles jess_doubles alex_doubles rob_total alex_total : ℕ) : 
  rob_doubles = 8 →
  jess_doubles = 40 →
  alex_doubles = 12 →
  rob_total = 24 →
  alex_total = 48 →
  rob_doubles * 3 = rob_total →
  jess_doubles = 5 * rob_doubles →
  alex_total = 2 * rob_total →
  alex_doubles * 4 = alex_total →
  rob_total + jess_doubles + alex_total = 112 := by
  sorry

end total_baseball_cards_l2634_263430


namespace lawn_mowing_time_l2634_263490

theorem lawn_mowing_time (mary_rate tom_rate : ℚ) (mary_time : ℚ) : 
  mary_rate = 1/3 →
  tom_rate = 1/6 →
  mary_time = 1 →
  (1 - mary_rate * mary_time) / tom_rate = 4 := by
sorry

end lawn_mowing_time_l2634_263490


namespace hundred_with_fewer_threes_l2634_263458

/-- An arithmetic expression using threes and basic operations -/
inductive Expr
  | three : Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

/-- Count the number of threes in an expression -/
def countThrees : Expr → Nat
  | Expr.three => 1
  | Expr.add e1 e2 => countThrees e1 + countThrees e2
  | Expr.sub e1 e2 => countThrees e1 + countThrees e2
  | Expr.mul e1 e2 => countThrees e1 + countThrees e2
  | Expr.div e1 e2 => countThrees e1 + countThrees e2

/-- Evaluate an expression to a rational number -/
def eval : Expr → ℚ
  | Expr.three => 3
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2

/-- There exists an expression using fewer than ten threes that evaluates to 100 -/
theorem hundred_with_fewer_threes : ∃ e : Expr, countThrees e < 10 ∧ eval e = 100 := by
  sorry

end hundred_with_fewer_threes_l2634_263458


namespace quadratic_equal_roots_l2634_263479

/-- 
Given a quadratic equation x^2 - mx + m - 1 = 0 with two equal real roots,
prove that m = 2 and the roots are x = 1
-/
theorem quadratic_equal_roots (m : ℝ) :
  (∃ x : ℝ, x^2 - m*x + m - 1 = 0 ∧ 
   ∀ y : ℝ, y^2 - m*y + m - 1 = 0 → y = x) →
  m = 2 ∧ ∃ x : ℝ, x^2 - m*x + m - 1 = 0 ∧ x = 1 := by
  sorry


end quadratic_equal_roots_l2634_263479


namespace common_tangents_count_l2634_263492

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 4*y + 1 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 1 = 0

-- Define the number of common tangents
def num_common_tangents (C₁ C₂ : (ℝ → ℝ → Prop)) : ℕ := sorry

-- Theorem statement
theorem common_tangents_count :
  num_common_tangents C₁ C₂ = 3 := by sorry

end common_tangents_count_l2634_263492


namespace correct_quadratic_equation_l2634_263413

def is_root (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

theorem correct_quadratic_equation (a b c : ℝ) :
  (∃ b₁ c₁, is_root 1 b₁ c₁ 7 ∧ is_root 1 b₁ c₁ 3) →
  (∃ b₂ c₂, is_root 1 b₂ c₂ 11 ∧ is_root 1 b₂ c₂ (-1)) →
  (a = 1 ∧ b = -10 ∧ c = 32) :=
sorry

end correct_quadratic_equation_l2634_263413


namespace symmetry_point_x_axis_l2634_263442

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define symmetry with respect to x-axis
def symmetricToXAxis (p q : Point3D) : Prop :=
  q.x = p.x ∧ q.y = -p.y ∧ q.z = -p.z

theorem symmetry_point_x_axis :
  let M : Point3D := ⟨1, 2, 3⟩
  let N : Point3D := ⟨1, -2, -3⟩
  symmetricToXAxis M N := by sorry

end symmetry_point_x_axis_l2634_263442


namespace consecutive_numbers_sum_l2634_263456

theorem consecutive_numbers_sum (a : ℤ) : 
  (a + (a + 1) + (a + 2) = 184) ∧
  (a + (a + 1) + (a + 3) = 201) ∧
  (a + (a + 2) + (a + 3) = 212) ∧
  ((a + 1) + (a + 2) + (a + 3) = 226) →
  (a + 3 = 70) := by
sorry

end consecutive_numbers_sum_l2634_263456


namespace additional_distance_with_speed_increase_l2634_263459

/-- Calculates the additional distance traveled when increasing speed for a given initial distance and speeds. -/
theorem additional_distance_with_speed_increase 
  (actual_speed : ℝ) 
  (faster_speed : ℝ) 
  (actual_distance : ℝ) 
  (h1 : actual_speed > 0)
  (h2 : faster_speed > actual_speed)
  (h3 : actual_distance > 0)
  : let time := actual_distance / actual_speed
    let faster_distance := faster_speed * time
    faster_distance - actual_distance = 20 :=
by sorry

end additional_distance_with_speed_increase_l2634_263459


namespace boarding_students_change_l2634_263486

theorem boarding_students_change (initial : ℝ) (increase_rate : ℝ) (decrease_rate : ℝ) 
  (h1 : increase_rate = 0.2) 
  (h2 : decrease_rate = 0.2) : 
  initial * (1 + increase_rate) * (1 - decrease_rate) = initial * 0.96 :=
by sorry

end boarding_students_change_l2634_263486


namespace solution_pairs_l2634_263489

theorem solution_pairs : ∀ x y : ℝ, 
  (x + y + 4 = (12*x + 11*y) / (x^2 + y^2) ∧ 
   y - x + 3 = (11*x - 12*y) / (x^2 + y^2)) ↔ 
  ((x = 2 ∧ y = 1) ∨ (x = -2.5 ∧ y = -4.5)) :=
by sorry

end solution_pairs_l2634_263489


namespace mean_temperature_l2634_263454

def temperatures : List ℝ := [75, 78, 80, 76, 77]

theorem mean_temperature : (temperatures.sum / temperatures.length : ℝ) = 77.2 := by
  sorry

end mean_temperature_l2634_263454


namespace cubic_inequality_l2634_263405

theorem cubic_inequality (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) :
  2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b := by
  sorry

end cubic_inequality_l2634_263405


namespace trigonometric_function_property_l2634_263480

theorem trigonometric_function_property (f : ℝ → ℝ) :
  (∀ α, f (Real.cos α) = Real.sin α) → f 1 = 0 ∧ f (-1) = 0 := by
  sorry

end trigonometric_function_property_l2634_263480


namespace complex_equation_solution_l2634_263427

/-- Given that (1-i)z = 3+i, prove that z = 1 + 2i -/
theorem complex_equation_solution (z : ℂ) (h : (1 - Complex.I) * z = 3 + Complex.I) : 
  z = 1 + 2 * Complex.I := by
  sorry

end complex_equation_solution_l2634_263427


namespace complex_percentage_calculation_l2634_263473

theorem complex_percentage_calculation : 
  let a := 0.15 * 50
  let b := 0.25 * 75
  let c := -0.10 * 120
  let sum := a + b + c
  let d := -0.05 * 150
  2.5 * d - (1/3) * sum = -23.5 := by
sorry

end complex_percentage_calculation_l2634_263473


namespace income_for_given_tax_l2634_263439

/-- Proves that given the tax conditions, an income of $56,000 results in a total tax of $8,000 --/
theorem income_for_given_tax : ∀ (I : ℝ),
  (min I 40000 * 0.12 + max (I - 40000) 0 * 0.20 = 8000) → I = 56000 := by
  sorry

end income_for_given_tax_l2634_263439


namespace outlet_pipe_time_l2634_263453

theorem outlet_pipe_time (inlet1 inlet2 outlet : ℚ) 
  (h1 : inlet1 = 1 / 18)
  (h2 : inlet2 = 1 / 20)
  (h3 : inlet1 + inlet2 - outlet = 1 / 12) :
  outlet = 1 / 45 := by
  sorry

end outlet_pipe_time_l2634_263453


namespace selling_price_with_loss_l2634_263414

def cost_price : ℝ := 1800
def loss_percentage : ℝ := 10

theorem selling_price_with_loss (cp : ℝ) (lp : ℝ) : 
  cp * (1 - lp / 100) = 1620 :=
by sorry

end selling_price_with_loss_l2634_263414


namespace optimal_pasture_length_l2634_263494

/-- Represents a rectangular cow pasture -/
structure Pasture where
  width : ℝ  -- Width of the pasture (perpendicular to the barn)
  length : ℝ  -- Length of the pasture (parallel to the barn)

/-- Calculates the area of the pasture -/
def Pasture.area (p : Pasture) : ℝ := p.width * p.length

/-- Theorem: The optimal length of the pasture that maximizes the area -/
theorem optimal_pasture_length (total_fence : ℝ) (barn_length : ℝ) :
  total_fence = 240 →
  barn_length = 600 →
  ∃ (optimal : Pasture),
    optimal.length = 120 ∧
    optimal.width = (total_fence - optimal.length) / 2 ∧
    ∀ (p : Pasture),
      p.length + 2 * p.width = total_fence →
      p.area ≤ optimal.area := by
  sorry

end optimal_pasture_length_l2634_263494


namespace isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l2634_263460

/-- An isosceles triangle with two sides of length 12 and one side of length 17 has a perimeter of 41 -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → Prop :=
  fun (equal_side : ℝ) (third_side : ℝ) =>
    equal_side = 12 ∧ third_side = 17 →
    2 * equal_side + third_side = 41

/-- Proof of the theorem -/
theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 12 17 :=
by
  sorry

#check isosceles_triangle_perimeter_proof

end isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l2634_263460


namespace sample_size_l2634_263441

theorem sample_size (n : ℕ) (f₁ f₂ f₃ f₄ f₅ f₆ : ℕ) : 
  f₁ + f₂ + f₃ + f₄ + f₅ + f₆ = n →
  f₁ + f₂ + f₃ = 27 →
  2 * f₆ = f₁ →
  3 * f₆ = f₂ →
  4 * f₆ = f₃ →
  6 * f₆ = f₄ →
  4 * f₆ = f₅ →
  n = 60 := by
sorry

end sample_size_l2634_263441


namespace binomial_prob_properties_l2634_263482

/-- A binomial distribution with parameters n and p -/
structure BinomialDist where
  n : ℕ+
  p : ℝ
  h_p_pos : 0 < p
  h_p_lt_one : p < 1

/-- The probability that X is odd in a binomial distribution -/
noncomputable def prob_odd (b : BinomialDist) : ℝ :=
  (1 - (1 - 2*b.p)^b.n.val) / 2

/-- The probability that X is even in a binomial distribution -/
noncomputable def prob_even (b : BinomialDist) : ℝ :=
  1 - prob_odd b

theorem binomial_prob_properties (b : BinomialDist) :
  (prob_odd b + prob_even b = 1) ∧
  (b.p = 1/2 → prob_odd b = prob_even b) ∧
  (0 < b.p ∧ b.p < 1/2 → ∀ m : ℕ+, m < b.n → prob_odd ⟨m, b.p, b.h_p_pos, b.h_p_lt_one⟩ < prob_odd b) :=
by sorry

end binomial_prob_properties_l2634_263482


namespace conditional_without_else_l2634_263421

-- Define the structure of conditional statements
inductive ConditionalStatement
  | ifThenElse (condition : Prop) (thenStmt : Prop) (elseStmt : Prop)
  | ifThen (condition : Prop) (thenStmt : Prop)

-- Define a property that checks if a conditional statement has an ELSE part
def hasElsePart : ConditionalStatement → Prop
  | ConditionalStatement.ifThenElse _ _ _ => true
  | ConditionalStatement.ifThen _ _ => false

-- Theorem stating that there exists a conditional statement without an ELSE part
theorem conditional_without_else : ∃ (stmt : ConditionalStatement), ¬(hasElsePart stmt) := by
  sorry


end conditional_without_else_l2634_263421


namespace new_average_after_doubling_l2634_263484

/-- Theorem: New average after doubling marks -/
theorem new_average_after_doubling (n : ℕ) (original_average : ℝ) :
  n > 0 →
  let total_marks := n * original_average
  let doubled_marks := 2 * total_marks
  let new_average := doubled_marks / n
  new_average = 2 * original_average := by
  sorry

/-- Given problem as an example -/
example : 
  let n : ℕ := 25
  let original_average : ℝ := 70
  let total_marks := n * original_average
  let doubled_marks := 2 * total_marks
  let new_average := doubled_marks / n
  new_average = 140 := by
  sorry

end new_average_after_doubling_l2634_263484


namespace consecutive_multiples_problem_l2634_263468

/-- Given a set of 50 consecutive multiples of a number, prove that the number is 2 -/
theorem consecutive_multiples_problem (n : ℕ) (s : Set ℕ) : 
  (∃ k : ℕ, s = {k * n | k ∈ Finset.range 50}) →  -- s is a set of 50 consecutive multiples of n
  (56 ∈ s) →  -- The smallest number in s is 56
  (154 ∈ s) →  -- The greatest number in s is 154
  (∀ x ∈ s, 56 ≤ x ∧ x ≤ 154) →  -- All elements in s are between 56 and 154
  n = 2 := by
  sorry

end consecutive_multiples_problem_l2634_263468


namespace min_value_a_l2634_263443

theorem min_value_a (a : ℝ) : (∀ x ∈ Set.Ioc (0 : ℝ) (1/2), x^2 + a*x + 1 ≥ 0) ↔ a ≥ -5/2 :=
sorry

end min_value_a_l2634_263443


namespace linear_function_m_value_l2634_263426

/-- Linear function passing through a point -/
def linear_function (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x - 4

/-- Theorem: For a linear function y = (m-1)x - 4 passing through (2, 4), m = 5 -/
theorem linear_function_m_value :
  ∃ (m : ℝ), linear_function m 2 = 4 ∧ m = 5 := by
  sorry

end linear_function_m_value_l2634_263426


namespace find_value_of_b_l2634_263404

/-- Given a configuration of numbers in circles with specific properties, prove the value of b. -/
theorem find_value_of_b (circle_sum : ℕ) (total_circles : ℕ) (total_sum : ℕ) 
  (overlap_sum : ℕ → ℕ → ℕ) (d_circle_sum : ℕ → ℕ) 
  (h1 : circle_sum = 21)
  (h2 : total_circles = 5)
  (h3 : total_sum = 69)
  (h4 : ∀ (b d : ℕ), overlap_sum b d = 2 + 8 + 9 + b + d)
  (h5 : ∀ (d : ℕ), d_circle_sum d = d + 5 + 9)
  (h6 : ∀ (d : ℕ), d_circle_sum d = circle_sum) :
  ∃ (b : ℕ), b = 10 := by
  sorry

end find_value_of_b_l2634_263404


namespace locus_and_max_dot_product_l2634_263420

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 8

-- Define point N
def N : ℝ × ℝ := (0, -1)

-- Define the locus C
def locus_C (x y : ℝ) : Prop := y^2 / 2 + x^2 = 1

-- Define the dot product OA · AN
def dot_product (x y : ℝ) : ℝ := -x^2 - y^2 - y

-- Theorem statement
theorem locus_and_max_dot_product :
  ∀ (x y : ℝ),
    (∃ (px py : ℝ), circle_M px py ∧
      (x - (px + 0) / 2)^2 + (y - (py + -1) / 2)^2 = ((px - 0)^2 + (py - -1)^2) / 4) →
    locus_C x y ∧
    (∀ (ax ay : ℝ), locus_C ax ay → dot_product ax ay ≤ -1/2) ∧
    (∃ (ax ay : ℝ), locus_C ax ay ∧ dot_product ax ay = -1/2) :=
by sorry


end locus_and_max_dot_product_l2634_263420


namespace train_length_calculation_l2634_263401

/-- Prove that given a train traveling at a certain speed that crosses a bridge in a given time, 
    and the total length of the bridge and train is known, we can calculate the length of the train. -/
theorem train_length_calculation 
  (train_speed : ℝ) 
  (crossing_time : ℝ) 
  (total_length : ℝ) 
  (h1 : train_speed = 45) -- km/hr
  (h2 : crossing_time = 30 / 3600) -- 30 seconds converted to hours
  (h3 : total_length = 195) -- meters
  : ∃ (train_length : ℝ), train_length = 180 := by
  sorry

#check train_length_calculation

end train_length_calculation_l2634_263401


namespace photographs_eighteen_hours_ago_l2634_263417

theorem photographs_eighteen_hours_ago (photos_18h_ago : ℕ) : 
  (photos_18h_ago : ℚ) + 0.8 * (photos_18h_ago : ℚ) = 180 →
  photos_18h_ago = 100 := by
sorry

end photographs_eighteen_hours_ago_l2634_263417


namespace smallest_gcd_multiple_l2634_263461

theorem smallest_gcd_multiple (m n : ℕ+) (h : Nat.gcd m.val n.val = 12) :
  ∃ (k : ℕ+), k.val = Nat.gcd (8 * m.val) (18 * n.val) ∧ 
  ∀ (l : ℕ+), l.val = Nat.gcd (8 * m.val) (18 * n.val) → k ≤ l ∧ k.val = 24 :=
by sorry

end smallest_gcd_multiple_l2634_263461


namespace garden_area_theorem_l2634_263423

/-- Represents a rectangular garden with given properties -/
structure RectangularGarden where
  length : ℝ
  width : ℝ
  perimeter_walk_count : ℕ
  length_walk_count : ℕ
  total_distance : ℝ

/-- The theorem stating the area of the garden given the conditions -/
theorem garden_area_theorem (g : RectangularGarden) 
  (h1 : g.perimeter_walk_count = 20)
  (h2 : g.length_walk_count = 50)
  (h3 : g.total_distance = 1500)
  (h4 : 2 * (g.length + g.width) = g.total_distance / g.perimeter_walk_count)
  (h5 : g.length = g.total_distance / g.length_walk_count) :
  g.length * g.width = 225 := by
  sorry

#check garden_area_theorem

end garden_area_theorem_l2634_263423


namespace isosceles_triangle_angles_l2634_263496

-- Define an isosceles triangle with one angle of 70°
structure IsoscelesTriangle :=
  (angle1 : Real)
  (angle2 : Real)
  (angle3 : Real)
  (isIsosceles : (angle1 = angle2) ∨ (angle1 = angle3) ∨ (angle2 = angle3))
  (has70Degree : angle1 = 70 ∨ angle2 = 70 ∨ angle3 = 70)
  (sumIs180 : angle1 + angle2 + angle3 = 180)

-- Theorem statement
theorem isosceles_triangle_angles (t : IsoscelesTriangle) :
  (t.angle1 = 55 ∧ t.angle2 = 55 ∧ t.angle3 = 70) ∨
  (t.angle1 = 55 ∧ t.angle2 = 70 ∧ t.angle3 = 55) ∨
  (t.angle1 = 70 ∧ t.angle2 = 55 ∧ t.angle3 = 55) ∨
  (t.angle1 = 70 ∧ t.angle2 = 70 ∧ t.angle3 = 40) ∨
  (t.angle1 = 70 ∧ t.angle2 = 40 ∧ t.angle3 = 70) ∨
  (t.angle1 = 40 ∧ t.angle2 = 70 ∧ t.angle3 = 70) :=
by
  sorry

end isosceles_triangle_angles_l2634_263496


namespace sqrt_x_minus_3_real_l2634_263433

theorem sqrt_x_minus_3_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 3) ↔ x ≥ 3 := by
  sorry

end sqrt_x_minus_3_real_l2634_263433


namespace factorial_division_l2634_263428

theorem factorial_division :
  (9 : ℕ).factorial / (4 : ℕ).factorial = 15120 :=
by
  have h1 : (9 : ℕ).factorial = 362880 := by sorry
  sorry

end factorial_division_l2634_263428


namespace factorization_problems_l2634_263435

theorem factorization_problems :
  (∀ x y : ℝ, xy - 1 - x + y = (y - 1) * (x + 1)) ∧
  (∀ a b : ℝ, (a^2 + b^2)^2 - 4 * a^2 * b^2 = (a + b)^2 * (a - b)^2) :=
by sorry

end factorization_problems_l2634_263435


namespace watch_sale_price_l2634_263477

/-- The final sale price of a watch after two consecutive discounts --/
theorem watch_sale_price (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  initial_price = 20 ∧ discount1 = 0.20 ∧ discount2 = 0.25 →
  initial_price * (1 - discount1) * (1 - discount2) = 12 := by
  sorry

end watch_sale_price_l2634_263477


namespace continued_fraction_equality_l2634_263457

theorem continued_fraction_equality : 
  2 + (3 / (2 + (5 / (4 + (7 / 3))))) = 91 / 19 := by sorry

end continued_fraction_equality_l2634_263457


namespace helen_raisin_cookies_l2634_263475

/-- Represents the number of cookies baked --/
structure CookieCount where
  yesterday_chocolate : ℕ
  yesterday_raisin : ℕ
  today_chocolate : ℕ
  today_raisin : ℕ
  total_chocolate : ℕ

/-- Helen's cookie baking scenario --/
def helen_cookies : CookieCount where
  yesterday_chocolate := 527
  yesterday_raisin := 527  -- This is what we want to prove
  today_chocolate := 554
  today_raisin := 554
  total_chocolate := 1081

/-- Theorem stating that Helen baked 527 raisin cookies yesterday --/
theorem helen_raisin_cookies : 
  helen_cookies.yesterday_raisin = 527 := by
  sorry

#check helen_raisin_cookies

end helen_raisin_cookies_l2634_263475


namespace two_bishops_placement_l2634_263436

/-- Represents a chessboard with 8 rows and 8 columns -/
structure Chessboard :=
  (rows : Nat)
  (columns : Nat)
  (total_squares : Nat)
  (white_squares : Nat)
  (black_squares : Nat)

/-- Represents the number of ways to place two bishops on a chessboard -/
def bishop_placements (board : Chessboard) : Nat :=
  board.white_squares * (board.black_squares - board.rows)

/-- Theorem stating the number of ways to place two bishops on a chessboard -/
theorem two_bishops_placement (board : Chessboard) 
  (h1 : board.rows = 8)
  (h2 : board.columns = 8)
  (h3 : board.total_squares = board.rows * board.columns)
  (h4 : board.white_squares = board.total_squares / 2)
  (h5 : board.black_squares = board.total_squares / 2) :
  bishop_placements board = 768 := by
  sorry

#eval bishop_placements {rows := 8, columns := 8, total_squares := 64, white_squares := 32, black_squares := 32}

end two_bishops_placement_l2634_263436


namespace jim_unknown_row_trees_l2634_263410

/-- Represents the production of a lemon grove over 5 years -/
structure LemonGrove where
  normal_production : ℕ  -- lemons per year for a normal tree
  increase_percent : ℕ   -- percentage increase for Jim's trees
  known_row : ℕ          -- number of trees in the known row
  total_production : ℕ   -- total lemons produced in 5 years

/-- Calculates the number of trees in the unknown row of Jim's lemon grove -/
def unknown_row_trees (grove : LemonGrove) : ℕ :=
  let jim_tree_production := grove.normal_production * (100 + grove.increase_percent) / 100
  let total_trees := grove.total_production / (jim_tree_production * 5)
  total_trees - grove.known_row

/-- Theorem stating the number of trees in the unknown row of Jim's lemon grove -/
theorem jim_unknown_row_trees :
  let grove := LemonGrove.mk 60 50 30 675000
  unknown_row_trees grove = 1470 := by
  sorry

end jim_unknown_row_trees_l2634_263410


namespace tan_addition_special_case_l2634_263422

theorem tan_addition_special_case (x : Real) (h : Real.tan x = 1/2) :
  Real.tan (x + π/3) = 7 + 4 * Real.sqrt 3 := by
  sorry

end tan_addition_special_case_l2634_263422


namespace fahrenheit_celsius_conversion_l2634_263409

theorem fahrenheit_celsius_conversion (C F : ℚ) : 
  C = (4/7) * (F - 32) → C = 35 → F = 93.25 := by
  sorry

end fahrenheit_celsius_conversion_l2634_263409


namespace macaron_difference_l2634_263452

/-- The number of macarons made by each person and given to kids --/
structure MacaronProblem where
  mitch : ℕ
  joshua : ℕ
  miles : ℕ
  renz : ℕ
  kids : ℕ
  macarons_per_kid : ℕ

/-- The conditions of the macaron problem --/
def validMacaronProblem (p : MacaronProblem) : Prop :=
  p.mitch = 20 ∧
  p.joshua = p.miles / 2 ∧
  p.joshua > p.mitch ∧
  p.renz = (3 * p.miles) / 4 - 1 ∧
  p.kids = 68 ∧
  p.macarons_per_kid = 2 ∧
  p.mitch + p.joshua + p.miles + p.renz = p.kids * p.macarons_per_kid

/-- The theorem stating the difference between Joshua's and Mitch's macarons --/
theorem macaron_difference (p : MacaronProblem) (h : validMacaronProblem p) :
  p.joshua - p.mitch = 27 := by
  sorry

end macaron_difference_l2634_263452


namespace unique_factorization_l2634_263449

theorem unique_factorization (E F G H : ℕ+) : 
  E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ F ≠ G ∧ F ≠ H ∧ G ≠ H →
  E * F = 120 →
  G * H = 120 →
  E - F = G + H - 2 →
  E = 30 := by
sorry

end unique_factorization_l2634_263449


namespace two_valid_inequalities_l2634_263418

theorem two_valid_inequalities : 
  (∃ (f₁ f₂ f₃ : Prop), 
    (f₁ ↔ ∀ x : ℝ, Real.sqrt 5 + Real.sqrt 9 > 2 * Real.sqrt 7) ∧ 
    (f₂ ↔ ∀ a b c : ℝ, a^2 + 2*b^2 + 3*c^2 ≥ (1/6) * (a + 2*b + 3*c)^2) ∧ 
    (f₃ ↔ ∀ x : ℝ, Real.exp x ≥ x + 1) ∧ 
    (f₁ ∨ f₂ ∨ f₃) ∧ 
    (f₁ ∧ f₂ ∨ f₁ ∧ f₃ ∨ f₂ ∧ f₃) ∧ 
    ¬(f₁ ∧ f₂ ∧ f₃)) :=
by sorry

end two_valid_inequalities_l2634_263418


namespace complex_multiplication_simplification_l2634_263499

theorem complex_multiplication_simplification :
  let z₁ : ℂ := 5 + 3 * Complex.I
  let z₂ : ℂ := -2 - 6 * Complex.I
  let z₃ : ℂ := 1 - 2 * Complex.I
  (z₁ - z₂) * z₃ = 25 - 5 * Complex.I := by
  sorry

end complex_multiplication_simplification_l2634_263499


namespace percentage_of_number_l2634_263408

theorem percentage_of_number (percentage : ℝ) (number : ℝ) (result : ℝ) :
  percentage = 110 ∧ number = 500 ∧ result = 550 →
  (percentage / 100) * number = result := by
  sorry

end percentage_of_number_l2634_263408


namespace max_value_expression_l2634_263412

theorem max_value_expression (A M C : ℕ) (h : A + M + C = 15) :
  (∀ a m c : ℕ, a + m + c = 15 → 2*a*m*c + a*m + m*c + c*a ≤ 2*A*M*C + A*M + M*C + C*A) →
  2*A*M*C + A*M + M*C + C*A = 325 :=
by sorry

end max_value_expression_l2634_263412


namespace min_students_solved_both_l2634_263446

theorem min_students_solved_both (total : ℕ) (first : ℕ) (second : ℕ) :
  total = 30 →
  first = 21 →
  second = 18 →
  ∃ (both : ℕ), both ≥ 9 ∧
    both ≤ first ∧
    both ≤ second ∧
    (∀ (x : ℕ), x < both → x + (first - x) + (second - x) > total) :=
by sorry

end min_students_solved_both_l2634_263446


namespace multiples_of_four_l2634_263470

theorem multiples_of_four (n : ℕ) : n = 20 ↔ (
  (∃ (m : List ℕ), 
    m.length = 24 ∧ 
    (∀ x ∈ m, x % 4 = 0) ∧
    (∀ x ∈ m, n ≤ x ∧ x ≤ 112) ∧
    (∀ y, n ≤ y ∧ y ≤ 112 ∧ y % 4 = 0 → y ∈ m)
  )
) := by sorry

end multiples_of_four_l2634_263470


namespace prob_same_first_last_pancake_l2634_263488

/-- Represents the types of pancake fillings -/
inductive Filling
  | Meat
  | CottageCheese
  | Strawberry

/-- Represents a plate of pancakes -/
structure PlatePancakes where
  total : Nat
  meat : Nat
  cheese : Nat
  strawberry : Nat

/-- Calculates the probability of selecting the same filling for first and last pancake -/
def probSameFirstLast (plate : PlatePancakes) : Rat :=
  sorry

/-- Theorem stating the probability of selecting the same filling for first and last pancake -/
theorem prob_same_first_last_pancake (plate : PlatePancakes) :
  plate.total = 10 ∧ plate.meat = 2 ∧ plate.cheese = 3 ∧ plate.strawberry = 5 →
  probSameFirstLast plate = 14 / 45 := by
  sorry

end prob_same_first_last_pancake_l2634_263488


namespace bacteria_growth_without_offset_bacteria_growth_non_negative_without_offset_l2634_263402

/-- The number of bacteria after 60 minutes of doubling every minute, starting with 10 bacteria -/
def final_bacteria_count : ℕ := 10240

/-- The initial number of bacteria -/
def initial_bacteria_count : ℕ := 10

/-- The number of minutes in one hour -/
def minutes_in_hour : ℕ := 60

/-- Theorem stating that the initial number of bacteria without offset would be 0 -/
theorem bacteria_growth_without_offset :
  ∀ n : ℤ, (n + initial_bacteria_count) * 2^minutes_in_hour = final_bacteria_count → n = -initial_bacteria_count :=
by sorry

/-- Corollary stating that the non-negative initial number of bacteria without offset is 0 -/
theorem bacteria_growth_non_negative_without_offset :
  ∀ n : ℕ, (n + initial_bacteria_count) * 2^minutes_in_hour = final_bacteria_count → n = 0 :=
by sorry

end bacteria_growth_without_offset_bacteria_growth_non_negative_without_offset_l2634_263402


namespace point_transformation_l2634_263493

def initial_point : ℝ × ℝ × ℝ := (2, 2, 2)

def rotate_z_90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-y, x, z)

def reflect_xy (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, y, -z)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

def rotate_x_90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -z, y)

def transform (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  p |> rotate_z_90
    |> reflect_xy
    |> reflect_yz
    |> rotate_x_90
    |> reflect_xy

theorem point_transformation :
  transform initial_point = (2, -2, 2) := by sorry

end point_transformation_l2634_263493


namespace set_operation_result_l2634_263462

-- Define the universal set U
def U : Set Nat := {0, 1, 2, 3, 4, 5}

-- Define set A
def A : Set Nat := {1, 2, 4}

-- Define set B
def B : Set Nat := {2, 3, 5}

-- Theorem statement
theorem set_operation_result :
  (U \ A) ∪ B = {0, 2, 3, 5} := by sorry

end set_operation_result_l2634_263462


namespace root_sum_squares_equality_l2634_263447

theorem root_sum_squares_equality (a b : ℝ) : 
  (∃ x y : ℝ, x^2 + a*x + b = 0 ∧ y^2 + b*y + a = 0) →  -- both equations have real roots
  (∃ p q r s : ℝ, p^2 + q^2 = r^2 + s^2 ∧               -- sum of squares of roots are equal
                  p^2 + a*p + b = 0 ∧ q^2 + a*q + b = 0 ∧ 
                  r^2 + b*r + a = 0 ∧ s^2 + b*s + a = 0) →
  a ≠ b →                                               -- a is not equal to b
  a + b = -2 :=                                         -- conclusion
by sorry

end root_sum_squares_equality_l2634_263447


namespace sum_factorials_6_mod_20_l2634_263419

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_factorials (n : ℕ) : ℕ :=
  match n with
  | 0 => factorial 0
  | n + 1 => factorial (n + 1) + sum_factorials n

theorem sum_factorials_6_mod_20 :
  sum_factorials 6 % 20 = 13 := by sorry

end sum_factorials_6_mod_20_l2634_263419


namespace coexistent_pair_properties_l2634_263424

/-- Definition of coexistent rational number pairs -/
def is_coexistent_pair (a b : ℚ) : Prop := a - b = a * b + 1

theorem coexistent_pair_properties :
  /- (1) -/
  is_coexistent_pair 3 (1/2) ∧
  /- (2) -/
  (∀ m n : ℚ, is_coexistent_pair m n → is_coexistent_pair (-n) (-m)) ∧
  /- (3) -/
  is_coexistent_pair 4 (3/5) ∧
  /- (4) -/
  (∀ a : ℚ, is_coexistent_pair a 3 → a = -2) :=
by sorry

end coexistent_pair_properties_l2634_263424


namespace a_squared_minus_b_squared_l2634_263416

theorem a_squared_minus_b_squared (a b : ℚ) 
  (h1 : a + b = 2/3) 
  (h2 : a - b = 1/6) : 
  a^2 - b^2 = 1/9 := by
  sorry

end a_squared_minus_b_squared_l2634_263416


namespace distance_when_in_step_l2634_263432

/-- The stride length of Jack in centimeters. -/
def jackStride : ℕ := 64

/-- The stride length of Jill in centimeters. -/
def jillStride : ℕ := 56

/-- The theorem states that the distance walked when Jack and Jill are next in step
    is equal to the least common multiple of their stride lengths. -/
theorem distance_when_in_step :
  Nat.lcm jackStride jillStride = 448 := by sorry

end distance_when_in_step_l2634_263432


namespace sum_in_special_base_l2634_263400

theorem sum_in_special_base (b : ℕ) (h : b > 1) :
  (b + 3) * (b + 4) * (b + 5) = 2 * b^3 + 3 * b^2 + 2 * b + 5 →
  (b + 3) + (b + 4) + (b + 5) = 4 * b + 2 :=
by sorry

end sum_in_special_base_l2634_263400


namespace pythagorean_triple_7_24_25_l2634_263431

theorem pythagorean_triple_7_24_25 : 
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a = 7 ∧ b = 24 ∧ c = 25 ∧ a^2 + b^2 = c^2 := by
  sorry

end pythagorean_triple_7_24_25_l2634_263431


namespace john_needs_168_nails_l2634_263466

/-- The number of nails needed for a house wall -/
def nails_needed (num_planks : ℕ) (nails_per_plank : ℕ) : ℕ :=
  num_planks * nails_per_plank

/-- Theorem: John needs 168 nails for the house wall -/
theorem john_needs_168_nails :
  nails_needed 42 4 = 168 := by
  sorry

end john_needs_168_nails_l2634_263466


namespace count_valid_triples_l2634_263483

def validTriple (x y z : ℕ+) : Prop :=
  Nat.lcm x.val y.val = 120 ∧ 
  Nat.lcm x.val z.val = 450 ∧ 
  Nat.lcm y.val z.val = 180

theorem count_valid_triples : 
  ∃! (s : Finset (ℕ+ × ℕ+ × ℕ+)), 
    (∀ t ∈ s, validTriple t.1 t.2.1 t.2.2) ∧ 
    s.card = 6 :=
sorry

end count_valid_triples_l2634_263483


namespace journey_distance_l2634_263471

theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  total_time = 15 →
  speed1 = 21 →
  speed2 = 24 →
  ∃ (distance : ℝ),
    distance / 2 / speed1 + distance / 2 / speed2 = total_time ∧
    distance = 336 := by
  sorry

end journey_distance_l2634_263471


namespace smallest_perimeter_l2634_263485

/-- A triangle with side lengths that are three consecutive integers starting from 3 -/
structure ConsecutiveIntegerTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  consecutive : b = a + 1 ∧ c = b + 1
  start_from_three : a = 3

/-- The perimeter of a triangle -/
def perimeter (t : ConsecutiveIntegerTriangle) : ℕ :=
  t.a + t.b + t.c

/-- Theorem: The perimeter of a triangle with side lengths 3, 4, and 5 is 12 units -/
theorem smallest_perimeter (t : ConsecutiveIntegerTriangle) : perimeter t = 12 := by
  sorry

end smallest_perimeter_l2634_263485


namespace triangle_side_c_l2634_263476

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_side_c (t : Triangle) 
  (h1 : t.a = 5) 
  (h2 : t.b = 7) 
  (h3 : t.B = 60 * π / 180) : 
  t.c = 8 := by
  sorry


end triangle_side_c_l2634_263476


namespace largest_n_for_quadratic_equation_l2634_263469

theorem largest_n_for_quadratic_equation : ∃ (x y z : ℕ+),
  13^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 6*x + 6*y + 6*z - 12 ∧
  ∀ (n : ℕ+), n > 13 →
    ¬∃ (a b c : ℕ+), n^2 = a^2 + b^2 + c^2 + 2*a*b + 2*b*c + 2*c*a + 6*a + 6*b + 6*c - 12 :=
by sorry

end largest_n_for_quadratic_equation_l2634_263469
