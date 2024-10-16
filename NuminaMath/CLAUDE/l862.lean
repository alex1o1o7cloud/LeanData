import Mathlib

namespace NUMINAMATH_CALUDE_integer_root_values_l862_86259

theorem integer_root_values (a : ℤ) : 
  (∃ x : ℤ, x^3 + 3*x^2 + a*x + 7 = 0) ↔ a ∈ ({-71, -27, -11, 9} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_integer_root_values_l862_86259


namespace NUMINAMATH_CALUDE_kids_on_other_days_l862_86276

/-- 
Given that Julia played tag with some kids from Monday to Friday,
prove that the number of kids she played with on Monday, Thursday, and Friday combined
is equal to the total number of kids minus the number of kids on Tuesday and Wednesday.
-/
theorem kids_on_other_days 
  (total_kids : ℕ) 
  (tuesday_wednesday_kids : ℕ) 
  (h1 : total_kids = 75) 
  (h2 : tuesday_wednesday_kids = 36) : 
  total_kids - tuesday_wednesday_kids = 39 := by
sorry

end NUMINAMATH_CALUDE_kids_on_other_days_l862_86276


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_over_one_plus_i_l862_86289

theorem imaginary_part_of_i_over_one_plus_i : Complex.im (Complex.I / (1 + Complex.I)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_over_one_plus_i_l862_86289


namespace NUMINAMATH_CALUDE_base_b_not_perfect_square_l862_86251

theorem base_b_not_perfect_square (b : ℕ) (h : b ≥ 3) :
  ¬∃ (n : ℕ), 2 * b^2 + 2 * b + 1 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_base_b_not_perfect_square_l862_86251


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l862_86214

/-- Given a circle C that is symmetric to the circle (x+2)^2+(y-1)^2=1 with respect to the origin,
    prove that the equation of circle C is (x-2)^2+(y+1)^2=1 -/
theorem symmetric_circle_equation (C : Set (ℝ × ℝ)) : 
  (∀ (x y : ℝ), (x, y) ∈ C ↔ (-x, -y) ∈ {(x, y) | (x + 2)^2 + (y - 1)^2 = 1}) →
  C = {(x, y) | (x - 2)^2 + (y + 1)^2 = 1} :=
by sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l862_86214


namespace NUMINAMATH_CALUDE_range_of_a_min_value_of_g_l862_86229

-- Define the quadratic function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + a

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := a * f a x - a^2 * (x + 1) - 2*x

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 ∧
   f a x₁ = x₁ ∧ f a x₂ = x₂) →
  0 < a ∧ a < 3 - 2 * Real.sqrt 2 :=
sorry

-- Theorem for the minimum value of g
theorem min_value_of_g (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, 
    (a < 1 → g a x ≥ a - 2) ∧
    (a ≥ 1 → g a x ≥ -1/a)) ∧
  (∃ x ∈ Set.Icc 0 1, 
    (a < 1 → g a x = a - 2) ∧
    (a ≥ 1 → g a x = -1/a)) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_min_value_of_g_l862_86229


namespace NUMINAMATH_CALUDE_ratio_sum_problem_l862_86230

theorem ratio_sum_problem (a b c : ℝ) : 
  (a / b = 5 / 3) ∧ (c / b = 4 / 3) ∧ (b = 27) → a + b + c = 108 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_problem_l862_86230


namespace NUMINAMATH_CALUDE_sqrt_81_division_l862_86233

theorem sqrt_81_division :
  ∃ x : ℝ, x > 0 ∧ (Real.sqrt 81) / x = 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_81_division_l862_86233


namespace NUMINAMATH_CALUDE_greg_harvest_l862_86242

theorem greg_harvest (sharon_harvest : Real) (greg_additional : Real) : 
  sharon_harvest = 0.1 →
  greg_additional = 0.3 →
  sharon_harvest + greg_additional = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_greg_harvest_l862_86242


namespace NUMINAMATH_CALUDE_fourth_sample_id_l862_86231

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  total_students : ℕ
  sample_size : ℕ
  known_ids : List ℕ

/-- Calculates the sampling interval -/
def sampling_interval (s : SystematicSampling) : ℕ :=
  s.total_students / s.sample_size

/-- Checks if a given ID is part of the sample -/
def is_in_sample (s : SystematicSampling) (id : ℕ) : Prop :=
  ∃ k : ℕ, id = s.known_ids.head! + k * sampling_interval s

/-- The main theorem to prove -/
theorem fourth_sample_id (s : SystematicSampling)
  (h1 : s.total_students = 44)
  (h2 : s.sample_size = 4)
  (h3 : s.known_ids = [6, 28, 39]) :
  is_in_sample s 17 := by
  sorry

#check fourth_sample_id

end NUMINAMATH_CALUDE_fourth_sample_id_l862_86231


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l862_86252

theorem absolute_value_inequality (x : ℝ) : 
  (abs x + abs (abs x - 1) = 1) → (x + 1) * (x - 1) ≤ 0 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l862_86252


namespace NUMINAMATH_CALUDE_swap_meet_backpack_price_l862_86226

/-- Proves that the price of each backpack sold at the swap meet was $18 --/
theorem swap_meet_backpack_price :
  ∀ (swap_meet_price : ℕ),
    (48 : ℕ) = 17 + 10 + (48 - 17 - 10) →
    (576 : ℕ) = 48 * 12 →
    (442 : ℕ) = (17 * swap_meet_price + 10 * 25 + (48 - 17 - 10) * 22) - 576 →
    swap_meet_price = 18 := by
  sorry


end NUMINAMATH_CALUDE_swap_meet_backpack_price_l862_86226


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l862_86257

theorem partial_fraction_decomposition (C D : ℚ) :
  (∀ x : ℚ, x ≠ 3 ∧ x ≠ 5 →
    (6 * x - 3) / (x^2 - 8*x + 15) = C / (x - 3) + D / (x - 5)) →
  C = -15/2 ∧ D = 27/2 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l862_86257


namespace NUMINAMATH_CALUDE_first_half_rate_is_5_4_l862_86217

/-- Represents a cricket game with two halves --/
structure CricketGame where
  total_overs : ℕ
  target_runs : ℕ
  second_half_rate : ℚ

/-- Calculates the run rate for the first half of the game --/
def first_half_run_rate (game : CricketGame) : ℚ :=
  let first_half_overs : ℚ := game.total_overs / 2
  let second_half_runs : ℚ := game.second_half_rate * first_half_overs
  let first_half_runs : ℚ := game.target_runs - second_half_runs
  first_half_runs / first_half_overs

/-- Theorem stating the first half run rate for the given game conditions --/
theorem first_half_rate_is_5_4 (game : CricketGame) 
    (h1 : game.total_overs = 50)
    (h2 : game.target_runs = 400)
    (h3 : game.second_half_rate = 53 / 5) : 
  first_half_run_rate game = 27 / 5 := by
  sorry

#eval (53 : ℚ) / 5  -- Outputs 10.6
#eval (27 : ℚ) / 5  -- Outputs 5.4

end NUMINAMATH_CALUDE_first_half_rate_is_5_4_l862_86217


namespace NUMINAMATH_CALUDE_box_width_proof_l862_86268

/-- Given a rectangular box with length 12 cm, height 6 cm, and volume 1152 cubic cm,
    prove that the width of the box is 16 cm. -/
theorem box_width_proof (length : ℝ) (height : ℝ) (volume : ℝ) (width : ℝ) 
    (h1 : length = 12)
    (h2 : height = 6)
    (h3 : volume = 1152)
    (h4 : volume = length * width * height) :
  width = 16 := by
  sorry

end NUMINAMATH_CALUDE_box_width_proof_l862_86268


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l862_86238

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℚ) := ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

-- State the theorem
theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : a 7 + a 9 = 16) 
  (h_fourth : a 4 = 1) : 
  a 12 = 15 := by
sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l862_86238


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l862_86212

/-- Sum of arithmetic sequence -/
def arithmetic_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n / 2 * (2 * a₁ + (n - 1) * d)

/-- The sum of the first 3k terms of an arithmetic sequence with first term k^2 + k and common difference 1 -/
theorem arithmetic_sequence_sum (k : ℕ) :
  arithmetic_sum (k^2 + k) 1 (3 * k) = 3 * k^3 + (15 / 2) * k^2 - (3 / 2) * k :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l862_86212


namespace NUMINAMATH_CALUDE_projection_property_l862_86267

/-- A projection that takes [4, 4] to [60/13, 12/13] -/
def projection (v : ℝ × ℝ) : ℝ × ℝ :=
  sorry

theorem projection_property : 
  projection (4, 4) = (60/13, 12/13) ∧ 
  projection (-2, 2) = (-20/13, -4/13) := by
  sorry

end NUMINAMATH_CALUDE_projection_property_l862_86267


namespace NUMINAMATH_CALUDE_jamie_quiz_score_l862_86299

def school_quiz (total_questions correct_answers incorrect_answers unanswered_questions : ℕ)
  (points_correct points_incorrect points_unanswered : ℚ) : Prop :=
  total_questions = correct_answers + incorrect_answers + unanswered_questions ∧
  (correct_answers : ℚ) * points_correct +
  (incorrect_answers : ℚ) * points_incorrect +
  (unanswered_questions : ℚ) * points_unanswered = 28

theorem jamie_quiz_score :
  school_quiz 30 16 10 4 2 (-1/2) (1/4) :=
by sorry

end NUMINAMATH_CALUDE_jamie_quiz_score_l862_86299


namespace NUMINAMATH_CALUDE_john_earnings_proof_l862_86253

def hours_per_workday : ℕ := 12
def days_in_month : ℕ := 30
def former_hourly_wage : ℚ := 20
def raise_percentage : ℚ := 30 / 100

def john_monthly_earnings : ℚ :=
  (days_in_month / 2) * hours_per_workday * (former_hourly_wage * (1 + raise_percentage))

theorem john_earnings_proof :
  john_monthly_earnings = 4680 := by
  sorry

end NUMINAMATH_CALUDE_john_earnings_proof_l862_86253


namespace NUMINAMATH_CALUDE_brianna_marbles_l862_86245

/-- Calculates the number of marbles Brianna has remaining after a series of events. -/
def remaining_marbles (initial : ℕ) (lost : ℕ) : ℕ :=
  initial - lost - 2 * lost - lost / 2

/-- Theorem stating that Brianna has 10 marbles remaining given the initial conditions. -/
theorem brianna_marbles : remaining_marbles 24 4 = 10 := by
  sorry

#eval remaining_marbles 24 4

end NUMINAMATH_CALUDE_brianna_marbles_l862_86245


namespace NUMINAMATH_CALUDE_correct_conic_propositions_l862_86219

/-- Represents a proposition about conic sections -/
inductive ConicProposition
| Prop1
| Prop2
| Prop3
| Prop4
| Prop5

/-- Determines if a given proposition is correct -/
def is_correct (prop : ConicProposition) : Bool :=
  match prop with
  | ConicProposition.Prop1 => true
  | ConicProposition.Prop2 => false
  | ConicProposition.Prop3 => false
  | ConicProposition.Prop4 => true
  | ConicProposition.Prop5 => false

/-- The theorem to be proved -/
theorem correct_conic_propositions :
  (List.filter is_correct [ConicProposition.Prop1, ConicProposition.Prop2, 
                           ConicProposition.Prop3, ConicProposition.Prop4, 
                           ConicProposition.Prop5]).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_correct_conic_propositions_l862_86219


namespace NUMINAMATH_CALUDE_store_profit_calculation_l862_86223

/-- Represents the pricing strategy and profit calculation for a store selling turtleneck sweaters -/
theorem store_profit_calculation (C : ℝ) (h : C > 0) :
  let initial_markup := 1.20
  let new_year_markup := 1.25
  let february_discount := 0.80
  let final_price := C * initial_markup * new_year_markup * february_discount
  final_price = 1.20 * C ∧ (final_price - C) / C = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_store_profit_calculation_l862_86223


namespace NUMINAMATH_CALUDE_lizzies_group_size_l862_86220

theorem lizzies_group_size (total : ℕ) (difference : ℕ) : 
  total = 91 → difference = 17 → ∃ (other : ℕ), other + (other + difference) = total ∧ other + difference = 54 :=
by
  sorry

end NUMINAMATH_CALUDE_lizzies_group_size_l862_86220


namespace NUMINAMATH_CALUDE_solution_set_x_squared_geq_four_l862_86278

theorem solution_set_x_squared_geq_four :
  {x : ℝ | x^2 ≥ 4} = {x : ℝ | x ≤ -2 ∨ x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_x_squared_geq_four_l862_86278


namespace NUMINAMATH_CALUDE_julia_pet_food_cost_l862_86200

/-- The total amount Julia spent on food for her animals -/
def total_spent (weekly_total : ℕ) (rabbit_weeks : ℕ) (parrot_weeks : ℕ) (rabbit_cost : ℕ) : ℕ :=
  let parrot_cost := weekly_total - rabbit_cost
  rabbit_weeks * rabbit_cost + parrot_weeks * parrot_cost

/-- Theorem stating the total amount Julia spent on food for her animals -/
theorem julia_pet_food_cost :
  total_spent 30 5 3 12 = 114 := by
  sorry

end NUMINAMATH_CALUDE_julia_pet_food_cost_l862_86200


namespace NUMINAMATH_CALUDE_highway_speed_is_30_l862_86218

-- Define the problem parameters
def initial_reading : ℕ := 12321
def next_palindrome : ℕ := 12421
def total_time : ℕ := 4
def highway_time : ℕ := 2
def urban_time : ℕ := 2
def speed_difference : ℕ := 10
def total_distance : ℕ := 100

-- Define the theorem
theorem highway_speed_is_30 :
  let urban_speed := (total_distance - speed_difference * highway_time) / total_time
  urban_speed + speed_difference = 30 := by
  sorry


end NUMINAMATH_CALUDE_highway_speed_is_30_l862_86218


namespace NUMINAMATH_CALUDE_centroid_trajectory_on_hyperbola_l862_86254

/-- The trajectory of the centroid of a triangle formed by a point on a hyperbola and its foci -/
theorem centroid_trajectory_on_hyperbola (x y m n : ℝ) :
  let f₁ : ℝ × ℝ := (5, 0)
  let f₂ : ℝ × ℝ := (-5, 0)
  let p : ℝ × ℝ := (m, n)
  let g : ℝ × ℝ := (x, y)
  (m^2 / 16 - n^2 / 9 = 1) →  -- P is on the hyperbola
  (x = (m + 5 + (-5)) / 3 ∧ y = n / 3) →  -- G is the centroid of ΔF₁F₂P
  (y ≠ 0) →
  (x^2 / (16/9) - y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_centroid_trajectory_on_hyperbola_l862_86254


namespace NUMINAMATH_CALUDE_binomial_coefficient_identity_l862_86228

theorem binomial_coefficient_identity (r m k : ℕ) (h1 : k ≤ m) (h2 : m ≤ r) :
  (Nat.choose r m) * (Nat.choose m k) = (Nat.choose r k) * (Nat.choose (r - k) (m - k)) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_identity_l862_86228


namespace NUMINAMATH_CALUDE_max_d_is_four_l862_86260

/-- A function that constructs a 6-digit number of the form 6d6,33e -/
def construct_number (d e : ℕ) : ℕ := 
  600000 + d * 10000 + 6 * 1000 + 300 + 30 + e

/-- Proposition: The maximum value of d is 4 -/
theorem max_d_is_four :
  ∃ (d e : ℕ),
    d ≤ 9 ∧ e ≤ 9 ∧
    (construct_number d e) % 33 = 0 ∧
    d + e = 4 ∧
    ∀ (d' e' : ℕ), d' ≤ 9 ∧ e' ≤ 9 ∧ 
      (construct_number d' e') % 33 = 0 ∧ 
      d' + e' = 4 → 
      d' ≤ d :=
by
  sorry

end NUMINAMATH_CALUDE_max_d_is_four_l862_86260


namespace NUMINAMATH_CALUDE_f_one_is_zero_five_zeros_symmetric_center_l862_86249

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_property : ∀ x, f (x - 1) = f (x + 1)
axiom decreasing_property : ∀ x₁ x₂, x₁ ∈ Set.Ioo 0 1 → x₂ ∈ Set.Ioo 0 1 → x₁ ≠ x₂ → (f x₂ - f x₁) / (x₂ - x₁) < 0

-- Theorem statements
theorem f_one_is_zero : f 1 = 0 := by sorry

theorem five_zeros : 
  f (-2) = 0 ∧ f (-1) = 0 ∧ f 0 = 0 ∧ f 1 = 0 ∧ f 2 = 0 := by sorry

theorem symmetric_center : 
  ∀ x, f (2014 + x) = -f (2014 - x) := by sorry

end NUMINAMATH_CALUDE_f_one_is_zero_five_zeros_symmetric_center_l862_86249


namespace NUMINAMATH_CALUDE_fourth_group_frequency_l862_86209

/-- Given a set of data with 50 items divided into 5 groups, prove that the frequency of the fourth group is 12 -/
theorem fourth_group_frequency
  (total_items : ℕ)
  (num_groups : ℕ)
  (freq_group1 : ℕ)
  (freq_group2 : ℕ)
  (freq_group3 : ℕ)
  (freq_group5 : ℕ)
  (h_total : total_items = 50)
  (h_groups : num_groups = 5)
  (h_freq1 : freq_group1 = 10)
  (h_freq2 : freq_group2 = 8)
  (h_freq3 : freq_group3 = 11)
  (h_freq5 : freq_group5 = 9) :
  total_items - (freq_group1 + freq_group2 + freq_group3 + freq_group5) = 12 :=
by sorry

end NUMINAMATH_CALUDE_fourth_group_frequency_l862_86209


namespace NUMINAMATH_CALUDE_arithmetic_equality_l862_86262

theorem arithmetic_equality : 239 - 27 + 45 + 33 - 11 = 279 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l862_86262


namespace NUMINAMATH_CALUDE_polynomial_sum_equality_l862_86281

/-- Given polynomials p, q, and r, prove their sum is equal to the specified polynomial -/
theorem polynomial_sum_equality (x : ℝ) : 
  let p := fun x : ℝ => -4*x^2 + 2*x - 5
  let q := fun x : ℝ => -6*x^2 + 4*x - 9
  let r := fun x : ℝ => 6*x^2 + 6*x + 2
  p x + q x + r x = -4*x^2 + 12*x - 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_equality_l862_86281


namespace NUMINAMATH_CALUDE_cube_root_64_square_root_4_power_l862_86265

theorem cube_root_64_square_root_4_power (x y : ℝ) : 
  x^3 = 64 → y^2 = 4 → x^y = 16 := by
sorry

end NUMINAMATH_CALUDE_cube_root_64_square_root_4_power_l862_86265


namespace NUMINAMATH_CALUDE_hcf_problem_l862_86210

theorem hcf_problem (a b : ℕ) (h1 : a = 280) (h2 : Nat.lcm a b = Nat.gcd a b * 13 * 14) :
  Nat.gcd a b = 5 := by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l862_86210


namespace NUMINAMATH_CALUDE_complex_arithmetic_l862_86222

theorem complex_arithmetic (A M S : ℂ) (P : ℝ) 
  (hA : A = 5 - 2*I) 
  (hM : M = -3 + 2*I) 
  (hS : S = 2*I) 
  (hP : P = 3) : 
  A - M + S - (P : ℂ) = 5 - 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_l862_86222


namespace NUMINAMATH_CALUDE_probability_through_x_l862_86215

structure DirectedGraph where
  vertices : Finset Char
  edges : Finset (Char × Char)

def paths (g : DirectedGraph) (start finish : Char) : Nat :=
  sorry

theorem probability_through_x (g : DirectedGraph) :
  g.vertices = {'A', 'B', 'X', 'Y'} →
  paths g 'A' 'X' = 2 →
  paths g 'X' 'B' = 1 →
  paths g 'X' 'Y' = 1 →
  paths g 'Y' 'B' = 3 →
  paths g 'A' 'Y' = 3 →
  (paths g 'A' 'X' * paths g 'X' 'B' + paths g 'A' 'X' * paths g 'X' 'Y' * paths g 'Y' 'B') / 
  (paths g 'A' 'X' * paths g 'X' 'B' + paths g 'A' 'X' * paths g 'X' 'Y' * paths g 'Y' 'B' + paths g 'A' 'Y' * paths g 'Y' 'B') = 8 / 11 :=
sorry

end NUMINAMATH_CALUDE_probability_through_x_l862_86215


namespace NUMINAMATH_CALUDE_camp_cedar_boys_l862_86232

theorem camp_cedar_boys (boys : ℕ) (girls : ℕ) (counselors : ℕ) : 
  girls = 3 * boys →
  counselors = 20 →
  boys + girls = 8 * counselors →
  boys = 40 := by
sorry

end NUMINAMATH_CALUDE_camp_cedar_boys_l862_86232


namespace NUMINAMATH_CALUDE_obtuse_triangle_properties_l862_86294

/-- Properties of an obtuse triangle ABC -/
structure ObtuseTriangleABC where
  -- Side lengths
  a : ℝ
  b : ℝ
  -- Angle A in radians
  A : ℝ
  -- Triangle ABC is obtuse
  is_obtuse : Bool
  -- Given conditions
  ha : a = 7
  hb : b = 8
  hA : A = π / 3
  h_obtuse : is_obtuse = true

/-- Main theorem about the obtuse triangle ABC -/
theorem obtuse_triangle_properties (t : ObtuseTriangleABC) :
  -- 1. sin B = (4√3) / 7
  Real.sin (Real.arcsin ((t.b * Real.sin t.A) / t.a)) = (4 * Real.sqrt 3) / 7 ∧
  -- 2. Height on side BC = (12√3) / 7
  ∃ (h : ℝ), h = (12 * Real.sqrt 3) / 7 ∧ h = t.b * Real.sin (π - t.A - Real.arcsin ((t.b * Real.sin t.A) / t.a)) :=
by sorry

end NUMINAMATH_CALUDE_obtuse_triangle_properties_l862_86294


namespace NUMINAMATH_CALUDE_train_crossing_time_l862_86266

/-- A train crosses a platform and an electric pole -/
theorem train_crossing_time (train_speed : ℝ) (platform_length : ℝ) (platform_crossing_time : ℝ) :
  train_speed = 10 →
  platform_length = 320 →
  platform_crossing_time = 44 →
  (platform_length + train_speed * platform_crossing_time) / train_speed = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l862_86266


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_theorem_l862_86239

/-- Hyperbola eccentricity theorem -/
theorem hyperbola_eccentricity_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  (b / a ≥ Real.sqrt 3) → e ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_theorem_l862_86239


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l862_86275

theorem sqrt_expression_equality : Real.sqrt 3 * Real.sqrt 2 - Real.sqrt 2 + Real.sqrt 8 = Real.sqrt 6 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l862_86275


namespace NUMINAMATH_CALUDE_fibonacci_divisibility_l862_86263

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fibonacci_divisibility (m n : ℕ) (h1 : m ≥ 1) (h2 : n > 1) :
  ∃ k : ℕ, fib (m * n - 1) - (fib (n - 1))^m = k * (fib n)^2 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_divisibility_l862_86263


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l862_86216

theorem purely_imaginary_complex_number (m : ℝ) : 
  (((m^2 - m) : ℂ) + m * I).re = 0 → m = 1 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l862_86216


namespace NUMINAMATH_CALUDE_pink_roses_count_l862_86255

theorem pink_roses_count (total_rows : ℕ) (roses_per_row : ℕ) 
  (red_fraction : ℚ) (white_fraction : ℚ) :
  total_rows = 10 →
  roses_per_row = 20 →
  red_fraction = 1/2 →
  white_fraction = 3/5 →
  (total_rows * roses_per_row * (1 - red_fraction) * (1 - white_fraction) : ℚ) = 40 :=
by sorry

end NUMINAMATH_CALUDE_pink_roses_count_l862_86255


namespace NUMINAMATH_CALUDE_valid_word_count_l862_86236

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 3

/-- The length of the words -/
def word_length : ℕ := 20

/-- Function to calculate the number of valid words -/
def count_valid_words (n : ℕ) : ℕ :=
  alphabet_size * 2^(n - 1)

/-- Theorem stating the number of valid 20-letter words -/
theorem valid_word_count :
  count_valid_words word_length = 786432 :=
sorry

end NUMINAMATH_CALUDE_valid_word_count_l862_86236


namespace NUMINAMATH_CALUDE_knight_return_even_moves_l862_86274

/-- Represents a chess square --/
structure ChessSquare :=
  (color : Bool)

/-- Represents a knight's move on a chess board --/
def knightMove (start : ChessSquare) : ChessSquare :=
  { color := ¬start.color }

/-- Represents a sequence of knight moves --/
def knightMoves (start : ChessSquare) (n : ℕ) : ChessSquare :=
  match n with
  | 0 => start
  | m + 1 => knightMove (knightMoves start m)

/-- Theorem: If a knight returns to its starting square after n moves, then n is even --/
theorem knight_return_even_moves (start : ChessSquare) (n : ℕ) :
  knightMoves start n = start → Even n :=
by sorry

end NUMINAMATH_CALUDE_knight_return_even_moves_l862_86274


namespace NUMINAMATH_CALUDE_diana_biking_time_l862_86298

def total_distance : ℝ := 10
def initial_speed : ℝ := 3
def initial_duration : ℝ := 2
def tired_speed : ℝ := 1

theorem diana_biking_time : 
  let initial_distance := initial_speed * initial_duration
  let remaining_distance := total_distance - initial_distance
  let tired_duration := remaining_distance / tired_speed
  initial_duration + tired_duration = 6 := by sorry

end NUMINAMATH_CALUDE_diana_biking_time_l862_86298


namespace NUMINAMATH_CALUDE_fathers_age_when_sum_is_100_l862_86247

/-- Given a mother aged 42 and a father aged 44, prove that the father will be 51 years old when the sum of their ages is 100. -/
theorem fathers_age_when_sum_is_100 (mother_age father_age : ℕ) 
  (h1 : mother_age = 42) 
  (h2 : father_age = 44) : 
  ∃ (years : ℕ), mother_age + years + (father_age + years) = 100 ∧ father_age + years = 51 := by
  sorry

end NUMINAMATH_CALUDE_fathers_age_when_sum_is_100_l862_86247


namespace NUMINAMATH_CALUDE_m_range_l862_86234

def f (x : ℝ) : ℝ := -x^3 - 2*x^2 + 4*x

theorem m_range (m : ℝ) : 
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≥ m^2 - 14*m) → 
  m ∈ Set.Icc 3 11 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l862_86234


namespace NUMINAMATH_CALUDE_mary_warmth_hours_l862_86250

/-- Represents the number of sticks of wood produced by different furniture types -/
structure FurnitureWood where
  chair : Nat
  table : Nat
  cabinet : Nat
  stool : Nat

/-- Represents the quantity of each furniture type Mary chops -/
structure ChoppedFurniture where
  chairs : Nat
  tables : Nat
  cabinets : Nat
  stools : Nat

/-- Calculates the total number of sticks of wood produced -/
def totalWood (fw : FurnitureWood) (cf : ChoppedFurniture) : Nat :=
  fw.chair * cf.chairs + fw.table * cf.tables + fw.cabinet * cf.cabinets + fw.stool * cf.stools

/-- Theorem stating that Mary can keep warm for 64 hours with the chopped firewood -/
theorem mary_warmth_hours (fw : FurnitureWood) (cf : ChoppedFurniture) (sticksPerHour : Nat) :
  fw.chair = 8 →
  fw.table = 12 →
  fw.cabinet = 16 →
  fw.stool = 3 →
  cf.chairs = 25 →
  cf.tables = 12 →
  cf.cabinets = 5 →
  cf.stools = 8 →
  sticksPerHour = 7 →
  totalWood fw cf / sticksPerHour = 64 := by
  sorry

#check mary_warmth_hours

end NUMINAMATH_CALUDE_mary_warmth_hours_l862_86250


namespace NUMINAMATH_CALUDE_problem_solution_l862_86256

theorem problem_solution (x : ℝ) (h : x + 1/x = 3) :
  (x - 3)^4 + 81 / (x - 3)^4 = 63 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l862_86256


namespace NUMINAMATH_CALUDE_water_usage_for_car_cleaning_l862_86205

/-- Represents the problem of calculating water usage for car cleaning --/
theorem water_usage_for_car_cleaning
  (total_water : ℝ)
  (plant_water_difference : ℝ)
  (plate_clothes_water : ℝ)
  (h1 : total_water = 65)
  (h2 : plant_water_difference = 11)
  (h3 : plate_clothes_water = 24)
  (h4 : plate_clothes_water * 2 = total_water - (2 * car_water + (2 * car_water - plant_water_difference))) :
  ∃ (car_water : ℝ), car_water = 7 :=
by sorry

end NUMINAMATH_CALUDE_water_usage_for_car_cleaning_l862_86205


namespace NUMINAMATH_CALUDE_quadratic_sum_zero_l862_86237

/-- A quadratic function f(x) = ax^2 + bx + c with specified properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_sum_zero
  (a b c : ℝ)
  (h_min : ∃ x, ∀ y, QuadraticFunction a b c y ≥ QuadraticFunction a b c x ∧ QuadraticFunction a b c x = 36)
  (h_root1 : QuadraticFunction a b c 1 = 0)
  (h_root5 : QuadraticFunction a b c 5 = 0) :
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_zero_l862_86237


namespace NUMINAMATH_CALUDE_shelter_animals_count_l862_86208

/-- Calculates the total number of animals in a shelter given the initial conditions --/
def totalAnimals (initialCats : ℕ) : ℕ :=
  let adoptedCats := initialCats / 3
  let remainingCats := initialCats - adoptedCats
  let newCats := adoptedCats * 2
  let totalCats := remainingCats + newCats
  let dogs := totalCats * 2
  totalCats + dogs

/-- Theorem stating that given the initial conditions, the total number of animals is 60 --/
theorem shelter_animals_count : totalAnimals 15 = 60 := by
  sorry

end NUMINAMATH_CALUDE_shelter_animals_count_l862_86208


namespace NUMINAMATH_CALUDE_infinitely_many_consecutive_squares_sum_square_infinitely_many_solutions_for_non_square_l862_86264

-- Part a
def ConsecutiveSquaresSum (n : ℕ+) (x : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (fun k => (x + k) ^ 2)

def IsConsecutiveSquaresSumSquare (n : ℕ+) : Prop :=
  ∃ x k : ℕ, ConsecutiveSquaresSum n x = k ^ 2

theorem infinitely_many_consecutive_squares_sum_square :
  Set.Infinite {n : ℕ+ | IsConsecutiveSquaresSumSquare n} :=
sorry

-- Part b
theorem infinitely_many_solutions_for_non_square (n : ℕ+) (h : ¬ ∃ m : ℕ, n = m ^ 2) :
  (∃ x k : ℕ, ConsecutiveSquaresSum n x = k ^ 2) →
  Set.Infinite {y : ℕ | ∃ k : ℕ, ConsecutiveSquaresSum n y = k ^ 2} :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_consecutive_squares_sum_square_infinitely_many_solutions_for_non_square_l862_86264


namespace NUMINAMATH_CALUDE_complement_union_theorem_l862_86225

def U : Finset Nat := {0, 1, 2, 3, 4, 5}
def A : Finset Nat := {1, 2, 3, 5}
def B : Finset Nat := {2, 4}

theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l862_86225


namespace NUMINAMATH_CALUDE_seven_power_plus_one_prime_factors_l862_86291

theorem seven_power_plus_one_prime_factors (n : ℕ) :
  ∃ (primes : Finset ℕ), 
    (∀ p ∈ primes, Nat.Prime p) ∧ 
    primes.card = 2 * n + 3 ∧
    (primes.prod id = 7^(7^(7^(7^2))) + 1) := by
  sorry

end NUMINAMATH_CALUDE_seven_power_plus_one_prime_factors_l862_86291


namespace NUMINAMATH_CALUDE_system_solution_l862_86202

-- Define the system of equations
def system (x y : ℝ) : Prop :=
  x + y = 4 ∧ 2 * x - y = 2

-- State the theorem
theorem system_solution :
  ∃! (x y : ℝ), system x y ∧ x = 2 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l862_86202


namespace NUMINAMATH_CALUDE_prize_orders_count_l862_86269

/-- Represents the number of bowlers in the tournament -/
def num_bowlers : ℕ := 6

/-- Represents the number of games played in the tournament -/
def num_games : ℕ := 5

/-- Calculates the number of possible outcomes for a single game -/
def outcomes_per_game : ℕ := 2

/-- Calculates the total number of possible prize orders -/
def total_prize_orders : ℕ := outcomes_per_game ^ num_games

/-- Theorem stating that the total number of possible prize orders is 32 -/
theorem prize_orders_count : total_prize_orders = 32 := by
  sorry


end NUMINAMATH_CALUDE_prize_orders_count_l862_86269


namespace NUMINAMATH_CALUDE_toms_tickets_toms_remaining_tickets_l862_86248

/-- Tom's arcade tickets problem -/
theorem toms_tickets (whack_a_mole : ℕ) (skee_ball : ℕ) (spent : ℕ) : ℕ :=
  let total := whack_a_mole + skee_ball
  total - spent

/-- Proof of Tom's remaining tickets -/
theorem toms_remaining_tickets : toms_tickets 32 25 7 = 50 := by
  sorry

end NUMINAMATH_CALUDE_toms_tickets_toms_remaining_tickets_l862_86248


namespace NUMINAMATH_CALUDE_line_relationships_skew_lines_angle_range_line_plane_angle_range_dihedral_angle_range_parallel_line_plane_parallel_planes_perpendicular_line_plane_perpendicular_planes_l862_86273

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relationships
variable (parallel : Line → Line → Prop)
variable (intersect : Line → Line → Prop)
variable (skew : Line → Line → Prop)
variable (contained_in : Line → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (planes_parallel : Plane → Plane → Prop)
variable (planes_perpendicular : Plane → Plane → Prop)
variable (angle : Line → Line → ℝ)
variable (angle_line_plane : Line → Plane → ℝ)
variable (dihedral_angle : Plane → Plane → ℝ)

-- Theorem statements
theorem line_relationships (l1 l2 : Line) : 
  (parallel l1 l2 ∨ intersect l1 l2 ∨ skew l1 l2) ∧ 
  ¬(parallel l1 l2 ∧ intersect l1 l2) ∧ 
  ¬(parallel l1 l2 ∧ skew l1 l2) ∧ 
  ¬(intersect l1 l2 ∧ skew l1 l2) := sorry

theorem skew_lines_angle_range (l1 l2 : Line) (h : skew l1 l2) : 
  0 < angle l1 l2 ∧ angle l1 l2 ≤ Real.pi / 2 := sorry

theorem line_plane_angle_range (l : Line) (p : Plane) : 
  0 ≤ angle_line_plane l p ∧ angle_line_plane l p ≤ Real.pi / 2 := sorry

theorem dihedral_angle_range (p1 p2 : Plane) : 
  0 ≤ dihedral_angle p1 p2 ∧ dihedral_angle p1 p2 ≤ Real.pi := sorry

theorem parallel_line_plane (a b : Line) (α : Plane) 
  (h1 : ¬contained_in a α) (h2 : contained_in b α) (h3 : parallel a b) : 
  parallel_plane a α := sorry

theorem parallel_planes (a b : Line) (α β : Plane) (P : Point)
  (h1 : contained_in a β) (h2 : contained_in b β) 
  (h3 : intersect a b) (h4 : ¬parallel_plane a α) (h5 : ¬parallel_plane b α) : 
  planes_parallel α β := sorry

theorem perpendicular_line_plane (a b l : Line) (α : Plane) (A : Point)
  (h1 : contained_in a α) (h2 : contained_in b α) 
  (h3 : intersect a b) (h4 : perpendicular l a) (h5 : perpendicular l b) : 
  perpendicular_plane l α := sorry

theorem perpendicular_planes (l : Line) (α β : Plane)
  (h1 : perpendicular_plane l α) (h2 : contained_in l β) : 
  planes_perpendicular α β := sorry

end NUMINAMATH_CALUDE_line_relationships_skew_lines_angle_range_line_plane_angle_range_dihedral_angle_range_parallel_line_plane_parallel_planes_perpendicular_line_plane_perpendicular_planes_l862_86273


namespace NUMINAMATH_CALUDE_butterfat_mixture_l862_86279

theorem butterfat_mixture (initial_volume : ℝ) (initial_butterfat : ℝ) 
  (added_volume : ℝ) (added_butterfat : ℝ) (target_butterfat : ℝ) :
  initial_volume = 8 →
  initial_butterfat = 0.3 →
  added_butterfat = 0.1 →
  target_butterfat = 0.2 →
  added_volume = 8 →
  (initial_volume * initial_butterfat + added_volume * added_butterfat) / 
  (initial_volume + added_volume) = target_butterfat :=
by
  sorry

#check butterfat_mixture

end NUMINAMATH_CALUDE_butterfat_mixture_l862_86279


namespace NUMINAMATH_CALUDE_expression_simplification_l862_86203

theorem expression_simplification (a b c x : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  ((x + 2*a)^2) / ((a - b)*(a - c)) + ((x + 2*b)^2) / ((b - a)*(b - c)) + ((x + 2*c)^2) / ((c - a)*(c - b)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l862_86203


namespace NUMINAMATH_CALUDE_partner_q_active_months_l862_86271

/-- Represents the investment and activity of a partner in the business -/
structure Partner where
  investment : ℝ
  monthlyReturn : ℝ
  activeMonths : ℕ

/-- Represents the business venture with three partners -/
structure Business where
  p : Partner
  q : Partner
  r : Partner
  totalProfit : ℝ

/-- The main theorem stating that partner Q was active for 6 months -/
theorem partner_q_active_months (b : Business) : b.q.activeMonths = 6 :=
  by
  have h1 : b.p.investment / b.q.investment = 7 / 5.00001 := sorry
  have h2 : b.q.investment / b.r.investment = 5.00001 / 3.99999 := sorry
  have h3 : b.p.monthlyReturn / b.q.monthlyReturn = 7.00001 / 10 := sorry
  have h4 : b.q.monthlyReturn / b.r.monthlyReturn = 10 / 6 := sorry
  have h5 : b.p.activeMonths = 5 := sorry
  have h6 : b.r.activeMonths = 8 := sorry
  have h7 : b.totalProfit = 200000 := sorry
  have h8 : b.p.investment * b.p.monthlyReturn * b.p.activeMonths = 50000 := sorry
  sorry

end NUMINAMATH_CALUDE_partner_q_active_months_l862_86271


namespace NUMINAMATH_CALUDE_stone_placement_possible_l862_86213

/-- Represents the state of the stone placement game -/
structure GameState where
  cellStones : Nat → Bool
  bagStones : Nat

/-- Defines the allowed moves in the game -/
inductive Move
  | PlaceInFirst : Move
  | RemoveFromFirst : Move
  | PlaceInNext : Nat → Move
  | RemoveFromNext : Nat → Move

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.PlaceInFirst => sorry
  | Move.RemoveFromFirst => sorry
  | Move.PlaceInNext n => sorry
  | Move.RemoveFromNext n => sorry

/-- Checks if a cell contains a stone -/
def hasStone (state : GameState) (cell : Nat) : Bool :=
  state.cellStones cell

/-- The main theorem stating that with 10 stones, 
    we can place a stone in any cell from 1 to 1023 -/
theorem stone_placement_possible :
  ∀ n : Nat, n ≤ 1023 → 
  ∃ (moves : List Move), 
    let finalState := (moves.foldl applyMove 
      { cellStones := fun _ => false, bagStones := 10 })
    hasStone finalState n := by sorry

end NUMINAMATH_CALUDE_stone_placement_possible_l862_86213


namespace NUMINAMATH_CALUDE_movie_ticket_cost_proof_l862_86270

def movie_ticket_cost (total_money : ℚ) (change : ℚ) (num_sisters : ℕ) : ℚ :=
  (total_money - change) / num_sisters

theorem movie_ticket_cost_proof (total_money : ℚ) (change : ℚ) (num_sisters : ℕ) 
  (h1 : total_money = 25)
  (h2 : change = 9)
  (h3 : num_sisters = 2) :
  movie_ticket_cost total_money change num_sisters = 8 := by
  sorry

#eval movie_ticket_cost 25 9 2

end NUMINAMATH_CALUDE_movie_ticket_cost_proof_l862_86270


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l862_86285

def A : Set ℤ := {x | ∃ k : ℤ, x = 2 * k}
def B : Set ℤ := {-2, -1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {-2, 0, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l862_86285


namespace NUMINAMATH_CALUDE_gena_hits_target_l862_86227

/-- Calculates the number of hits given the total shots, initial shots, and additional shots per hit -/
def calculate_hits (total_shots initial_shots additional_shots_per_hit : ℕ) : ℕ :=
  (total_shots - initial_shots) / additional_shots_per_hit

/-- Theorem: Given the shooting range conditions, Gena hit the target 6 times -/
theorem gena_hits_target : 
  let initial_shots : ℕ := 5
  let additional_shots_per_hit : ℕ := 2
  let total_shots : ℕ := 17
  calculate_hits total_shots initial_shots additional_shots_per_hit = 6 := by
sorry

#eval calculate_hits 17 5 2

end NUMINAMATH_CALUDE_gena_hits_target_l862_86227


namespace NUMINAMATH_CALUDE_ball_max_height_l862_86241

/-- The path of a ball thrown on a planet with stronger gravity -/
def ballPath (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 60

/-- The maximum height reached by the ball -/
def maxHeight : ℝ := 140

theorem ball_max_height :
  ∃ t : ℝ, ballPath t = maxHeight ∧ ∀ s : ℝ, ballPath s ≤ maxHeight := by
  sorry

end NUMINAMATH_CALUDE_ball_max_height_l862_86241


namespace NUMINAMATH_CALUDE_min_value_and_fraction_sum_l862_86243

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + |2*x - 1|

-- Theorem statement
theorem min_value_and_fraction_sum :
  (∃ a : ℝ, (∀ x : ℝ, f x ≥ a) ∧ (∃ x₀ : ℝ, f x₀ = a) ∧ a = 3/2) ∧
  (∀ m n : ℝ, m > 0 → n > 0 → m + n = 3/2 → 1/m + 4/n ≥ 6) ∧
  (∃ m₀ n₀ : ℝ, m₀ > 0 ∧ n₀ > 0 ∧ m₀ + n₀ = 3/2 ∧ 1/m₀ + 4/n₀ = 6) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_fraction_sum_l862_86243


namespace NUMINAMATH_CALUDE_train_speed_calculation_l862_86277

/-- The speed of a train in km/hr -/
def train_speed : ℝ := 90

/-- The length of the train in meters -/
def train_length : ℝ := 750

/-- The time taken to cross the platform in minutes -/
def crossing_time : ℝ := 1

/-- The length of the platform in meters -/
def platform_length : ℝ := train_length

theorem train_speed_calculation :
  train_speed = (2 * train_length) / crossing_time * 60 / 1000 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l862_86277


namespace NUMINAMATH_CALUDE_inequality_proof_l862_86211

theorem inequality_proof (x y z : ℝ) (h : x + y + z = 1) :
  Real.sqrt (3 * x + 1) + Real.sqrt (3 * y + 2) + Real.sqrt (3 * z + 3) ≤ 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l862_86211


namespace NUMINAMATH_CALUDE_mean_books_read_l862_86258

def readers_3 : ℕ := 4
def books_3 : ℕ := 3
def readers_5 : ℕ := 5
def books_5 : ℕ := 5
def readers_7 : ℕ := 2
def books_7 : ℕ := 7
def readers_10 : ℕ := 1
def books_10 : ℕ := 10

def total_readers : ℕ := readers_3 + readers_5 + readers_7 + readers_10
def total_books : ℕ := readers_3 * books_3 + readers_5 * books_5 + readers_7 * books_7 + readers_10 * books_10

theorem mean_books_read :
  (total_books : ℚ) / (total_readers : ℚ) = 61 / 12 :=
sorry

end NUMINAMATH_CALUDE_mean_books_read_l862_86258


namespace NUMINAMATH_CALUDE_square_area_error_l862_86235

theorem square_area_error (x : ℝ) (h : x > 0) :
  let actual_edge := x
  let calculated_edge := x * (1 + 0.02)
  let actual_area := x^2
  let calculated_area := calculated_edge^2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.0404 := by sorry

end NUMINAMATH_CALUDE_square_area_error_l862_86235


namespace NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l862_86221

theorem quadratic_polynomial_satisfies_conditions :
  ∃ (q : ℝ → ℝ),
    (∀ x, q x = -3 * x^2 + 9 * x + 54) ∧
    q (-3) = 0 ∧
    q 6 = 0 ∧
    q 0 = -54 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l862_86221


namespace NUMINAMATH_CALUDE_line_l_equation_l862_86224

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 2 * x - y - 1 = 0
def l₂ (x y : ℝ) : Prop := x + y + 2 = 0

-- Define point P
def P : ℝ × ℝ := (2, 1)

-- Define the property of l passing through P
def passes_through_P (l : ℝ → ℝ → Prop) : Prop := l P.1 P.2

-- Define the intersection points A and B
def A (l : ℝ → ℝ → Prop) : ℝ × ℝ := sorry
def B (l : ℝ → ℝ → Prop) : ℝ × ℝ := sorry

-- Define the property of P being the midpoint of AB
def P_is_midpoint (l : ℝ → ℝ → Prop) : Prop :=
  P.1 = (A l).1 / 2 + (B l).1 / 2 ∧ P.2 = (A l).2 / 2 + (B l).2 / 2

-- Define the property of A and B being on l₁ and l₂ respectively
def A_on_l₁ (l : ℝ → ℝ → Prop) : Prop := l₁ (A l).1 (A l).2
def B_on_l₂ (l : ℝ → ℝ → Prop) : Prop := l₂ (B l).1 (B l).2

-- Define the equation of line l
def line_l (x y : ℝ) : Prop := 4 * x - y - 7 = 0

theorem line_l_equation : 
  ∀ l : ℝ → ℝ → Prop, 
    passes_through_P l → 
    P_is_midpoint l → 
    A_on_l₁ l → 
    B_on_l₂ l → 
    ∀ x y : ℝ, l x y ↔ line_l x y :=
sorry

end NUMINAMATH_CALUDE_line_l_equation_l862_86224


namespace NUMINAMATH_CALUDE_complex_roots_on_circle_l862_86288

theorem complex_roots_on_circle :
  ∀ (z : ℂ), (z + 2)^6 = 64 * z^6 →
  Complex.abs (z + 2/3) = 2/3 := by sorry

end NUMINAMATH_CALUDE_complex_roots_on_circle_l862_86288


namespace NUMINAMATH_CALUDE_complement_of_A_l862_86292

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | (x - 1) * (x + 2) > 0}

theorem complement_of_A : Set.compl A = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l862_86292


namespace NUMINAMATH_CALUDE_total_teachers_l862_86293

theorem total_teachers (num_departments : ℕ) (teachers_per_department : ℕ) 
  (h1 : num_departments = 15) 
  (h2 : teachers_per_department = 35) : 
  num_departments * teachers_per_department = 525 := by
  sorry

end NUMINAMATH_CALUDE_total_teachers_l862_86293


namespace NUMINAMATH_CALUDE_batch_size_proof_l862_86297

theorem batch_size_proof (x : ℕ) (N : ℕ) :
  (20 * (x - 1) = N) →                   -- Condition 1
  (∃ r : ℕ, r = 20) →                    -- Original rate
  ((25 * (x - 7)) = N - 80) →            -- Condition 2 (after rate increase)
  (x = 14) →                             -- Derived from solution
  N = 280 :=                             -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_batch_size_proof_l862_86297


namespace NUMINAMATH_CALUDE_position_of_b_l862_86295

theorem position_of_b (a b c : ℚ) (h : |a| + |b - c| = |a - c|) :
  (∃ a b c : ℚ, a < b ∧ c < b ∧ |a| + |b - c| = |a - c|) ∧
  (∃ a b c : ℚ, b < a ∧ b < c ∧ |a| + |b - c| = |a - c|) ∧
  (∃ a b c : ℚ, a < b ∧ b < c ∧ |a| + |b - c| = |a - c|) :=
sorry

end NUMINAMATH_CALUDE_position_of_b_l862_86295


namespace NUMINAMATH_CALUDE_total_exercise_time_l862_86246

def natasha_daily_exercise : ℕ := 30
def natasha_days : ℕ := 7
def esteban_daily_exercise : ℕ := 10
def esteban_days : ℕ := 9
def minutes_per_hour : ℕ := 60

theorem total_exercise_time :
  (natasha_daily_exercise * natasha_days + esteban_daily_exercise * esteban_days) / minutes_per_hour = 5 := by
  sorry

end NUMINAMATH_CALUDE_total_exercise_time_l862_86246


namespace NUMINAMATH_CALUDE_angle_calculation_l862_86296

/-- Given an angle α with its vertex at the origin, its initial side coinciding with
    the non-negative half-axis of the x-axis, and a point P(-2, -1) on its terminal side,
    prove that 2cos²α - sin(π - 2α) = 4/5 -/
theorem angle_calculation (α : ℝ) :
  (∃ (P : ℝ × ℝ), P = (-2, -1) ∧
    P.1 = Real.cos α * Real.sqrt 5 ∧
    P.2 = Real.sin α * Real.sqrt 5) →
  2 * (Real.cos α)^2 - Real.sin (π - 2 * α) = 4/5 :=
by sorry

end NUMINAMATH_CALUDE_angle_calculation_l862_86296


namespace NUMINAMATH_CALUDE_sin_alpha_value_l862_86240

theorem sin_alpha_value (m : ℝ) (α : ℝ) :
  (∃ P : ℝ × ℝ, P = (m, -3) ∧ P.1 * Real.cos α = P.2 * Real.sin α) →
  Real.tan α = -3/4 →
  Real.sin α = -3/5 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l862_86240


namespace NUMINAMATH_CALUDE_tennis_to_soccer_ratio_l862_86282

/-- Represents the number of balls of each type -/
structure BallCounts where
  total : ℕ
  soccer : ℕ
  basketball : ℕ
  baseball : ℕ
  volleyball : ℕ
  tennis : ℕ

/-- The ratio of two natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Theorem stating the ratio of tennis balls to soccer balls -/
theorem tennis_to_soccer_ratio (counts : BallCounts) : 
  counts.total = 145 →
  counts.soccer = 20 →
  counts.basketball = counts.soccer + 5 →
  counts.baseball = counts.soccer + 10 →
  counts.volleyball = 30 →
  counts.total = counts.soccer + counts.basketball + counts.baseball + counts.volleyball + counts.tennis →
  (Ratio.mk counts.tennis counts.soccer) = (Ratio.mk 2 1) := by
  sorry

end NUMINAMATH_CALUDE_tennis_to_soccer_ratio_l862_86282


namespace NUMINAMATH_CALUDE_divisibility_by_three_l862_86290

theorem divisibility_by_three (B : ℕ) : 
  B < 10 ∧ (5 + 2 + B + 6) % 3 = 0 ↔ B = 2 :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_three_l862_86290


namespace NUMINAMATH_CALUDE_dinner_bill_problem_l862_86283

theorem dinner_bill_problem (mother_charge : ℚ) (child_charge_per_year : ℚ) (total_bill : ℚ) :
  mother_charge = 5.95 →
  child_charge_per_year = 0.55 →
  total_bill = 11.15 →
  ∃ (triplet_age son_age : ℕ),
    son_age = 3 ∧
    triplet_age * child_charge_per_year * 3 + son_age * child_charge_per_year + mother_charge = total_bill :=
by sorry

end NUMINAMATH_CALUDE_dinner_bill_problem_l862_86283


namespace NUMINAMATH_CALUDE_airplane_average_speed_l862_86284

/-- Conversion factor from miles to kilometers -/
def miles_to_km : ℝ := 1.60934

/-- Distance traveled by the airplane in miles -/
def distance_miles : ℝ := 1584

/-- Time taken by the airplane in hours -/
def time_hours : ℝ := 24

/-- Theorem stating the average speed of the airplane -/
theorem airplane_average_speed :
  let distance_km := distance_miles * miles_to_km
  let average_speed := distance_km / time_hours
  ∃ ε > 0, |average_speed - 106.24| < ε :=
sorry

end NUMINAMATH_CALUDE_airplane_average_speed_l862_86284


namespace NUMINAMATH_CALUDE_stating_min_weighings_to_find_lighter_ball_l862_86287

/-- Represents the number of balls -/
def num_balls : ℕ := 9

/-- Represents the number of heavier balls -/
def num_heavy : ℕ := 8

/-- Represents the weight of the heavier balls in grams -/
def heavy_weight : ℕ := 10

/-- Represents the weight of the lighter ball in grams -/
def light_weight : ℕ := 9

/-- Represents the availability of a balance scale -/
def has_balance_scale : Prop := True

/-- 
Theorem stating that the minimum number of weighings required to find the lighter ball is 2
given the conditions of the problem.
-/
theorem min_weighings_to_find_lighter_ball :
  ∀ (balls : Fin num_balls → ℕ),
  (∃ (i : Fin num_balls), balls i = light_weight) ∧
  (∀ (i : Fin num_balls), balls i = light_weight ∨ balls i = heavy_weight) ∧
  has_balance_scale →
  (∃ (n : ℕ), n = 2 ∧ 
    ∀ (m : ℕ), (∃ (strategy : ℕ → ℕ → Bool), 
      (∀ (i : Fin num_balls), balls i = light_weight → 
        ∃ (k : Fin m), strategy k (balls i) = true) ∧
      (∀ (i j : Fin num_balls), i ≠ j → balls i ≠ balls j → 
        ∃ (k : Fin m), strategy k (balls i) ≠ strategy k (balls j))) → 
    m ≥ n) :=
sorry

end NUMINAMATH_CALUDE_stating_min_weighings_to_find_lighter_ball_l862_86287


namespace NUMINAMATH_CALUDE_merchant_profit_theorem_l862_86244

/-- Calculates the profit percentage given the ratio of articles sold to articles bought --/
def profit_percentage (articles_sold : ℕ) (articles_bought : ℕ) : ℚ :=
  ((articles_bought : ℚ) / (articles_sold : ℚ) - 1) * 100

/-- Proves that when 25 articles' cost price equals 18 articles' selling price, the profit is (7/18) * 100 percent --/
theorem merchant_profit_theorem (cost_price selling_price : ℚ) 
  (h : 25 * cost_price = 18 * selling_price) : 
  profit_percentage 18 25 = (7 / 18) * 100 := by
  sorry

#eval profit_percentage 18 25

end NUMINAMATH_CALUDE_merchant_profit_theorem_l862_86244


namespace NUMINAMATH_CALUDE_dusty_cake_purchase_l862_86280

/-- The number of double layer cake slices Dusty bought -/
def double_layer_slices : ℕ := sorry

/-- The price of a single layer cake slice in dollars -/
def single_layer_price : ℕ := 4

/-- The price of a double layer cake slice in dollars -/
def double_layer_price : ℕ := 7

/-- The number of single layer cake slices Dusty bought -/
def single_layer_bought : ℕ := 7

/-- The amount Dusty paid in dollars -/
def amount_paid : ℕ := 100

/-- The change Dusty received in dollars -/
def change_received : ℕ := 37

theorem dusty_cake_purchase : 
  double_layer_slices = 5 ∧
  amount_paid = 
    single_layer_price * single_layer_bought + 
    double_layer_price * double_layer_slices + 
    change_received :=
by sorry

end NUMINAMATH_CALUDE_dusty_cake_purchase_l862_86280


namespace NUMINAMATH_CALUDE_matrix_not_invertible_l862_86261

/-- A 2x2 matrix is not invertible if its determinant is zero. -/
def is_not_invertible (a b c d : ℚ) : Prop :=
  a * d - b * c = 0

/-- The matrix in question with x as a parameter. -/
def matrix (x : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![2 * x + 1, 9],
    ![4 - x, 10]]

/-- Theorem stating that the matrix is not invertible when x = 26/29. -/
theorem matrix_not_invertible :
  is_not_invertible (2 * (26/29) + 1) 9 (4 - (26/29)) 10 := by
  sorry

end NUMINAMATH_CALUDE_matrix_not_invertible_l862_86261


namespace NUMINAMATH_CALUDE_six_digit_numbers_with_zero_l862_86207

/-- The number of digits in the numbers we're considering -/
def num_digits : ℕ := 6

/-- The total number of possible digits (0 to 9) -/
def base : ℕ := 10

/-- The number of non-zero digits (1 to 9) -/
def non_zero_digits : ℕ := 9

/-- The total number of 6-digit numbers -/
def total_numbers : ℕ := non_zero_digits * base ^ (num_digits - 1)

/-- The number of 6-digit numbers with no zeros -/
def numbers_without_zero : ℕ := non_zero_digits ^ num_digits

/-- The number of 6-digit numbers with at least one zero -/
def numbers_with_zero : ℕ := total_numbers - numbers_without_zero

theorem six_digit_numbers_with_zero : numbers_with_zero = 368559 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_numbers_with_zero_l862_86207


namespace NUMINAMATH_CALUDE_defective_pens_count_l862_86206

def total_pens : ℕ := 10

def prob_non_defective : ℚ := 0.6222222222222222

theorem defective_pens_count (defective : ℕ) 
  (h1 : defective ≤ total_pens)
  (h2 : (((total_pens - defective) : ℚ) / total_pens) * 
        (((total_pens - defective - 1) : ℚ) / (total_pens - 1)) = prob_non_defective) :
  defective = 2 := by sorry

end NUMINAMATH_CALUDE_defective_pens_count_l862_86206


namespace NUMINAMATH_CALUDE_sock_selection_combinations_l862_86272

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem sock_selection_combinations :
  choose 7 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_sock_selection_combinations_l862_86272


namespace NUMINAMATH_CALUDE_cone_volume_l862_86201

/-- Given a cone with slant height 13 cm and height 12 cm, its volume is 100π cubic centimeters -/
theorem cone_volume (s h r : ℝ) (hs : s = 13) (hh : h = 12) 
  (hpythag : s^2 = h^2 + r^2) : (1/3 : ℝ) * π * r^2 * h = 100 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l862_86201


namespace NUMINAMATH_CALUDE_brownies_per_pan_l862_86204

/-- Proves that the number of pieces in each pan of brownies is 16 given the problem conditions --/
theorem brownies_per_pan (total_pans : ℕ) (eaten_pans : ℚ) (ice_cream_tubs : ℕ) 
  (scoops_per_tub : ℕ) (scoops_per_guest : ℕ) (guests_without_ice_cream : ℕ) :
  total_pans = 2 →
  eaten_pans = 1 + 3/4 →
  scoops_per_tub = 8 →
  scoops_per_guest = 2 →
  ice_cream_tubs = 6 →
  guests_without_ice_cream = 4 →
  ∃ (pieces_per_pan : ℕ), pieces_per_pan = 16 ∧ 
    (ice_cream_tubs * scoops_per_tub / scoops_per_guest + guests_without_ice_cream) / eaten_pans = pieces_per_pan := by
  sorry

#check brownies_per_pan

end NUMINAMATH_CALUDE_brownies_per_pan_l862_86204


namespace NUMINAMATH_CALUDE_initial_money_calculation_l862_86286

theorem initial_money_calculation (initial_money : ℚ) : 
  (2 / 5 : ℚ) * initial_money = 200 → initial_money = 500 := by
  sorry

#check initial_money_calculation

end NUMINAMATH_CALUDE_initial_money_calculation_l862_86286
