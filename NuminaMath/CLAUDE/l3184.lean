import Mathlib

namespace NUMINAMATH_CALUDE_fraction_equality_l3184_318424

theorem fraction_equality (w z : ℝ) (h : (1/w + 1/z)/(1/w - 1/z) = 2023) : 
  (w + z)/(w - z) = -2023 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l3184_318424


namespace NUMINAMATH_CALUDE_is_334th_term_l3184_318421

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem is_334th_term :
  arithmetic_sequence 7 6 334 = 2005 :=
by sorry

end NUMINAMATH_CALUDE_is_334th_term_l3184_318421


namespace NUMINAMATH_CALUDE_sandy_scooter_price_l3184_318439

/-- The initial price of Sandy's scooter -/
def initial_price : ℝ := 800

/-- The cost of repairs -/
def repair_cost : ℝ := 200

/-- The selling price of the scooter -/
def selling_price : ℝ := 1200

/-- The gain percentage -/
def gain_percent : ℝ := 20

theorem sandy_scooter_price :
  ∃ (P : ℝ),
    P = initial_price ∧
    selling_price = (1 + gain_percent / 100) * (P + repair_cost) :=
by sorry

end NUMINAMATH_CALUDE_sandy_scooter_price_l3184_318439


namespace NUMINAMATH_CALUDE_circle_condition_l3184_318451

/-- The equation x^2 + y^2 + ax - ay + 2 = 0 represents a circle if and only if a > 2 or a < -2 -/
theorem circle_condition (a : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + a*x - a*y + 2 = 0 ∧ 
   ∀ x' y' : ℝ, x'^2 + y'^2 + a*x' - a*y' + 2 = 0 → (x' - x)^2 + (y' - y)^2 = ((x' - x)^2 + (y' - y)^2)) 
  ↔ (a > 2 ∨ a < -2) :=
sorry

end NUMINAMATH_CALUDE_circle_condition_l3184_318451


namespace NUMINAMATH_CALUDE_fraction_equality_l3184_318471

theorem fraction_equality : (18 : ℚ) / (0.5 * 106) = 18 / 53 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3184_318471


namespace NUMINAMATH_CALUDE_larry_channels_l3184_318474

/-- Calculates the final number of channels Larry has after all changes --/
def final_channels (initial : ℕ) (removed : ℕ) (added : ℕ) (reduced : ℕ) (sports : ℕ) (supreme : ℕ) : ℕ :=
  initial - removed + added - reduced + sports + supreme

/-- Theorem stating that Larry's final number of channels is 147 --/
theorem larry_channels :
  final_channels 150 20 12 10 8 7 = 147 := by
  sorry

end NUMINAMATH_CALUDE_larry_channels_l3184_318474


namespace NUMINAMATH_CALUDE_frank_kibble_ratio_l3184_318494

-- Define the problem parameters
def initial_kibble : ℕ := 12
def remaining_kibble : ℕ := 7
def mary_total : ℕ := 2
def frank_afternoon : ℕ := 1

-- Define Frank's late evening amount
def frank_late_evening : ℕ := initial_kibble - remaining_kibble - mary_total - frank_afternoon

-- Theorem statement
theorem frank_kibble_ratio :
  frank_late_evening = 2 * frank_afternoon :=
sorry

end NUMINAMATH_CALUDE_frank_kibble_ratio_l3184_318494


namespace NUMINAMATH_CALUDE_additional_machines_for_half_time_l3184_318490

/-- Represents the number of machines needed to complete a job in a given time -/
def machines_needed (initial_machines : ℕ) (initial_days : ℕ) (new_days : ℕ) : ℕ :=
  initial_machines * initial_days / new_days

/-- Proof that 95 additional machines are needed to complete the job in half the time -/
theorem additional_machines_for_half_time (initial_machines : ℕ) (initial_days : ℕ) 
    (h1 : initial_machines = 5) (h2 : initial_days = 20) :
  machines_needed initial_machines initial_days (initial_days / 2) - initial_machines = 95 := by
  sorry

#eval machines_needed 5 20 10 - 5  -- Should output 95

end NUMINAMATH_CALUDE_additional_machines_for_half_time_l3184_318490


namespace NUMINAMATH_CALUDE_interest_calculation_l3184_318430

/-- Calculates the simple interest and proves the interest credited is 63 cents. -/
theorem interest_calculation (initial_savings : ℝ) (interest_rate : ℝ) (time : ℝ) 
  (additional_deposit : ℝ) (total_amount : ℝ) : ℝ :=
  let interest := initial_savings * interest_rate * time
  let amount_after_interest := initial_savings + interest
  let amount_after_deposit := amount_after_interest + additional_deposit
  let interest_credited := total_amount - (initial_savings + additional_deposit)
by
  have h1 : initial_savings = 500 := by sorry
  have h2 : interest_rate = 0.03 := by sorry
  have h3 : time = 1/4 := by sorry
  have h4 : additional_deposit = 15 := by sorry
  have h5 : total_amount = 515.63 := by sorry
  
  -- Prove that the interest credited is 63 cents
  sorry

#eval (515.63 - (500 + 15)) * 100 -- Should evaluate to 63.0

end NUMINAMATH_CALUDE_interest_calculation_l3184_318430


namespace NUMINAMATH_CALUDE_marcos_dad_strawberries_weight_l3184_318482

theorem marcos_dad_strawberries_weight (marco_weight dad_weight total_weight : ℕ) :
  marco_weight = 8 →
  total_weight = 40 →
  total_weight = marco_weight + dad_weight →
  dad_weight = 32 := by
sorry

end NUMINAMATH_CALUDE_marcos_dad_strawberries_weight_l3184_318482


namespace NUMINAMATH_CALUDE_distribute_five_contestants_three_companies_l3184_318456

/-- The number of ways to distribute contestants among companies -/
def distribute_contestants (num_contestants : ℕ) (num_companies : ℕ) : ℕ :=
  -- Definition goes here
  sorry

/-- Theorem: The number of ways to distribute 5 contestants among 3 companies,
    where each company must have at least 1 and at most 2 contestants, is 90 -/
theorem distribute_five_contestants_three_companies :
  distribute_contestants 5 3 = 90 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_contestants_three_companies_l3184_318456


namespace NUMINAMATH_CALUDE_dollar_calculation_l3184_318418

-- Define the $ operation
def dollar (a b : ℝ) : ℝ := (a^2 - b^2)^2

-- Theorem statement
theorem dollar_calculation (x : ℝ) : 
  dollar (x^3 + x) (x - x^3) = 16 * x^8 := by
  sorry

end NUMINAMATH_CALUDE_dollar_calculation_l3184_318418


namespace NUMINAMATH_CALUDE_zhang_wang_sum_difference_l3184_318447

/-- Sum of arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * a₁ + (n * (n - 1) * d) / 2

/-- Sum of 26 consecutive odd numbers starting from 27 -/
def zhang_sum : ℕ := arithmetic_sum 27 2 26

/-- Sum of 26 consecutive natural numbers starting from 26 -/
def wang_sum : ℕ := arithmetic_sum 26 1 26

theorem zhang_wang_sum_difference :
  zhang_sum - wang_sum = 351 := by sorry

end NUMINAMATH_CALUDE_zhang_wang_sum_difference_l3184_318447


namespace NUMINAMATH_CALUDE_hyperbola_construction_equivalence_l3184_318453

/-- The equation of a hyperbola in standard form -/
def is_hyperbola_point (a b x y : ℝ) : Prop :=
  (x / a)^2 - (y / b)^2 = 1

/-- The construction equation for a point on the hyperbola -/
def satisfies_construction (a b x y : ℝ) : Prop :=
  x = (a / b) * Real.sqrt (b^2 + y^2)

/-- Theorem: Any point satisfying the hyperbola equation also satisfies the construction equation -/
theorem hyperbola_construction_equivalence (a b x y : ℝ) (ha : a > 0) (hb : b > 0) :
  is_hyperbola_point a b x y → satisfies_construction a b x y :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_construction_equivalence_l3184_318453


namespace NUMINAMATH_CALUDE_pyramid_multiplication_l3184_318475

theorem pyramid_multiplication (z x : ℕ) : z = 2 → x = 24 →
  (12 * x = 84 ∧ x * 7 = 168 ∧ 12 * z = x) := by
  sorry

end NUMINAMATH_CALUDE_pyramid_multiplication_l3184_318475


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3184_318481

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geo : GeometricSequence a)
  (h1 : a 2010 * a 2011 * a 2012 = 3)
  (h2 : a 2013 * a 2014 * a 2015 = 24) :
  a 2016 * a 2017 * a 2018 = 192 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3184_318481


namespace NUMINAMATH_CALUDE_quadratic_ratio_l3184_318473

theorem quadratic_ratio (b c : ℝ) : 
  (∀ x, x^2 + 1500*x + 2400 = (x + b)^2 + c) → 
  c / b = -746.8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_ratio_l3184_318473


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3184_318469

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x < 0}

-- Define set B
def B : Set ℝ := {x | 1/3 ≤ x ∧ x ≤ 5}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1/3 ≤ x ∧ x < 4} :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3184_318469


namespace NUMINAMATH_CALUDE_jacks_total_yen_l3184_318444

/-- Represents the amount of money Jack has in different currencies -/
structure JacksMoney where
  pounds : ℕ
  euros : ℕ
  yen : ℕ

/-- Represents the exchange rates between currencies -/
structure ExchangeRates where
  pounds_per_euro : ℕ
  yen_per_pound : ℕ

/-- Calculates the total amount of yen Jack has -/
def total_yen (money : JacksMoney) (rates : ExchangeRates) : ℕ :=
  money.yen +
  money.pounds * rates.yen_per_pound +
  money.euros * rates.pounds_per_euro * rates.yen_per_pound

/-- Theorem stating that Jack's total amount in yen is 9400 -/
theorem jacks_total_yen :
  let money := JacksMoney.mk 42 11 3000
  let rates := ExchangeRates.mk 2 100
  total_yen money rates = 9400 := by
  sorry

end NUMINAMATH_CALUDE_jacks_total_yen_l3184_318444


namespace NUMINAMATH_CALUDE_flag_distribution_theorem_l3184_318492

/-- Represents the box of flags -/
structure FlagBox where
  total : ℕ
  blue : ℕ
  red : ℕ

/-- Represents the distribution of flags among children -/
structure FlagDistribution where
  total : ℕ
  blue : ℕ
  red : ℕ
  both : ℕ

def is_valid_box (box : FlagBox) : Prop :=
  box.total = box.blue + box.red ∧ box.total % 2 = 0

def is_valid_distribution (box : FlagBox) (dist : FlagDistribution) : Prop :=
  dist.total = box.total / 2 ∧
  dist.blue = (6 * dist.total) / 10 ∧
  dist.red = (6 * dist.total) / 10 ∧
  dist.total = dist.blue + dist.red - dist.both

theorem flag_distribution_theorem (box : FlagBox) (dist : FlagDistribution) :
  is_valid_box box → is_valid_distribution box dist →
  dist.both = dist.total / 5 :=
sorry

end NUMINAMATH_CALUDE_flag_distribution_theorem_l3184_318492


namespace NUMINAMATH_CALUDE_alphazian_lost_words_l3184_318443

/-- The number of letters in the Alphazian alphabet -/
def alphabet_size : ℕ := 128

/-- The number of forbidden letters -/
def forbidden_letters : ℕ := 2

/-- The maximum word length in Alphazia -/
def max_word_length : ℕ := 2

/-- Calculates the number of lost words due to letter prohibition in Alphazia -/
def lost_words : ℕ :=
  forbidden_letters + (alphabet_size * forbidden_letters)

theorem alphazian_lost_words :
  lost_words = 258 := by sorry

end NUMINAMATH_CALUDE_alphazian_lost_words_l3184_318443


namespace NUMINAMATH_CALUDE_sum_reciprocals_bound_l3184_318495

theorem sum_reciprocals_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (1 / a + 1 / b) ≥ 2 ∧ ∀ M : ℝ, ∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧ a' + b' = 2 ∧ 1 / a' + 1 / b' > M :=
by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_bound_l3184_318495


namespace NUMINAMATH_CALUDE_oil_cylinder_capacity_l3184_318485

theorem oil_cylinder_capacity : ∀ (C : ℚ),
  (4 / 5 : ℚ) * C - (3 / 4 : ℚ) * C = 4 →
  C = 80 := by
sorry

end NUMINAMATH_CALUDE_oil_cylinder_capacity_l3184_318485


namespace NUMINAMATH_CALUDE_shirt_ratio_l3184_318414

/-- Given that Hazel received 6 shirts and the total number of shirts is 18,
    prove that the ratio of Razel's shirts to Hazel's shirts is 2:1. -/
theorem shirt_ratio (hazel_shirts : ℕ) (total_shirts : ℕ) (razel_shirts : ℕ) : 
  hazel_shirts = 6 → total_shirts = 18 → razel_shirts = total_shirts - hazel_shirts →
  (razel_shirts : ℚ) / hazel_shirts = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_shirt_ratio_l3184_318414


namespace NUMINAMATH_CALUDE_square_root_squared_sqrt_2023_squared_l3184_318433

theorem square_root_squared (x : ℝ) (h : x > 0) : (Real.sqrt x)^2 = x := by sorry

theorem sqrt_2023_squared : (Real.sqrt 2023)^2 = 2023 := by
  apply square_root_squared
  norm_num

end NUMINAMATH_CALUDE_square_root_squared_sqrt_2023_squared_l3184_318433


namespace NUMINAMATH_CALUDE_surface_classification_l3184_318410

/-- A surface in 3D space -/
inductive Surface
  | CircularCone
  | OneSheetHyperboloid
  | TwoSheetHyperboloid
  | EllipticParaboloid

/-- Determine the type of surface given its equation -/
def determine_surface_type (equation : ℝ → ℝ → ℝ → Prop) : Surface :=
  sorry

theorem surface_classification :
  (determine_surface_type (fun x y z => x^2 - y^2 = z^2) = Surface.CircularCone) ∧
  (determine_surface_type (fun x y z => -2*x^2 + 2*y^2 + z^2 = 4) = Surface.OneSheetHyperboloid) ∧
  (determine_surface_type (fun x y z => 2*x^2 - y^2 + z^2 + 2 = 0) = Surface.TwoSheetHyperboloid) ∧
  (determine_surface_type (fun x y z => 3*y^2 + 2*z^2 = 6*x) = Surface.EllipticParaboloid) :=
by
  sorry

end NUMINAMATH_CALUDE_surface_classification_l3184_318410


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_set_l3184_318432

theorem fraction_inequality_solution_set (x : ℝ) :
  (x ≠ -1) → ((2 - x) / (x + 1) ≥ 0 ↔ -1 < x ∧ x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_set_l3184_318432


namespace NUMINAMATH_CALUDE_prob_sum_odd_is_13_27_l3184_318458

/-- Represents an unfair die where even numbers are twice as likely as odd numbers -/
structure UnfairDie where
  /-- Probability of rolling an odd number -/
  odd_prob : ℝ
  /-- Probability of rolling an even number -/
  even_prob : ℝ
  /-- The probabilities sum to 1 -/
  prob_sum : odd_prob + even_prob = 1
  /-- Even numbers are twice as likely as odd numbers -/
  even_twice_odd : even_prob = 2 * odd_prob

/-- The probability of rolling a sum of three rolls being odd -/
def prob_sum_odd (d : UnfairDie) : ℝ :=
  3 * d.odd_prob * d.even_prob^2 + d.odd_prob^3

/-- Theorem stating the probability of rolling a sum of three rolls being odd is 13/27 -/
theorem prob_sum_odd_is_13_27 (d : UnfairDie) : prob_sum_odd d = 13/27 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_odd_is_13_27_l3184_318458


namespace NUMINAMATH_CALUDE_probability_neither_red_nor_purple_l3184_318486

theorem probability_neither_red_nor_purple :
  let total_balls : ℕ := 120
  let red_balls : ℕ := 15
  let purple_balls : ℕ := 3
  let neither_red_nor_purple : ℕ := total_balls - (red_balls + purple_balls)
  (neither_red_nor_purple : ℚ) / total_balls = 17 / 20 := by
  sorry

end NUMINAMATH_CALUDE_probability_neither_red_nor_purple_l3184_318486


namespace NUMINAMATH_CALUDE_building_height_l3184_318416

/-- Given a flagpole and a building casting shadows under similar conditions,
    calculate the height of the building. -/
theorem building_height
  (flagpole_height : ℝ)
  (flagpole_shadow : ℝ)
  (building_shadow : ℝ)
  (h_flagpole : flagpole_height = 18)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building_shadow : building_shadow = 65)
  : (flagpole_height * building_shadow) / flagpole_shadow = 26 := by
  sorry

#check building_height

end NUMINAMATH_CALUDE_building_height_l3184_318416


namespace NUMINAMATH_CALUDE_part_to_whole_ratio_l3184_318460

theorem part_to_whole_ratio (N P : ℚ) (h1 : N = 240) (h2 : P + 6 = N / 4 - 6) : 
  (P + 6) / N = 9 / 40 := by
  sorry

end NUMINAMATH_CALUDE_part_to_whole_ratio_l3184_318460


namespace NUMINAMATH_CALUDE_solve_system_l3184_318480

theorem solve_system (x y : ℝ) (eq1 : 3 * x - 2 * y = 18) (eq2 : x + 2 * y = 10) : y = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l3184_318480


namespace NUMINAMATH_CALUDE_real_solutions_iff_a_in_range_l3184_318465

/-- Given a system of equations with real parameter a, 
    prove that real solutions exist if and only if 1 ≤ a ≤ 2 -/
theorem real_solutions_iff_a_in_range (a : ℝ) :
  (∃ x y : ℝ, x + y = a * (Real.sqrt x - Real.sqrt y) ∧
               x^2 + y^2 = a * (Real.sqrt x - Real.sqrt y)^2) ↔
  1 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_real_solutions_iff_a_in_range_l3184_318465


namespace NUMINAMATH_CALUDE_preschool_nap_problem_l3184_318498

theorem preschool_nap_problem (initial_kids : ℕ) (awake_after_first_round : ℕ) (awake_after_second_round : ℕ) : 
  initial_kids = 20 →
  awake_after_first_round = initial_kids - initial_kids / 2 →
  awake_after_second_round = awake_after_first_round - awake_after_first_round / 2 →
  awake_after_second_round = 5 :=
by sorry

end NUMINAMATH_CALUDE_preschool_nap_problem_l3184_318498


namespace NUMINAMATH_CALUDE_existence_of_nine_digit_combination_l3184_318440

theorem existence_of_nine_digit_combination : ∃ (a b c d e f g h i : ℕ),
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
  g ≠ h ∧ g ≠ i ∧
  h ≠ i ∧
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧ g < 10 ∧ h < 10 ∧ i < 100 ∧
  (a + b + c + d + e + f + g + h * i = 100 ∨
   a + b + c + d + e * f + g + h = 100 ∨
   a + b + c + d + e - f - g + h + i = 100) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_nine_digit_combination_l3184_318440


namespace NUMINAMATH_CALUDE_smallest_cut_for_non_triangle_l3184_318467

theorem smallest_cut_for_non_triangle (a b c : ℝ) (ha : a = 10) (hb : b = 24) (hc : c = 26) :
  let f := fun x => (a - x) + (b - x) ≤ (c - x)
  ∃ x₀ : ℝ, x₀ = 8 ∧ (∀ x, 0 ≤ x ∧ x < a → (f x → x ≥ x₀) ∧ (x < x₀ → ¬f x)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_cut_for_non_triangle_l3184_318467


namespace NUMINAMATH_CALUDE_student_pet_difference_l3184_318441

/-- Represents a fourth-grade classroom at Pine Hill Elementary -/
structure Classroom where
  students : ℕ
  rabbits : ℕ
  hamsters : ℕ

/-- Creates a standard fourth-grade classroom -/
def standard_classroom : Classroom :=
  { students := 20, rabbits := 2, hamsters := 1 }

/-- Calculates the total number of pets in a classroom -/
def pets_in_classroom (c : Classroom) : ℕ :=
  c.rabbits + c.hamsters

/-- Calculates the total number of students in n classrooms -/
def total_students (n : ℕ) : ℕ :=
  n * standard_classroom.students

/-- Calculates the total number of pets in n classrooms -/
def total_pets (n : ℕ) : ℕ :=
  n * pets_in_classroom standard_classroom

/-- The main theorem: difference between students and pets in 5 classrooms is 85 -/
theorem student_pet_difference : total_students 5 - total_pets 5 = 85 := by
  sorry

end NUMINAMATH_CALUDE_student_pet_difference_l3184_318441


namespace NUMINAMATH_CALUDE_sum_of_number_and_its_square_l3184_318450

theorem sum_of_number_and_its_square : 17 + 17^2 = 306 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_number_and_its_square_l3184_318450


namespace NUMINAMATH_CALUDE_sales_difference_is_25_l3184_318402

-- Define the prices and quantities for each company
def company_a_price : ℝ := 4
def company_b_price : ℝ := 3.5
def company_a_quantity : ℕ := 300
def company_b_quantity : ℕ := 350

-- Define the sales difference function
def sales_difference : ℝ :=
  (company_b_price * company_b_quantity) - (company_a_price * company_a_quantity)

-- Theorem statement
theorem sales_difference_is_25 : sales_difference = 25 := by
  sorry

end NUMINAMATH_CALUDE_sales_difference_is_25_l3184_318402


namespace NUMINAMATH_CALUDE_calculate_F_of_5_f_6_l3184_318497

-- Define the functions f and F
def f (a : ℝ) : ℝ := a + 3
def F (a b : ℝ) : ℝ := b^3 - 2*a

-- State the theorem
theorem calculate_F_of_5_f_6 : F 5 (f 6) = 719 := by
  sorry

end NUMINAMATH_CALUDE_calculate_F_of_5_f_6_l3184_318497


namespace NUMINAMATH_CALUDE_line_perp_plane_iff_planes_perp_l3184_318489

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes and between a line and a plane
variable (perpPlanes : Plane → Plane → Prop)
variable (perpLinePlane : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (subset : Line → Plane → Prop)

variable (α β : Plane)
variable (l : Line)

-- State the theorem
theorem line_perp_plane_iff_planes_perp 
  (h_intersect : α ≠ β) 
  (h_subset : subset l α) :
  perpLinePlane l β ↔ perpPlanes α β := by
  sorry

end NUMINAMATH_CALUDE_line_perp_plane_iff_planes_perp_l3184_318489


namespace NUMINAMATH_CALUDE_ripe_oranges_count_l3184_318479

/-- The number of sacks of unripe oranges harvested per day -/
def unripe_oranges : ℕ := 25

/-- The difference between the number of sacks of ripe and unripe oranges -/
def difference : ℕ := 19

/-- The number of sacks of ripe oranges harvested per day -/
def ripe_oranges : ℕ := unripe_oranges + difference

theorem ripe_oranges_count : ripe_oranges = 44 := by
  sorry

end NUMINAMATH_CALUDE_ripe_oranges_count_l3184_318479


namespace NUMINAMATH_CALUDE_johnson_family_seating_l3184_318499

/-- The number of ways to seat 5 daughters and 3 sons in a row of 8 chairs
    such that at least 2 girls are next to each other -/
def seatingArrangements (total : ℕ) (daughters : ℕ) (sons : ℕ) : ℕ :=
  Nat.factorial total - (Nat.factorial sons * Nat.factorial daughters)

/-- Theorem stating that the number of seating arrangements for the Johnson family
    with at least 2 girls next to each other is 39600 -/
theorem johnson_family_seating :
  seatingArrangements 8 5 3 = 39600 := by
  sorry

#eval seatingArrangements 8 5 3

end NUMINAMATH_CALUDE_johnson_family_seating_l3184_318499


namespace NUMINAMATH_CALUDE_initial_boys_count_l3184_318462

theorem initial_boys_count (initial_total : ℕ) (initial_boys : ℕ) (final_boys : ℕ) : 
  initial_boys = initial_total / 2 →                   -- Initially, 50% are boys
  final_boys = initial_boys - 3 →                      -- 3 boys leave
  final_boys * 10 = 4 * initial_total →                -- After changes, 40% are boys
  initial_boys = 15 := by
sorry

end NUMINAMATH_CALUDE_initial_boys_count_l3184_318462


namespace NUMINAMATH_CALUDE_fibonacci_identity_l3184_318493

-- Define Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

theorem fibonacci_identity (θ : ℝ) (x : ℝ) (n : ℕ) 
  (h1 : 0 < θ ∧ θ < π)
  (h2 : x + 1/x = 2 * Real.cos (2 * θ)) :
  x^(fib n) + 1/(x^(fib n)) = 2 * Real.cos (2 * (fib n) * θ) := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_identity_l3184_318493


namespace NUMINAMATH_CALUDE_sin_monotone_increasing_interval_l3184_318445

/-- The function f(x) = sin(2π/3 - 2x) is monotonically increasing on the interval [7π/12, 13π/12] -/
theorem sin_monotone_increasing_interval :
  let f : ℝ → ℝ := λ x => Real.sin (2 * Real.pi / 3 - 2 * x)
  ∀ x y, 7 * Real.pi / 12 ≤ x ∧ x < y ∧ y ≤ 13 * Real.pi / 12 → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_sin_monotone_increasing_interval_l3184_318445


namespace NUMINAMATH_CALUDE_cost_of_eggs_l3184_318484

/-- The amount Samantha spent on the crate of eggs -/
def cost : ℝ := 5

/-- The number of eggs in the crate -/
def total_eggs : ℕ := 30

/-- The price of each egg in dollars -/
def price_per_egg : ℝ := 0.20

/-- The number of eggs left when Samantha recovers her capital -/
def eggs_left : ℕ := 5

/-- Theorem stating that the cost of the crate is $5 -/
theorem cost_of_eggs : cost = (total_eggs - eggs_left) * price_per_egg := by
  sorry

end NUMINAMATH_CALUDE_cost_of_eggs_l3184_318484


namespace NUMINAMATH_CALUDE_prime_natural_equation_solutions_l3184_318436

theorem prime_natural_equation_solutions :
  ∀ p n : ℕ,
    Prime p →
    p^2 + n^2 = 3*p*n + 1 →
    ((p = 3 ∧ n = 1) ∨ (p = 3 ∧ n = 8)) := by
  sorry

end NUMINAMATH_CALUDE_prime_natural_equation_solutions_l3184_318436


namespace NUMINAMATH_CALUDE_fourth_power_equation_l3184_318411

theorem fourth_power_equation : 10^4 + 15^4 + 8^4 + 2*3^4 = 16^4 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_equation_l3184_318411


namespace NUMINAMATH_CALUDE_common_divisors_count_l3184_318449

def a : Nat := 12600
def b : Nat := 14400

theorem common_divisors_count : (Nat.divisors (Nat.gcd a b)).card = 45 := by
  sorry

end NUMINAMATH_CALUDE_common_divisors_count_l3184_318449


namespace NUMINAMATH_CALUDE_inheritance_division_l3184_318454

/-- Proves that dividing $527,500 equally among 5 people results in each person receiving $105,500 -/
theorem inheritance_division (total_amount : ℕ) (num_people : ℕ) (individual_share : ℕ) : 
  total_amount = 527500 → num_people = 5 → individual_share = total_amount / num_people → 
  individual_share = 105500 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_division_l3184_318454


namespace NUMINAMATH_CALUDE_pizza_remainder_l3184_318408

theorem pizza_remainder (john_portion emma_fraction : ℚ) : 
  john_portion = 4/5 →
  emma_fraction = 1/4 →
  (1 - john_portion) * (1 - emma_fraction) = 3/20 :=
by sorry

end NUMINAMATH_CALUDE_pizza_remainder_l3184_318408


namespace NUMINAMATH_CALUDE_congruent_to_one_mod_seven_l3184_318413

theorem congruent_to_one_mod_seven (n : ℕ) : 
  (Finset.filter (fun k => k % 7 = 1) (Finset.range 300)).card = 43 := by
  sorry

end NUMINAMATH_CALUDE_congruent_to_one_mod_seven_l3184_318413


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_specific_quadratic_roots_difference_l3184_318483

theorem quadratic_roots_difference (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * (x^2) + b * x + c = 0 → |r₁ - r₂| = Real.sqrt ((b^2 - 4*a*c) / (a^2)) :=
by sorry

theorem specific_quadratic_roots_difference :
  let r₁ := (-(-7) + Real.sqrt ((-7)^2 - 4*1*12)) / (2*1)
  let r₂ := (-(-7) - Real.sqrt ((-7)^2 - 4*1*12)) / (2*1)
  |r₁ - r₂| = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_specific_quadratic_roots_difference_l3184_318483


namespace NUMINAMATH_CALUDE_problem_solution_l3184_318412

theorem problem_solution :
  (∀ a b : ℝ, 4 * a^4 * b^3 / (-2 * a * b)^2 = a^2 * b) ∧
  (∀ x y : ℝ, (3*x - y)^2 - (3*x + 2*y) * (3*x - 2*y) = -6*x*y + 5*y^2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3184_318412


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l3184_318419

theorem fraction_equals_zero (x : ℝ) : (x - 2) / (x + 5) = 0 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l3184_318419


namespace NUMINAMATH_CALUDE_andys_walk_distance_l3184_318491

/-- Proves the distance between Andy's house and the market given his walking routes --/
theorem andys_walk_distance (house_to_school : ℝ) (school_to_park : ℝ) (total_distance : ℝ)
  (h1 : house_to_school = 50)
  (h2 : school_to_park = 25)
  (h3 : total_distance = 345) :
  total_distance - (2 * house_to_school + school_to_park + school_to_park / 2) = 195 := by
  sorry


end NUMINAMATH_CALUDE_andys_walk_distance_l3184_318491


namespace NUMINAMATH_CALUDE_megan_museum_pictures_l3184_318403

/-- Represents the number of pictures Megan took at the museum -/
def museum_pictures : ℕ := 18

/-- Represents the number of pictures Megan took at the zoo -/
def zoo_pictures : ℕ := 15

/-- Represents the number of pictures Megan deleted -/
def deleted_pictures : ℕ := 31

/-- Represents the number of pictures Megan had left after deleting -/
def remaining_pictures : ℕ := 2

theorem megan_museum_pictures :
  zoo_pictures + museum_pictures = remaining_pictures + deleted_pictures :=
by sorry

end NUMINAMATH_CALUDE_megan_museum_pictures_l3184_318403


namespace NUMINAMATH_CALUDE_continuous_fraction_identity_l3184_318452

-- Define x as the solution to the equation x = 1 + √3 / x
noncomputable def x : ℝ := Real.sqrt 3 + 1

-- State the theorem
theorem continuous_fraction_identity :
  1 / ((x + 2) * (x - 3)) = (Real.sqrt 3 + 6) / (-33) := by
  sorry

end NUMINAMATH_CALUDE_continuous_fraction_identity_l3184_318452


namespace NUMINAMATH_CALUDE_heartsuit_ratio_theorem_l3184_318464

-- Define the ♡ operation
def heartsuit (n m : ℕ) : ℕ := n^2 * m^3

-- Theorem statement
theorem heartsuit_ratio_theorem : 
  (heartsuit 3 5) / (heartsuit 5 3) = 5 / 3 := by sorry

end NUMINAMATH_CALUDE_heartsuit_ratio_theorem_l3184_318464


namespace NUMINAMATH_CALUDE_friends_receiving_pens_l3184_318468

/-- The number of friends Kendra and Tony will give pens to -/
def num_friends (kendra_packs tony_packs kendra_pens_per_pack tony_pens_per_pack kept_pens : ℕ) : ℕ :=
  kendra_packs * kendra_pens_per_pack + tony_packs * tony_pens_per_pack - 2 * kept_pens

/-- Theorem stating the number of friends Kendra and Tony will give pens to -/
theorem friends_receiving_pens :
  num_friends 7 5 4 6 3 = 52 := by
  sorry

end NUMINAMATH_CALUDE_friends_receiving_pens_l3184_318468


namespace NUMINAMATH_CALUDE_parabola_hyperbola_tangent_l3184_318429

theorem parabola_hyperbola_tangent (a b p k : ℝ) : 
  a > 0 → 
  b > 0 → 
  p > 0 → 
  (2 * a = 4 * Real.sqrt 2) → 
  (b = p / 2) → 
  (k = p / (4 * Real.sqrt 2)) → 
  (∀ x y : ℝ, y = k * x - 1 → x^2 = 2 * p * y) →
  p = 4 :=
by sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_tangent_l3184_318429


namespace NUMINAMATH_CALUDE_bank_account_increase_percentage_l3184_318477

def al_initial_balance : ℝ := 236.36
def eliot_initial_balance : ℝ := 200

theorem bank_account_increase_percentage :
  (al_initial_balance > eliot_initial_balance) →
  (al_initial_balance - eliot_initial_balance = (al_initial_balance + eliot_initial_balance) / 12) →
  (∃ p : ℝ, (al_initial_balance * 1.1 = eliot_initial_balance * (1 + p / 100) + 20) ∧ p = 20) :=
by sorry

end NUMINAMATH_CALUDE_bank_account_increase_percentage_l3184_318477


namespace NUMINAMATH_CALUDE_xyz_product_l3184_318438

theorem xyz_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x + 1/y = 5) (eq2 : y + 1/z = 2) (eq3 : z + 2/x = 10/3) :
  x * y * z = (21 + Real.sqrt 433) / 2 := by
  sorry

end NUMINAMATH_CALUDE_xyz_product_l3184_318438


namespace NUMINAMATH_CALUDE_rachel_book_count_l3184_318404

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 9

/-- The number of shelves with mystery books -/
def mystery_shelves : ℕ := 6

/-- The number of shelves with picture books -/
def picture_shelves : ℕ := 2

/-- The total number of books Rachel has -/
def total_books : ℕ := books_per_shelf * (mystery_shelves + picture_shelves)

theorem rachel_book_count : total_books = 72 := by
  sorry

end NUMINAMATH_CALUDE_rachel_book_count_l3184_318404


namespace NUMINAMATH_CALUDE_fifth_root_division_l3184_318446

theorem fifth_root_division (x : ℝ) (h : x > 0) :
  (x ^ (1 / 3)) / (x ^ (1 / 5)) = x ^ (2 / 15) :=
sorry

end NUMINAMATH_CALUDE_fifth_root_division_l3184_318446


namespace NUMINAMATH_CALUDE_shoe_size_age_game_l3184_318448

theorem shoe_size_age_game (shoe_size age : ℕ) : 
  let current_year := 1952
  let birth_year := current_year - age
  let game_result := ((shoe_size + 7) * 2 + 5) * 50 + 1711 - birth_year
  game_result = 5059 → shoe_size = 43 ∧ age = 50 := by
sorry

end NUMINAMATH_CALUDE_shoe_size_age_game_l3184_318448


namespace NUMINAMATH_CALUDE_cylinder_cone_volume_l3184_318417

/-- Given a cylinder and cone with the same base and height, and a combined volume of 48cm³,
    prove that the volume of the cylinder is 36cm³ and the volume of the cone is 12cm³. -/
theorem cylinder_cone_volume (cylinder_volume cone_volume : ℝ) : 
  cylinder_volume + cone_volume = 48 →
  cylinder_volume = 3 * cone_volume →
  cylinder_volume = 36 ∧ cone_volume = 12 := by
sorry

end NUMINAMATH_CALUDE_cylinder_cone_volume_l3184_318417


namespace NUMINAMATH_CALUDE_smallest_non_prime_non_square_no_small_factors_l3184_318476

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def has_no_prime_factor_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p < k → ¬(n % p = 0)

theorem smallest_non_prime_non_square_no_small_factors : 
  (∀ m : ℕ, m < 4091 → 
    is_prime m ∨ 
    is_perfect_square m ∨ 
    ¬(has_no_prime_factor_less_than m 60)) ∧ 
  ¬(is_prime 4091) ∧ 
  ¬(is_perfect_square 4091) ∧ 
  has_no_prime_factor_less_than 4091 60 :=
sorry

end NUMINAMATH_CALUDE_smallest_non_prime_non_square_no_small_factors_l3184_318476


namespace NUMINAMATH_CALUDE_multiplication_addition_equality_l3184_318401

theorem multiplication_addition_equality : 45 * 72 + 28 * 45 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_addition_equality_l3184_318401


namespace NUMINAMATH_CALUDE_negation_of_existence_l3184_318405

theorem negation_of_existence (T S : Type → Prop) : 
  (¬ ∃ x, T x ∧ S x) ↔ (∀ x, T x → ¬ S x) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l3184_318405


namespace NUMINAMATH_CALUDE_total_games_is_140_l3184_318422

/-- The number of teams in the "High School Ten" basketball conference -/
def num_teams : ℕ := 10

/-- The number of times each team plays every other team in the conference -/
def games_against_each_team : ℕ := 2

/-- The number of games each team plays against non-conference opponents -/
def non_conference_games_per_team : ℕ := 5

/-- The total number of games in a season involving the "High School Ten" teams -/
def total_games : ℕ := (num_teams * (num_teams - 1) / 2) * games_against_each_team + num_teams * non_conference_games_per_team

/-- Theorem stating that the total number of games in a season is 140 -/
theorem total_games_is_140 : total_games = 140 := by
  sorry

end NUMINAMATH_CALUDE_total_games_is_140_l3184_318422


namespace NUMINAMATH_CALUDE_always_positive_l3184_318470

theorem always_positive (x y : ℝ) : 3 * x^2 - 8 * x * y + 9 * y^2 - 4 * x + 6 * y + 13 > 0 := by
  sorry

end NUMINAMATH_CALUDE_always_positive_l3184_318470


namespace NUMINAMATH_CALUDE_factory_workers_count_l3184_318463

/-- The total number of workers in the factory -/
def total_workers : ℕ := 900

/-- The number of workers in Workshop B -/
def workshop_b_workers : ℕ := 300

/-- The total sample size -/
def total_sample : ℕ := 45

/-- The number of people sampled from Workshop A -/
def sample_a : ℕ := 20

/-- The number of people sampled from Workshop C -/
def sample_c : ℕ := 10

/-- The number of people sampled from Workshop B -/
def sample_b : ℕ := total_sample - sample_a - sample_c

theorem factory_workers_count :
  (sample_b : ℚ) / workshop_b_workers = (total_sample : ℚ) / total_workers :=
by sorry

end NUMINAMATH_CALUDE_factory_workers_count_l3184_318463


namespace NUMINAMATH_CALUDE_valid_arrangement_5_cubes_valid_arrangement_6_cubes_l3184_318488

/-- A cube in 3D space --/
structure Cube where
  position : ℝ × ℝ × ℝ

/-- An arrangement of cubes in 3D space --/
def Arrangement (n : ℕ) := Fin n → Cube

/-- Predicate to check if two cubes share a polygonal face --/
def SharesFace (c1 c2 : Cube) : Prop := sorry

/-- Predicate to check if an arrangement is valid (each cube shares a face with every other) --/
def ValidArrangement (arr : Arrangement n) : Prop :=
  ∀ i j, i ≠ j → SharesFace (arr i) (arr j)

/-- Theorem stating the existence of a valid arrangement for 5 cubes --/
theorem valid_arrangement_5_cubes : ∃ (arr : Arrangement 5), ValidArrangement arr := sorry

/-- Theorem stating the existence of a valid arrangement for 6 cubes --/
theorem valid_arrangement_6_cubes : ∃ (arr : Arrangement 6), ValidArrangement arr := sorry

end NUMINAMATH_CALUDE_valid_arrangement_5_cubes_valid_arrangement_6_cubes_l3184_318488


namespace NUMINAMATH_CALUDE_smallest_positive_angle_2014_l3184_318426

def same_terminal_side (a b : ℝ) : Prop :=
  ∃ k : ℤ, a = b + k * 360

theorem smallest_positive_angle_2014 :
  ∃! θ : ℝ, 0 ≤ θ ∧ θ < 360 ∧ same_terminal_side θ (-2014) ∧ θ = 146 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_2014_l3184_318426


namespace NUMINAMATH_CALUDE_bird_sanctuary_geese_percentage_l3184_318487

theorem bird_sanctuary_geese_percentage :
  let total_percentage : ℚ := 100
  let geese_percentage : ℚ := 40
  let swan_percentage : ℚ := 20
  let heron_percentage : ℚ := 15
  let duck_percentage : ℚ := 25
  let non_duck_percentage : ℚ := total_percentage - duck_percentage
  geese_percentage / non_duck_percentage * 100 = 53 + 1/3 :=
by sorry

end NUMINAMATH_CALUDE_bird_sanctuary_geese_percentage_l3184_318487


namespace NUMINAMATH_CALUDE_tan_alpha_beta_eq_three_tan_alpha_l3184_318437

/-- Given that 2 sin β = sin(2α + β), prove that tan(α + β) = 3 tan α -/
theorem tan_alpha_beta_eq_three_tan_alpha (α β : ℝ) 
  (h : 2 * Real.sin β = Real.sin (2 * α + β)) :
  Real.tan (α + β) = 3 * Real.tan α := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_beta_eq_three_tan_alpha_l3184_318437


namespace NUMINAMATH_CALUDE_a_is_perfect_square_l3184_318461

def a : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => 7 * a (n + 1) - a n - 2

theorem a_is_perfect_square : ∀ n : ℕ, ∃ k : ℤ, a n = k^2 := by
  sorry

end NUMINAMATH_CALUDE_a_is_perfect_square_l3184_318461


namespace NUMINAMATH_CALUDE_max_stores_visited_is_four_l3184_318442

/-- Represents the shopping scenario in the town -/
structure ShoppingScenario where
  total_visits : Nat
  unique_shoppers : Nat
  two_store_visitors : Nat
  stores_in_town : Nat

/-- Calculates the maximum number of stores visited by any single person -/
def max_stores_visited (scenario : ShoppingScenario) : Nat :=
  let remaining_visits := scenario.total_visits - 2 * scenario.two_store_visitors
  let remaining_shoppers := scenario.unique_shoppers - scenario.two_store_visitors
  let extra_visits := remaining_visits - remaining_shoppers
  1 + extra_visits

/-- Theorem stating the maximum number of stores visited by any single person -/
theorem max_stores_visited_is_four (scenario : ShoppingScenario) : 
  scenario.total_visits = 23 →
  scenario.unique_shoppers = 12 →
  scenario.two_store_visitors = 8 →
  scenario.stores_in_town = 8 →
  max_stores_visited scenario = 4 := by
  sorry

#eval max_stores_visited ⟨23, 12, 8, 8⟩

end NUMINAMATH_CALUDE_max_stores_visited_is_four_l3184_318442


namespace NUMINAMATH_CALUDE_smallest_n_for_sqrt_inequality_l3184_318425

theorem smallest_n_for_sqrt_inequality : 
  ∃ (n : ℕ), n > 0 ∧ Real.sqrt n - Real.sqrt (n - 1) < 0.02 ∧ 
  ∀ (m : ℕ), m > 0 ∧ m < n → Real.sqrt m - Real.sqrt (m - 1) ≥ 0.02 :=
by
  use 626
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_sqrt_inequality_l3184_318425


namespace NUMINAMATH_CALUDE_collinear_points_b_value_l3184_318466

theorem collinear_points_b_value :
  ∀ b : ℚ,
  let p1 : ℚ × ℚ := (4, -6)
  let p2 : ℚ × ℚ := (b + 3, 4)
  let p3 : ℚ × ℚ := (3*b + 4, 3)
  (p1.2 - p2.2) * (p2.1 - p3.1) = (p2.2 - p3.2) * (p1.1 - p2.1) →
  b = -3/7 := by
sorry

end NUMINAMATH_CALUDE_collinear_points_b_value_l3184_318466


namespace NUMINAMATH_CALUDE_triangles_are_similar_l3184_318459

/-- Two triangles are similar if the ratios of their corresponding sides are equal -/
def are_similar (a b c d e f : ℝ) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ b = k * e ∧ c = k * f ∧ a = k * d

/-- Triangle ABC has sides of length 1, √2, and √5 -/
def triangle_ABC (a b c : ℝ) : Prop :=
  a = 1 ∧ b = Real.sqrt 2 ∧ c = Real.sqrt 5

/-- Triangle DEF has sides of length √3, √6, and √15 -/
def triangle_DEF (d e f : ℝ) : Prop :=
  d = Real.sqrt 3 ∧ e = Real.sqrt 6 ∧ f = Real.sqrt 15

theorem triangles_are_similar :
  ∀ (a b c d e f : ℝ),
    triangle_ABC a b c →
    triangle_DEF d e f →
    are_similar a b c d e f :=
by sorry

end NUMINAMATH_CALUDE_triangles_are_similar_l3184_318459


namespace NUMINAMATH_CALUDE_no_solution_inequalities_l3184_318400

theorem no_solution_inequalities (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ¬∃ x : ℝ, x > a ∧ x < -b := by
sorry

end NUMINAMATH_CALUDE_no_solution_inequalities_l3184_318400


namespace NUMINAMATH_CALUDE_A_subset_B_l3184_318431

-- Define set A
def A : Set ℤ := {x | ∃ k : ℕ, x = 7 * k + 3}

-- Define set B
def B : Set ℤ := {x | ∃ k : ℤ, x = 7 * k - 4}

-- Theorem stating A is a subset of B
theorem A_subset_B : A ⊆ B := by
  sorry

end NUMINAMATH_CALUDE_A_subset_B_l3184_318431


namespace NUMINAMATH_CALUDE_johns_total_payment_l3184_318420

/-- Calculates the total amount John paid for his dog's vet appointments and insurance -/
def total_payment (num_appointments : ℕ) (appointment_cost : ℚ) (insurance_cost : ℚ) (insurance_coverage : ℚ) : ℚ :=
  let first_appointment_cost := appointment_cost
  let insurance_payment := insurance_cost
  let subsequent_appointments_cost := appointment_cost * (num_appointments - 1 : ℚ)
  let covered_amount := subsequent_appointments_cost * insurance_coverage
  let out_of_pocket := subsequent_appointments_cost - covered_amount
  first_appointment_cost + insurance_payment + out_of_pocket

/-- Theorem stating that John's total payment for his dog's vet appointments and insurance is $660 -/
theorem johns_total_payment :
  total_payment 3 400 100 0.8 = 660 := by
  sorry

end NUMINAMATH_CALUDE_johns_total_payment_l3184_318420


namespace NUMINAMATH_CALUDE_percentage_loss_l3184_318423

theorem percentage_loss (cost_price selling_price : ℝ) 
  (h1 : cost_price = 1400)
  (h2 : selling_price = 1050) : 
  (cost_price - selling_price) / cost_price * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_loss_l3184_318423


namespace NUMINAMATH_CALUDE_fish_dog_lifespan_difference_l3184_318434

/-- The difference between a fish's lifespan and a dog's lifespan is 2 years -/
theorem fish_dog_lifespan_difference :
  let hamster_lifespan : ℝ := 2.5
  let dog_lifespan : ℝ := 4 * hamster_lifespan
  let fish_lifespan : ℝ := 12
  fish_lifespan - dog_lifespan = 2 := by
  sorry

end NUMINAMATH_CALUDE_fish_dog_lifespan_difference_l3184_318434


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3184_318428

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 - 3*x > 0) ↔ (∃ x : ℝ, x^3 - 3*x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3184_318428


namespace NUMINAMATH_CALUDE_factory_price_decrease_and_sales_optimization_l3184_318457

/-- The average decrease rate in factory price over two years -/
def average_decrease_rate : ℝ := 0.1

/-- The price reduction that maximizes sales while maintaining the target profit -/
def price_reduction : ℝ := 15

/-- The initial factory price in 2019 -/
def initial_price : ℝ := 200

/-- The final factory price in 2021 -/
def final_price : ℝ := 162

/-- The initial number of pieces sold per day at the original price -/
def initial_sales : ℝ := 20

/-- The increase in sales for every 5 yuan reduction in price -/
def sales_increase_rate : ℝ := 2

/-- The target daily profit after price reduction -/
def target_profit : ℝ := 1150

theorem factory_price_decrease_and_sales_optimization :
  (initial_price * (1 - average_decrease_rate)^2 = final_price) ∧
  ((initial_price - final_price - price_reduction) * 
   (initial_sales + sales_increase_rate * price_reduction) = target_profit) ∧
  (∀ m : ℝ, m > price_reduction → 
   ((initial_price - final_price - m) * (initial_sales + sales_increase_rate * m) ≠ target_profit)) :=
by sorry

end NUMINAMATH_CALUDE_factory_price_decrease_and_sales_optimization_l3184_318457


namespace NUMINAMATH_CALUDE_smallest_valid_number_l3184_318427

def is_valid (n : ℕ) : Prop :=
  ∃ (k : ℕ), 
    n % 10 = 2 ∧ 
    n * 2 = k * 10 + 2

theorem smallest_valid_number : 
  (∀ m : ℕ, m < 105263157894736842 → ¬(is_valid m)) ∧ 
  is_valid 105263157894736842 :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l3184_318427


namespace NUMINAMATH_CALUDE_five_students_three_villages_l3184_318406

/-- The number of ways to assign n students to m villages with at least one student per village -/
def assignmentCount (n m : ℕ) : ℕ := sorry

/-- The number of ways to assign 5 students to 3 villages with at least one student per village -/
theorem five_students_three_villages : assignmentCount 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_five_students_three_villages_l3184_318406


namespace NUMINAMATH_CALUDE_sum_of_fractions_geq_one_l3184_318415

theorem sum_of_fractions_geq_one (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (a + 2*b) + b / (b + 2*c) + c / (c + 2*a) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_geq_one_l3184_318415


namespace NUMINAMATH_CALUDE_sets_subset_theorem_l3184_318409

-- Define the sets P₁, P₂, Q₁, and Q₂
def P₁ (a : ℝ) : Set ℝ := {x | x^2 + a*x + 1 > 0}
def P₂ (a : ℝ) : Set ℝ := {x | x^2 + a*x + 2 > 0}
def Q₁ (b : ℝ) : Set ℝ := {x | x^2 + x + b > 0}
def Q₂ (b : ℝ) : Set ℝ := {x | x^2 + 2*x + b > 0}

-- State the theorem
theorem sets_subset_theorem :
  (∀ a : ℝ, P₁ a ⊆ P₂ a) ∧ (∃ b : ℝ, Q₁ b ⊆ Q₂ b) := by
  sorry


end NUMINAMATH_CALUDE_sets_subset_theorem_l3184_318409


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3184_318478

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | -3 < x ∧ x < 0}

-- Define set B
def B : Set ℝ := {x : ℝ | x < -1}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ (U \ B) = {x : ℝ | -1 ≤ x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3184_318478


namespace NUMINAMATH_CALUDE_twenty_player_tournament_games_l3184_318407

/-- Calculates the number of games in a chess tournament --/
def chess_tournament_games (n : ℕ) : ℕ :=
  n * (n - 1)

/-- Theorem: In a chess tournament with 20 players, where each player plays twice with every other player, 
    the total number of games played is 760. --/
theorem twenty_player_tournament_games : 
  chess_tournament_games 20 * 2 = 760 := by
  sorry

end NUMINAMATH_CALUDE_twenty_player_tournament_games_l3184_318407


namespace NUMINAMATH_CALUDE_abs_minus_2010_l3184_318496

theorem abs_minus_2010 : |(-2010 : ℤ)| = 2010 := by
  sorry

end NUMINAMATH_CALUDE_abs_minus_2010_l3184_318496


namespace NUMINAMATH_CALUDE_cannot_tile_with_sphinxes_l3184_318455

/-- Represents a sphinx shape -/
structure Sphinx :=
  (angles : Finset ℝ)
  (smallTriangles : ℕ)

/-- Represents the large equilateral triangle -/
structure LargeTriangle :=
  (sideLength : ℕ)
  (smallTriangles : ℕ)
  (grayTriangles : ℕ)
  (whiteTriangles : ℕ)

/-- Theorem stating the impossibility of tiling the large triangle with sphinxes -/
theorem cannot_tile_with_sphinxes (s : Sphinx) (t : LargeTriangle) : 
  s.angles = {60, 120, 240} ∧ 
  s.smallTriangles = 6 ∧
  t.sideLength = 6 ∧
  t.smallTriangles = 21 ∧
  t.grayTriangles = 15 ∧
  t.whiteTriangles = 21 →
  ¬ ∃ (n : ℕ), n * s.smallTriangles = t.smallTriangles :=
by sorry

end NUMINAMATH_CALUDE_cannot_tile_with_sphinxes_l3184_318455


namespace NUMINAMATH_CALUDE_center_equidistant_from_hexagon_vertices_l3184_318472

/-- Represents a nickel coin -/
structure Nickel where
  diameter : ℝ
  diameter_pos : diameter > 0

/-- Represents a circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- Represents a regular hexagon -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ

/-- States that a circle's diameter is equal to a nickel's diameter -/
def circle_diameter_eq_nickel (c : Circle) (n : Nickel) : Prop :=
  c.radius * 2 = n.diameter

/-- States that a hexagon is inscribed in a circle -/
def hexagon_inscribed_in_circle (h : RegularHexagon) (c : Circle) : Prop :=
  ∀ i : Fin 6, dist c.center (h.vertices i) = c.radius

/-- States that a hexagon can be constructed using three nickels -/
def hexagon_constructible_with_nickels (h : RegularHexagon) (n : Nickel) : Prop :=
  ∀ i : Fin 6, ∀ j : Fin 6, i ≠ j → dist (h.vertices i) (h.vertices j) = n.diameter

/-- The main theorem -/
theorem center_equidistant_from_hexagon_vertices
  (c : Circle) (n : Nickel) (h : RegularHexagon)
  (h1 : circle_diameter_eq_nickel c n)
  (h2 : hexagon_inscribed_in_circle h c)
  (h3 : hexagon_constructible_with_nickels h n) :
  ∀ i j : Fin 6, dist c.center (h.vertices i) = dist c.center (h.vertices j) := by
  sorry

end NUMINAMATH_CALUDE_center_equidistant_from_hexagon_vertices_l3184_318472


namespace NUMINAMATH_CALUDE_divides_product_l3184_318435

theorem divides_product (a b c d : ℤ) (h1 : a ∣ b) (h2 : c ∣ d) : a * c ∣ b * d := by
  sorry

end NUMINAMATH_CALUDE_divides_product_l3184_318435
