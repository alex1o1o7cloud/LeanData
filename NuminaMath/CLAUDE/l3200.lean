import Mathlib

namespace NUMINAMATH_CALUDE_tan_alpha_value_l3200_320087

theorem tan_alpha_value (α β : Real) 
  (h1 : Real.tan (α - β) = 2) 
  (h2 : Real.tan β = 3) : 
  Real.tan α = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3200_320087


namespace NUMINAMATH_CALUDE_sams_highlighter_boxes_l3200_320060

/-- Represents the problem of determining how many boxes Sam bought --/
theorem sams_highlighter_boxes :
  ∀ (B : ℕ),
  (∃ (package_revenue separate_revenue cost : ℚ),
    -- Revenue from packages
    package_revenue = 75 ∧
    -- Revenue from separate highlighters
    separate_revenue = (B - 5) * 20 ∧
    -- Total cost
    cost = B * 10 ∧
    -- Total profit
    package_revenue + separate_revenue - cost = 115) →
  B = 14 :=
by sorry

end NUMINAMATH_CALUDE_sams_highlighter_boxes_l3200_320060


namespace NUMINAMATH_CALUDE_min_value_sum_product_l3200_320051

theorem min_value_sum_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z) * ((x + y)⁻¹ + (x + z)⁻¹ + (y + z)⁻¹) ≥ (9 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_product_l3200_320051


namespace NUMINAMATH_CALUDE_evaluate_expression_l3200_320006

theorem evaluate_expression : (((3^2 : ℚ) - 2^3 + 7^1 - 1 + 4^2)⁻¹ * (5/6)) = 5/138 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3200_320006


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l3200_320021

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 3*x + 2) / Real.log (1/3)

theorem f_strictly_increasing :
  ∀ x y, x < y ∧ y < 1 → f x < f y :=
sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l3200_320021


namespace NUMINAMATH_CALUDE_prob_not_yellow_is_seven_tenths_l3200_320074

/-- Represents the contents of a bag of jelly beans -/
structure JellyBeanBag where
  red : ℕ
  green : ℕ
  yellow : ℕ
  blue : ℕ

/-- Calculates the probability of selecting a non-yellow jelly bean -/
def probNotYellow (bag : JellyBeanBag) : ℚ :=
  let total := bag.red + bag.green + bag.yellow + bag.blue
  let notYellow := bag.red + bag.green + bag.blue
  notYellow / total

/-- Theorem: The probability of selecting a non-yellow jelly bean from a bag
    containing 4 red, 7 green, 9 yellow, and 10 blue jelly beans is 7/10 -/
theorem prob_not_yellow_is_seven_tenths :
  probNotYellow { red := 4, green := 7, yellow := 9, blue := 10 } = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_yellow_is_seven_tenths_l3200_320074


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3200_320084

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence with a_3 = 2 and a_6 = 5, prove a_9 = 8 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : arithmetic_sequence a) 
    (h_3 : a 3 = 2) 
    (h_6 : a 6 = 5) : 
  a 9 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3200_320084


namespace NUMINAMATH_CALUDE_negation_of_proposition_sin_inequality_negation_l3200_320028

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬ p x) :=
by sorry

theorem sin_inequality_negation :
  (¬ ∀ x : ℝ, x > Real.sin x) ↔ (∃ x : ℝ, x ≤ Real.sin x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_sin_inequality_negation_l3200_320028


namespace NUMINAMATH_CALUDE_magazine_choice_count_l3200_320010

theorem magazine_choice_count : 
  let science_count : Nat := 4
  let digest_count : Nat := 3
  let entertainment_count : Nat := 2
  science_count + digest_count + entertainment_count = 9 :=
by sorry

end NUMINAMATH_CALUDE_magazine_choice_count_l3200_320010


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3200_320083

theorem complex_equation_solution (z : ℂ) : z * Complex.I = Complex.I - 1 → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3200_320083


namespace NUMINAMATH_CALUDE_fair_hair_percentage_l3200_320045

theorem fair_hair_percentage 
  (total_employees : ℝ) 
  (women_fair_hair_percentage : ℝ) 
  (women_percentage_of_fair_hair : ℝ) 
  (h1 : women_fair_hair_percentage = 30) 
  (h2 : women_percentage_of_fair_hair = 40) 
  (h3 : total_employees > 0) :
  (women_fair_hair_percentage * total_employees / 100) / 
  (women_percentage_of_fair_hair / 100) / 
  total_employees * 100 = 75 := by
sorry

end NUMINAMATH_CALUDE_fair_hair_percentage_l3200_320045


namespace NUMINAMATH_CALUDE_binomial_expansion_theorem_l3200_320089

theorem binomial_expansion_theorem (n : ℕ) (a b : ℝ) (k : ℕ+) : 
  n ≥ 2 →
  a ≠ 0 →
  b ≠ 0 →
  a = b + k →
  2 * n * b^2 + (4 * n * (n - 1) * (n - 2) / 3) * k^2 = 0 →
  n = 3 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_theorem_l3200_320089


namespace NUMINAMATH_CALUDE_intersection_M_N_l3200_320085

def M : Set ℝ := {y | ∃ x : ℝ, y = x^2}
def N : Set ℝ := {y | ∃ x : ℝ, y = 2 - |x|}

theorem intersection_M_N : M ∩ N = {y : ℝ | 0 ≤ y ∧ y ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3200_320085


namespace NUMINAMATH_CALUDE_rectangle_area_from_quadratic_roots_l3200_320011

theorem rectangle_area_from_quadratic_roots : 
  ∀ (length width : ℝ),
  (2 * length^2 - 11 * length + 5 = 0) →
  (2 * width^2 - 11 * width + 5 = 0) →
  (length * width = 5 / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_from_quadratic_roots_l3200_320011


namespace NUMINAMATH_CALUDE_no_solutions_condition_l3200_320068

theorem no_solutions_condition (b : ℝ) : 
  (∀ a x : ℝ, a > 1 → a^(2-2*x^2) + (b+4)*a^(1-x^2) + 3*b + 4 ≠ 0) ↔ 
  (b ∈ Set.Ioc (-4/3) 0 ∪ Set.Ici 4) := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_condition_l3200_320068


namespace NUMINAMATH_CALUDE_multiple_p_values_exist_l3200_320032

theorem multiple_p_values_exist : ∃ p₁ p₂ : ℝ, 
  0 < p₁ ∧ p₁ < 1 ∧ 
  0 < p₂ ∧ p₂ < 1 ∧ 
  p₁ ≠ p₂ ∧
  (Nat.choose 5 3 : ℝ) * p₁^3 * (1 - p₁)^2 = 144/625 ∧
  (Nat.choose 5 3 : ℝ) * p₂^3 * (1 - p₂)^2 = 144/625 :=
by sorry

end NUMINAMATH_CALUDE_multiple_p_values_exist_l3200_320032


namespace NUMINAMATH_CALUDE_points_per_round_l3200_320022

theorem points_per_round (total_rounds : ℕ) (total_points : ℕ) 
  (h1 : total_rounds = 177)
  (h2 : total_points = 8142) : 
  total_points / total_rounds = 46 := by
  sorry

end NUMINAMATH_CALUDE_points_per_round_l3200_320022


namespace NUMINAMATH_CALUDE_piece_length_in_cm_l3200_320024

-- Define the length of the rod in meters
def rod_length : ℝ := 25.5

-- Define the number of pieces that can be cut from the rod
def num_pieces : ℕ := 30

-- Define the conversion factor from meters to centimeters
def meters_to_cm : ℝ := 100

-- Theorem statement
theorem piece_length_in_cm : 
  (rod_length / num_pieces) * meters_to_cm = 85 := by
  sorry

end NUMINAMATH_CALUDE_piece_length_in_cm_l3200_320024


namespace NUMINAMATH_CALUDE_modulus_of_z_l3200_320043

theorem modulus_of_z (z : ℂ) (h : z * Complex.I = 2 + Complex.I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3200_320043


namespace NUMINAMATH_CALUDE_sichuan_selected_count_l3200_320027

/-- Represents the number of students selected from Sichuan University in a stratified sampling -/
def sichuan_selected (total_students : ℕ) (sichuan_students : ℕ) (other_students : ℕ) (selected_students : ℕ) : ℕ :=
  (selected_students * sichuan_students) / (sichuan_students + other_students)

/-- Theorem stating that 10 students from Sichuan University are selected in the given scenario -/
theorem sichuan_selected_count :
  sichuan_selected 40 25 15 16 = 10 := by
  sorry

#eval sichuan_selected 40 25 15 16

end NUMINAMATH_CALUDE_sichuan_selected_count_l3200_320027


namespace NUMINAMATH_CALUDE_prop_or_quadratic_always_positive_parallel_iff_l3200_320001

-- Define propositions p and q
def p : Prop := ∀ x : ℚ, (x : ℝ) = x
def q : Prop := ∀ x : ℝ, x > 0 → Real.log x < 0

-- Statement 1
theorem prop_or : p ∨ q := by sorry

-- Statement 2
theorem quadratic_always_positive : ∀ x : ℝ, x^2 + x + 2 > 0 := by sorry

-- Define the lines
def line1 (a : ℝ) (x y : ℝ) : Prop := x + a * y + 6 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := (a - 2) * x + 3 * y + 2 * a = 0

-- Define parallel lines
def parallel (a : ℝ) : Prop := ∀ x y : ℝ, line1 a x y ↔ ∃ k : ℝ, line2 a (x + k) (y + k)

-- Statement 3
theorem parallel_iff : ∀ a : ℝ, parallel a ↔ a = -1 := by sorry

end NUMINAMATH_CALUDE_prop_or_quadratic_always_positive_parallel_iff_l3200_320001


namespace NUMINAMATH_CALUDE_basketball_sales_solution_l3200_320099

/-- Represents the cost and sales information for basketballs --/
structure BasketballSales where
  cost_a : ℝ  -- Cost of A brand basketball
  cost_b : ℝ  -- Cost of B brand basketball
  price_a : ℝ  -- Original selling price of A brand basketball
  markup_b : ℝ  -- Markup percentage for B brand basketball
  discount_a : ℝ  -- Discount percentage for A brand basketball

/-- Theorem stating the solution to the basketball sales problem --/
theorem basketball_sales_solution (s : BasketballSales) : 
  (40 * s.cost_a + 40 * s.cost_b = 7200) →
  (50 * s.cost_a + 30 * s.cost_b = 7400) →
  (s.price_a = 140) →
  (s.markup_b = 0.3) →
  (40 * (s.price_a - s.cost_a) + 10 * (s.price_a * (1 - s.discount_a / 100) - s.cost_a) + 
   30 * s.cost_b * s.markup_b = 2440) →
  (s.cost_a = 100 ∧ s.cost_b = 80 ∧ s.discount_a = 8) := by
  sorry


end NUMINAMATH_CALUDE_basketball_sales_solution_l3200_320099


namespace NUMINAMATH_CALUDE_nickel_count_equality_l3200_320080

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The number of quarters in the first group -/
def quarters1 : ℕ := 15

/-- The number of nickels in the first group -/
def nickels1 : ℕ := 20

/-- The number of quarters in the second group -/
def quarters2 : ℕ := 10

theorem nickel_count_equality (n : ℕ) : 
  quarters1 * quarter_value + nickels1 * nickel_value = 
  quarters2 * quarter_value + n * nickel_value → n = 45 := by
sorry

end NUMINAMATH_CALUDE_nickel_count_equality_l3200_320080


namespace NUMINAMATH_CALUDE_complement_of_hit_at_least_once_l3200_320058

-- Define the sample space
def Ω : Type := Bool × Bool

-- Define the event of hitting the target at least once
def hit_at_least_once (ω : Ω) : Prop :=
  ω.1 ∨ ω.2

-- Define the event of missing the target both times
def miss_both_times (ω : Ω) : Prop :=
  ¬ω.1 ∧ ¬ω.2

-- Theorem stating that missing both times is the complement of hitting at least once
theorem complement_of_hit_at_least_once :
  ∀ ω : Ω, ¬(hit_at_least_once ω) ↔ miss_both_times ω :=
by sorry

end NUMINAMATH_CALUDE_complement_of_hit_at_least_once_l3200_320058


namespace NUMINAMATH_CALUDE_roden_gold_fish_l3200_320092

/-- The number of fish Roden bought in total -/
def total_fish : ℕ := 22

/-- The number of blue fish Roden bought -/
def blue_fish : ℕ := 7

/-- The number of gold fish Roden bought -/
def gold_fish : ℕ := total_fish - blue_fish

theorem roden_gold_fish : gold_fish = 15 := by
  sorry

end NUMINAMATH_CALUDE_roden_gold_fish_l3200_320092


namespace NUMINAMATH_CALUDE_number_difference_l3200_320064

theorem number_difference (L S : ℕ) (h1 : L = 1631) (h2 : L = 6 * S + 35) : L - S = 1365 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l3200_320064


namespace NUMINAMATH_CALUDE_profit_without_discount_l3200_320014

theorem profit_without_discount (discount_percent : ℝ) (profit_with_discount_percent : ℝ) :
  discount_percent = 5 →
  profit_with_discount_percent = 20.65 →
  let cost_price := 100
  let selling_price_with_discount := cost_price * (1 - discount_percent / 100)
  let profit := cost_price * profit_with_discount_percent / 100
  let selling_price_without_discount := cost_price + profit
  profit / cost_price * 100 = 20.65 :=
by sorry

end NUMINAMATH_CALUDE_profit_without_discount_l3200_320014


namespace NUMINAMATH_CALUDE_valentino_farm_birds_l3200_320065

/-- The number of birds on Mr. Valentino's farm -/
def total_birds (chickens ducks turkeys : ℕ) : ℕ := chickens + ducks + turkeys

/-- Theorem stating the total number of birds on Mr. Valentino's farm -/
theorem valentino_farm_birds :
  ∃ (chickens ducks turkeys : ℕ),
    chickens = 200 ∧
    ducks = 2 * chickens ∧
    turkeys = 3 * ducks ∧
    total_birds chickens ducks turkeys = 1800 := by
  sorry

end NUMINAMATH_CALUDE_valentino_farm_birds_l3200_320065


namespace NUMINAMATH_CALUDE_reflection_theorem_l3200_320086

/-- Represents a 3D vector -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a reflection plane -/
inductive ReflectionPlane
  | XY
  | YZ
  | ZX

/-- Reflects a vector across a given plane -/
def reflect (v : Vector3D) (plane : ReflectionPlane) : Vector3D :=
  match plane with
  | ReflectionPlane.XY => ⟨v.x, v.y, -v.z⟩
  | ReflectionPlane.YZ => ⟨-v.x, v.y, v.z⟩
  | ReflectionPlane.ZX => ⟨v.x, -v.y, v.z⟩

/-- Reflects a vector across all three coordinate planes -/
def reflectAll (v : Vector3D) : Vector3D :=
  reflect (reflect (reflect v ReflectionPlane.XY) ReflectionPlane.YZ) ReflectionPlane.ZX

theorem reflection_theorem (v : Vector3D) :
  reflectAll v = Vector3D.mk (-v.x) (-v.y) (-v.z) := by
  sorry

#check reflection_theorem

end NUMINAMATH_CALUDE_reflection_theorem_l3200_320086


namespace NUMINAMATH_CALUDE_sandwich_availability_l3200_320075

theorem sandwich_availability (total : ℕ) (sold_out : ℕ) (available : ℕ) 
  (h1 : total = 50) 
  (h2 : sold_out = 33) 
  (h3 : available = total - sold_out) : 
  available = 17 := by
sorry

end NUMINAMATH_CALUDE_sandwich_availability_l3200_320075


namespace NUMINAMATH_CALUDE_daisy_count_proof_l3200_320013

theorem daisy_count_proof (white : ℕ) (pink : ℕ) (red : ℕ) 
  (h1 : white = 6)
  (h2 : pink = 9 * white)
  (h3 : red = 4 * pink - 3) :
  white + pink + red = 273 := by
  sorry

end NUMINAMATH_CALUDE_daisy_count_proof_l3200_320013


namespace NUMINAMATH_CALUDE_xiaoming_age_is_10_l3200_320062

/-- Xiao Ming's age this year -/
def xiaoming_age : ℕ := sorry

/-- Father's age this year -/
def father_age : ℕ := 4 * xiaoming_age

/-- The sum of their ages 25 years later -/
def sum_ages_25_years_later : ℕ := (xiaoming_age + 25) + (father_age + 25)

theorem xiaoming_age_is_10 :
  xiaoming_age = 10 ∧ father_age = 4 * xiaoming_age ∧ sum_ages_25_years_later = 100 :=
sorry

end NUMINAMATH_CALUDE_xiaoming_age_is_10_l3200_320062


namespace NUMINAMATH_CALUDE_prob_one_female_is_half_l3200_320078

/-- Represents the composition of the extracurricular interest group -/
structure InterestGroup :=
  (male_count : Nat)
  (female_count : Nat)

/-- Calculates the probability of selecting exactly one female student
    from two selections in the interest group -/
def prob_one_female (group : InterestGroup) : Real :=
  let total := group.male_count + group.female_count
  let prob_first_female := group.female_count / total
  let prob_second_male := group.male_count / (total - 1)
  let prob_first_male := group.male_count / total
  let prob_second_female := group.female_count / (total - 1)
  prob_first_female * prob_second_male + prob_first_male * prob_second_female

/-- Theorem: The probability of selecting exactly one female student
    from two selections in a group of 3 males and 1 female is 0.5 -/
theorem prob_one_female_is_half :
  let group := InterestGroup.mk 3 1
  prob_one_female group = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_one_female_is_half_l3200_320078


namespace NUMINAMATH_CALUDE_marker_selection_ways_l3200_320081

theorem marker_selection_ways :
  let n : ℕ := 15  -- Total number of markers
  let k : ℕ := 5   -- Number of markers to be selected
  Nat.choose n k = 3003 :=
by sorry

end NUMINAMATH_CALUDE_marker_selection_ways_l3200_320081


namespace NUMINAMATH_CALUDE_james_diet_result_l3200_320039

/-- Represents James' food intake and exercise routine --/
structure JamesDiet where
  cheezitBags : ℕ
  cheezitOuncesPerBag : ℕ
  cheezitCaloriesPerOunce : ℕ
  chocolateBars : ℕ
  chocolateBarCalories : ℕ
  popcornCalories : ℕ
  runningMinutes : ℕ
  runningCaloriesPerMinute : ℕ
  swimmingMinutes : ℕ
  swimmingCaloriesPerMinute : ℕ
  cyclingMinutes : ℕ
  cyclingCaloriesPerMinute : ℕ
  caloriesPerPound : ℕ

/-- Calculates the total calories consumed --/
def totalCaloriesConsumed (d : JamesDiet) : ℕ :=
  d.cheezitBags * d.cheezitOuncesPerBag * d.cheezitCaloriesPerOunce +
  d.chocolateBars * d.chocolateBarCalories +
  d.popcornCalories

/-- Calculates the total calories burned --/
def totalCaloriesBurned (d : JamesDiet) : ℕ :=
  d.runningMinutes * d.runningCaloriesPerMinute +
  d.swimmingMinutes * d.swimmingCaloriesPerMinute +
  d.cyclingMinutes * d.cyclingCaloriesPerMinute

/-- Calculates the excess calories --/
def excessCalories (d : JamesDiet) : ℤ :=
  (totalCaloriesConsumed d : ℤ) - (totalCaloriesBurned d : ℤ)

/-- Calculates the potential weight gain in pounds --/
def potentialWeightGain (d : JamesDiet) : ℚ :=
  (excessCalories d : ℚ) / d.caloriesPerPound

/-- Theorem stating James' excess calorie consumption and potential weight gain --/
theorem james_diet_result (d : JamesDiet) 
  (h1 : d.cheezitBags = 3)
  (h2 : d.cheezitOuncesPerBag = 2)
  (h3 : d.cheezitCaloriesPerOunce = 150)
  (h4 : d.chocolateBars = 2)
  (h5 : d.chocolateBarCalories = 250)
  (h6 : d.popcornCalories = 500)
  (h7 : d.runningMinutes = 40)
  (h8 : d.runningCaloriesPerMinute = 12)
  (h9 : d.swimmingMinutes = 30)
  (h10 : d.swimmingCaloriesPerMinute = 15)
  (h11 : d.cyclingMinutes = 20)
  (h12 : d.cyclingCaloriesPerMinute = 10)
  (h13 : d.caloriesPerPound = 3500) :
  excessCalories d = 770 ∧ potentialWeightGain d = 11/50 := by
  sorry

end NUMINAMATH_CALUDE_james_diet_result_l3200_320039


namespace NUMINAMATH_CALUDE_arithmetic_mean_first_n_odd_l3200_320047

/-- The sum of the first n odd positive integers -/
def sum_first_n_odd (n : ℕ) : ℕ := n^2

/-- The arithmetic mean of a list of numbers -/
def arithmetic_mean (sum : ℕ) (count : ℕ) : ℚ := sum / count

theorem arithmetic_mean_first_n_odd (n : ℕ) :
  arithmetic_mean (sum_first_n_odd n) n = n := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_first_n_odd_l3200_320047


namespace NUMINAMATH_CALUDE_pi_approximation_after_three_tiaoRi_l3200_320049

def tiaoRiMethod (a b c d : ℕ) : ℚ := (b + d) / (a + c)

theorem pi_approximation_after_three_tiaoRi :
  let initial_lower : ℚ := 31 / 10
  let initial_upper : ℚ := 49 / 15
  let first_upper : ℚ := tiaoRiMethod 10 31 15 49
  let second_upper : ℚ := tiaoRiMethod 15 47 5 16
  let third_upper : ℚ := tiaoRiMethod 15 47 5 16
  initial_lower < Real.pi ∧ Real.pi < initial_upper →
  third_upper = 63 / 20 :=
by sorry

end NUMINAMATH_CALUDE_pi_approximation_after_three_tiaoRi_l3200_320049


namespace NUMINAMATH_CALUDE_vector_equality_l3200_320046

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, -1)
def c : ℝ × ℝ := (-2, 4)

theorem vector_equality : c = a - 3 • b := by sorry

end NUMINAMATH_CALUDE_vector_equality_l3200_320046


namespace NUMINAMATH_CALUDE_range_of_a_l3200_320016

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*x + a ≥ 0}

-- State the theorem
theorem range_of_a (a : ℝ) : (1 ∉ A a) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3200_320016


namespace NUMINAMATH_CALUDE_sakshi_work_duration_l3200_320073

-- Define the efficiency ratio between Tanya and Sakshi
def efficiency_ratio : ℝ := 1.25

-- Define Tanya's work duration in days
def tanya_days : ℝ := 4

-- Theorem stating that Sakshi takes 5 days to complete the work
theorem sakshi_work_duration :
  efficiency_ratio * tanya_days = 5 := by
  sorry

end NUMINAMATH_CALUDE_sakshi_work_duration_l3200_320073


namespace NUMINAMATH_CALUDE_sum_of_w_and_z_l3200_320033

theorem sum_of_w_and_z (w x y z : ℤ) 
  (eq1 : w + x = 45)
  (eq2 : x + y = 51)
  (eq3 : y + z = 28) :
  w + z = 22 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_w_and_z_l3200_320033


namespace NUMINAMATH_CALUDE_roger_trays_first_table_l3200_320018

/-- The number of trays Roger can carry at a time -/
def trays_per_trip : ℕ := 4

/-- The number of trips Roger made -/
def num_trips : ℕ := 3

/-- The number of trays Roger picked up from the second table -/
def trays_from_second_table : ℕ := 2

/-- The number of trays Roger picked up from the first table -/
def trays_from_first_table : ℕ := trays_per_trip * num_trips - trays_from_second_table

theorem roger_trays_first_table :
  trays_from_first_table = 10 := by sorry

end NUMINAMATH_CALUDE_roger_trays_first_table_l3200_320018


namespace NUMINAMATH_CALUDE_max_books_borrowed_l3200_320019

theorem max_books_borrowed (total_students : ℕ) (no_books : ℕ) (two_books : ℕ) (three_books : ℕ) 
  (h1 : total_students = 50)
  (h2 : no_books = 10)
  (h3 : two_books = 18)
  (h4 : three_books = 8)
  (h5 : (total_students - no_books - two_books - three_books) * 7 ≤ 
        total_students * 4 - no_books * 0 - two_books * 2 - three_books * 3) :
  ∃ (max_books : ℕ), max_books = 49 ∧ 
    ∀ (student_books : ℕ), student_books ≤ max_books := by
  sorry

end NUMINAMATH_CALUDE_max_books_borrowed_l3200_320019


namespace NUMINAMATH_CALUDE_triangle_max_area_l3200_320041

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C,
    if (3+b)(sin A - sin B) = (c-b)sin C and a = 3,
    then the maximum area of triangle ABC is 9√3/4 -/
theorem triangle_max_area (a b c A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  (3 + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C ∧
  a = 3 →
  (∃ (S : ℝ), S = (1/2) * b * c * Real.sin A ∧ 
    ∀ (S' : ℝ), S' = (1/2) * b * c * Real.sin A → S' ≤ S) ∧
  (1/2) * b * c * Real.sin A ≤ (9 * Real.sqrt 3) / 4 :=
by sorry


end NUMINAMATH_CALUDE_triangle_max_area_l3200_320041


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3200_320098

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 10) :
  4 * Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) = 52 := by
  sorry

#check rhombus_perimeter

end NUMINAMATH_CALUDE_rhombus_perimeter_l3200_320098


namespace NUMINAMATH_CALUDE_intersection_of_perpendicular_lines_l3200_320030

/-- Given two lines in a plane, where one is perpendicular to the other and passes through a specific point, this theorem proves that their intersection point is as calculated. -/
theorem intersection_of_perpendicular_lines 
  (line1 : ℝ → ℝ)
  (line2 : ℝ → ℝ)
  (h1 : ∀ x, line1 x = -3 * x + 4)
  (h2 : ∀ x, line2 x = (1/3) * x - 1)
  (h3 : line2 3 = -2)
  (h4 : ∀ x y, line1 x = y → line2 x = y → x = 1.5 ∧ y = -0.5) :
  ∃ x y, line1 x = y ∧ line2 x = y ∧ x = 1.5 ∧ y = -0.5 := by
sorry


end NUMINAMATH_CALUDE_intersection_of_perpendicular_lines_l3200_320030


namespace NUMINAMATH_CALUDE_fraction_simplification_l3200_320069

theorem fraction_simplification (x : ℝ) (h : x = 5) :
  (x^6 - 2*x^3 + 1) / (x^3 - 1) = 124 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3200_320069


namespace NUMINAMATH_CALUDE_kirin_990_calculations_l3200_320048

-- Define the number of calculations per second
def calculations_per_second : ℝ := 10^11

-- Define the number of seconds
def seconds : ℝ := 2022

-- Theorem to prove
theorem kirin_990_calculations :
  calculations_per_second * seconds = 2.022 * 10^13 := by
  sorry

end NUMINAMATH_CALUDE_kirin_990_calculations_l3200_320048


namespace NUMINAMATH_CALUDE_smallest_divisors_sum_of_powers_l3200_320095

theorem smallest_divisors_sum_of_powers (n a b : ℕ) : 
  (a > 1) →
  (∀ k, 1 < k → k < a → ¬(k ∣ n)) →
  (a ∣ n) →
  (b > a) →
  (b ∣ n) →
  (∀ k, a < k → k < b → ¬(k ∣ n)) →
  (n = a^a + b^b) →
  (n = 260) :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisors_sum_of_powers_l3200_320095


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3200_320004

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 560) 
  (h2 : a*b + b*c + c*a = 8) : 
  a + b + c = 24 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3200_320004


namespace NUMINAMATH_CALUDE_triangle_median_inequalities_l3200_320038

/-- Given a triangle ABC with sides a, b, c and medians ma, mb, mc, prove two inequalities involving the medians. -/
theorem triangle_median_inequalities (a b c ma mb mc : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_pos_ma : ma > 0) (h_pos_mb : mb > 0) (h_pos_mc : mc > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_median_a : ma^2 = (2*b^2 + 2*c^2 - a^2) / 4)
  (h_median_b : mb^2 = (2*c^2 + 2*a^2 - b^2) / 4)
  (h_median_c : mc^2 = (2*a^2 + 2*b^2 - c^2) / 4) :
  (ma^2 / (b*c) + mb^2 / (c*a) + mc^2 / (a*b) ≥ 9/4) ∧
  ((mb^2 + mc^2 - ma^2) / (b*c) + (mc^2 + ma^2 - mb^2) / (c*a) + (ma^2 + mb^2 - mc^2) / (a*b) ≥ 9/4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_median_inequalities_l3200_320038


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3200_320029

/-- Given a parabola and an ellipse with the following properties:
  1) The parabola has the equation x^2 = 2py where p > 0
  2) The ellipse has the equation x^2/3 + y^2/4 = 1
  3) The focus of the parabola coincides with one of the vertices of the ellipse
This theorem states that the distance from the focus of the parabola to its directrix is 4. -/
theorem parabola_focus_directrix_distance (p : ℝ) 
  (h_p_pos : p > 0)
  (h_focus_coincides : ∃ (x y : ℝ), x^2/3 + y^2/4 = 1 ∧ x^2 = 2*p*y ∧ (x = 0 ∨ y = 2 ∨ y = -2)) :
  p = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3200_320029


namespace NUMINAMATH_CALUDE_units_digit_of_squares_l3200_320096

theorem units_digit_of_squares (n : ℕ) : 
  (n ≥ 10 ∧ n ≤ 99) → 
  (n % 10 = 2 ∨ n % 10 = 7) → 
  (n^2 % 10 ≠ 2 ∧ n^2 % 10 ≠ 6 ∧ n^2 % 10 ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_squares_l3200_320096


namespace NUMINAMATH_CALUDE_f_three_intersections_iff_a_in_range_l3200_320034

/-- The function f(x) = √(ax + 4) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (a * x + 4)

/-- The inverse function of f -/
noncomputable def f_inv (a : ℝ) (x : ℝ) : ℝ := (x^2 - 4) / a

/-- Predicate for f and f_inv having exactly three distinct intersection points -/
def has_three_intersections (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f a x₁ = f_inv a x₁ ∧ f a x₂ = f_inv a x₂ ∧ f a x₃ = f_inv a x₃ ∧
    ∀ x : ℝ, f a x = f_inv a x → x = x₁ ∨ x = x₂ ∨ x = x₃

theorem f_three_intersections_iff_a_in_range (a : ℝ) :
  a ≠ 0 → (has_three_intersections a ↔ -4 * Real.sqrt 3 / 3 < a ∧ a ≤ -2) :=
sorry

end NUMINAMATH_CALUDE_f_three_intersections_iff_a_in_range_l3200_320034


namespace NUMINAMATH_CALUDE_customers_who_tipped_l3200_320094

theorem customers_who_tipped (initial_customers : ℕ) (additional_customers : ℕ) (non_tipping_customers : ℕ)
  (h1 : initial_customers = 29)
  (h2 : additional_customers = 20)
  (h3 : non_tipping_customers = 34) :
  initial_customers + additional_customers - non_tipping_customers = 15 :=
by sorry

end NUMINAMATH_CALUDE_customers_who_tipped_l3200_320094


namespace NUMINAMATH_CALUDE_shaded_area_between_circles_l3200_320009

theorem shaded_area_between_circles (r₁ r₂ : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 5) : 
  let R := (r₁ + r₂) / 2
  π * R^2 - (π * r₁^2 + π * r₂^2) = 40 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_between_circles_l3200_320009


namespace NUMINAMATH_CALUDE_inradius_of_specific_triangle_l3200_320082

/-- Represents a triangle with side lengths a, b, c and incenter distance to one vertex d -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the inradius of a triangle -/
def inradius (t : Triangle) : ℝ := sorry

/-- Theorem stating that for the given triangle, the inradius is 0.8 * √14 -/
theorem inradius_of_specific_triangle :
  let t : Triangle := { a := 30, b := 36, c := 34, d := 18 }
  inradius t = 0.8 * Real.sqrt 14 := by sorry

end NUMINAMATH_CALUDE_inradius_of_specific_triangle_l3200_320082


namespace NUMINAMATH_CALUDE_circle_and_line_properties_l3200_320056

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x + 3)^2 + (y + 2)^2 = 25

-- Define the line m
def line_m (x y : ℝ) : Prop :=
  y = (5/12) * x + 43/12

-- Theorem statement
theorem circle_and_line_properties :
  -- Circle C passes through (0, 2) and (2, -2)
  circle_C 0 2 ∧ circle_C 2 (-2) ∧
  -- Center of C lies on x - y + 1 = 0
  (∃ t : ℝ, circle_C t (t + 1)) ∧
  -- Line m passes through (1, 4)
  line_m 1 4 ∧
  -- Chord length on C is 6
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    line_m x₁ y₁ ∧ line_m x₂ y₂ ∧
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 36) →
  -- The standard equation of C is correct
  (∀ x y : ℝ, (x + 3)^2 + (y + 2)^2 = 25 ↔ circle_C x y) ∧
  -- The slope-intercept equation of m is correct
  (∀ x y : ℝ, y = (5/12) * x + 43/12 ↔ line_m x y) :=
sorry

end NUMINAMATH_CALUDE_circle_and_line_properties_l3200_320056


namespace NUMINAMATH_CALUDE_like_terms_imply_exponents_l3200_320037

/-- Two algebraic terms are considered like terms if they have the same variables with the same exponents. -/
def are_like_terms (term1 term2 : ℝ → ℝ → ℝ) : Prop :=
  ∃ (c₁ c₂ : ℝ) (p q : ℕ), ∀ (a b : ℝ), term1 a b = c₁ * a^p * b^q ∧ term2 a b = c₂ * a^p * b^q

/-- The theorem states that if the given terms are like terms, then m = 4 and n = 2. -/
theorem like_terms_imply_exponents 
  (m n : ℕ) 
  (h : are_like_terms (λ a b => (1/3) * a^2 * b^m) (λ a b => (-1/2) * a^n * b^4)) : 
  m = 4 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_exponents_l3200_320037


namespace NUMINAMATH_CALUDE_sqrt_real_implies_x_leq_two_l3200_320005

theorem sqrt_real_implies_x_leq_two (x : ℝ) : (∃ y : ℝ, y * y = 2 - x) → x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_real_implies_x_leq_two_l3200_320005


namespace NUMINAMATH_CALUDE_floor_tiles_proof_l3200_320053

/-- Represents the number of tiles in a row given its position -/
def tiles_in_row (n : ℕ) : ℕ := 53 - 2 * (n - 1)

/-- Represents the total number of tiles in the first n rows -/
def total_tiles (n : ℕ) : ℕ := n * (tiles_in_row 1 + tiles_in_row n) / 2

/-- The number of rows in the floor -/
def num_rows : ℕ := 9

theorem floor_tiles_proof :
  (total_tiles num_rows = 405) ∧
  (∀ i : ℕ, i > 0 → i ≤ num_rows → tiles_in_row i > 0) :=
sorry

#eval num_rows

end NUMINAMATH_CALUDE_floor_tiles_proof_l3200_320053


namespace NUMINAMATH_CALUDE_price_decrease_calculation_l3200_320097

/-- The original price of an article before a price decrease -/
def original_price : ℝ := 421.05

/-- The percentage of the original price after the decrease -/
def percentage_after_decrease : ℝ := 0.76

/-- The price of the article after the decrease -/
def price_after_decrease : ℝ := 320

/-- Theorem stating that the original price is correct given the conditions -/
theorem price_decrease_calculation :
  price_after_decrease = percentage_after_decrease * original_price := by
  sorry

end NUMINAMATH_CALUDE_price_decrease_calculation_l3200_320097


namespace NUMINAMATH_CALUDE_adams_shopping_cost_l3200_320050

/-- Calculates the total cost of Adam's shopping given the specified conditions --/
def calculate_total_cost (sandwich_price : ℚ) (sandwich_count : ℕ) 
                         (chip_price : ℚ) (chip_count : ℕ) 
                         (water_price : ℚ) (water_count : ℕ) : ℚ :=
  let sandwich_cost := (sandwich_count - 1) * sandwich_price
  let chip_cost := chip_count * chip_price * (1 - 0.2)
  let water_cost := water_count * water_price * 1.05
  sandwich_cost + chip_cost + water_cost

/-- Theorem stating that Adam's total shopping cost is $31.75 --/
theorem adams_shopping_cost : 
  calculate_total_cost 4 5 3.5 3 1.75 4 = 31.75 := by
  sorry

end NUMINAMATH_CALUDE_adams_shopping_cost_l3200_320050


namespace NUMINAMATH_CALUDE_modulus_of_3_minus_i_l3200_320012

theorem modulus_of_3_minus_i :
  let z : ℂ := 3 - I
  Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_3_minus_i_l3200_320012


namespace NUMINAMATH_CALUDE_college_student_ticket_cost_l3200_320026

/-- Proves that the cost of a college student ticket is $4 given the specified conditions -/
theorem college_student_ticket_cost : 
  ∀ (total_visitors : ℕ) 
    (nyc_resident_ratio : ℚ) 
    (college_student_ratio : ℚ) 
    (total_revenue : ℚ),
  total_visitors = 200 →
  nyc_resident_ratio = 1/2 →
  college_student_ratio = 3/10 →
  total_revenue = 120 →
  (total_visitors : ℚ) * nyc_resident_ratio * college_student_ratio * 4 = total_revenue :=
by
  sorry


end NUMINAMATH_CALUDE_college_student_ticket_cost_l3200_320026


namespace NUMINAMATH_CALUDE_point_covering_theorem_l3200_320002

/-- A point in the unit square -/
structure Point where
  x : Real
  y : Real
  x_in_unit : 0 ≤ x ∧ x ≤ 1
  y_in_unit : 0 ≤ y ∧ y ≤ 1

/-- A rectangle inside the unit square with sides parallel to the square's sides -/
structure Rectangle where
  x1 : Real
  y1 : Real
  x2 : Real
  y2 : Real
  x1_le_x2 : x1 ≤ x2
  y1_le_y2 : y1 ≤ y2
  in_unit_square : 0 ≤ x1 ∧ x2 ≤ 1 ∧ 0 ≤ y1 ∧ y2 ≤ 1

/-- Check if a point is inside a rectangle -/
def pointInRectangle (p : Point) (r : Rectangle) : Prop :=
  r.x1 ≤ p.x ∧ p.x ≤ r.x2 ∧ r.y1 ≤ p.y ∧ p.y ≤ r.y2

/-- The area of a rectangle -/
def rectangleArea (r : Rectangle) : Real :=
  (r.x2 - r.x1) * (r.y2 - r.y1)

/-- The main theorem -/
theorem point_covering_theorem :
  ∃ (points : Finset Point),
    points.card = 1965 ∧
    ∀ (r : Rectangle),
      rectangleArea r = 1 / 200 →
      ∃ (p : Point), p ∈ points ∧ pointInRectangle p r :=
sorry

end NUMINAMATH_CALUDE_point_covering_theorem_l3200_320002


namespace NUMINAMATH_CALUDE_valid_permutations_count_l3200_320000

/-- Given integers 1 to n, where n ≥ 2, this function returns the number of permutations
    satisfying the condition that for all k = 1 to n, the kth element is ≥ k-2 -/
def countValidPermutations (n : ℕ) : ℕ :=
  2 * 3^(n-2)

/-- Theorem stating that for n ≥ 2, the number of permutations of integers 1 to n
    satisfying the condition that for all k = 1 to n, the kth element is ≥ k-2,
    is equal to 2 * 3^(n-2) -/
theorem valid_permutations_count (n : ℕ) (h : n ≥ 2) :
  (Finset.univ.filter (fun p : Fin n → Fin n =>
    ∀ k : Fin n, p k ≥ ⟨k - 2, by sorry⟩)).card = countValidPermutations n := by
  sorry

end NUMINAMATH_CALUDE_valid_permutations_count_l3200_320000


namespace NUMINAMATH_CALUDE_integer_product_characterization_l3200_320035

theorem integer_product_characterization (a : ℝ) : 
  (∀ n : ℕ, ∃ m : ℤ, a * n * (n + 2) * (n + 3) * (n + 4) = m) ↔ 
  (∃ k : ℤ, a = k / 6) :=
sorry

end NUMINAMATH_CALUDE_integer_product_characterization_l3200_320035


namespace NUMINAMATH_CALUDE_range_of_a_l3200_320070

theorem range_of_a (p q : Prop) (a : ℝ) : 
  (∀ x > -1, p → x^2 / (x + 1) ≥ a) →
  (q ↔ ∃ x : ℝ, a * x^2 - a * x + 1 = 0) →
  (¬p ∧ ¬q) →
  (p ∨ q) →
  (a = 0 ∨ a ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3200_320070


namespace NUMINAMATH_CALUDE_water_remaining_in_cylinder_l3200_320007

/-- The volume of water remaining in a cylinder after pouring some into a cone -/
theorem water_remaining_in_cylinder (cylinder_volume cone_volume : ℝ) : 
  cylinder_volume = 18 →
  cylinder_volume = 3 * cone_volume →
  cylinder_volume - cone_volume = 12 :=
by sorry

end NUMINAMATH_CALUDE_water_remaining_in_cylinder_l3200_320007


namespace NUMINAMATH_CALUDE_tan_135_degrees_l3200_320055

theorem tan_135_degrees : Real.tan (135 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_135_degrees_l3200_320055


namespace NUMINAMATH_CALUDE_game_x_vs_game_y_l3200_320052

def coin_prob_heads : ℚ := 3/4
def coin_prob_tails : ℚ := 1/4

def game_x_win_prob : ℚ :=
  4 * (coin_prob_heads^4 * coin_prob_tails + coin_prob_tails^4 * coin_prob_heads)

def game_y_win_prob : ℚ :=
  coin_prob_heads^6 + coin_prob_tails^6

theorem game_x_vs_game_y :
  game_x_win_prob - game_y_win_prob = 298/2048 := by sorry

end NUMINAMATH_CALUDE_game_x_vs_game_y_l3200_320052


namespace NUMINAMATH_CALUDE_remainder_problem_l3200_320003

theorem remainder_problem : ∃ k : ℤ, 
  2^6 * 3^10 * 5^12 - 75^4 * (26^2 - 1)^2 + 3^10 - 50^6 + 5^12 = 1001 * k + 400 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3200_320003


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3200_320057

-- Define the function f
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem inequality_solution_set (x : ℝ) :
  (f (2*x) + f (x-1) < 0) ↔ (x < 1/3) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3200_320057


namespace NUMINAMATH_CALUDE_brad_carl_weight_difference_l3200_320093

/-- Given the weights of Billy, Brad, and Carl, prove that Brad weighs 5 pounds more than Carl. -/
theorem brad_carl_weight_difference
  (billy_weight : ℕ)
  (brad_weight : ℕ)
  (carl_weight : ℕ)
  (h1 : billy_weight = brad_weight + 9)
  (h2 : brad_weight > carl_weight)
  (h3 : carl_weight = 145)
  (h4 : billy_weight = 159) :
  brad_weight - carl_weight = 5 := by
  sorry

end NUMINAMATH_CALUDE_brad_carl_weight_difference_l3200_320093


namespace NUMINAMATH_CALUDE_unique_solution_cube_equation_l3200_320054

theorem unique_solution_cube_equation :
  ∃! (x : ℕ+), (2 * x.val)^3 - x.val = 726 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cube_equation_l3200_320054


namespace NUMINAMATH_CALUDE_a_minus_b_value_l3200_320025

theorem a_minus_b_value (a b : ℝ) : 
  (|a - 2| = 5) → (|b| = 9) → (a + b < 0) → (a - b = 16 ∨ a - b = 6) := by
  sorry

end NUMINAMATH_CALUDE_a_minus_b_value_l3200_320025


namespace NUMINAMATH_CALUDE_fifteenth_valid_number_l3200_320059

def digit_sum (n : ℕ) : ℕ := sorry

def is_valid_number (n : ℕ) : Prop :=
  n > 0 ∧ digit_sum n = 14

def nth_valid_number (n : ℕ) : ℕ := sorry

theorem fifteenth_valid_number :
  nth_valid_number 15 = 266 := by sorry

end NUMINAMATH_CALUDE_fifteenth_valid_number_l3200_320059


namespace NUMINAMATH_CALUDE_james_age_when_thomas_reaches_current_l3200_320042

theorem james_age_when_thomas_reaches_current (T : ℕ) : 
  let shay_age := T + 13
  let james_current_age := T + 18
  james_current_age = 42 →
  james_current_age + (james_current_age - T) = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_james_age_when_thomas_reaches_current_l3200_320042


namespace NUMINAMATH_CALUDE_fraction_decomposition_l3200_320023

theorem fraction_decomposition (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  (2 / (x^2 - 1) = 1 / (x - 1) - 1 / (x + 1)) ∧
  (2*x / (x^2 - 1) = 1 / (x - 1) + 1 / (x + 1)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l3200_320023


namespace NUMINAMATH_CALUDE_min_occupied_seats_for_150_l3200_320008

/-- The minimum number of occupied seats required to ensure the next person must sit next to someone, given a total number of seats. -/
def min_occupied_seats (total_seats : ℕ) : ℕ :=
  2 * (total_seats / 4)

theorem min_occupied_seats_for_150 :
  min_occupied_seats 150 = 74 := by
  sorry

#eval min_occupied_seats 150

end NUMINAMATH_CALUDE_min_occupied_seats_for_150_l3200_320008


namespace NUMINAMATH_CALUDE_amount_C_is_correct_l3200_320044

/-- The amount C receives when $5000 is divided among A, B, C, and D in the ratio 1:3:5:7 -/
def amount_C : ℚ :=
  let total_amount : ℚ := 5000
  let ratio_A : ℚ := 1
  let ratio_B : ℚ := 3
  let ratio_C : ℚ := 5
  let ratio_D : ℚ := 7
  let total_ratio : ℚ := ratio_A + ratio_B + ratio_C + ratio_D
  (total_amount / total_ratio) * ratio_C

theorem amount_C_is_correct : amount_C = 1562.50 := by
  sorry

end NUMINAMATH_CALUDE_amount_C_is_correct_l3200_320044


namespace NUMINAMATH_CALUDE_distinct_numbers_count_l3200_320063

/-- Represents a two-sided card with distinct numbers on each side -/
structure Card where
  side1 : ℕ
  side2 : ℕ
  distinct : side1 ≠ side2

/-- The set of four cards as described in the problem -/
def card_set : Finset Card := sorry

/-- A function that generates all possible three-digit numbers from the card set -/
def generate_numbers (cards : Finset Card) : Finset ℕ := sorry

/-- The main theorem stating that the number of distinct three-digit numbers is 192 -/
theorem distinct_numbers_count : 
  (generate_numbers card_set).card = 192 := by sorry

end NUMINAMATH_CALUDE_distinct_numbers_count_l3200_320063


namespace NUMINAMATH_CALUDE_passengers_landed_late_l3200_320036

theorem passengers_landed_late (on_time passengers : ℕ) (total_passengers : ℕ) 
  (h1 : on_time_passengers = 14507)
  (h2 : total_passengers = 14720) :
  total_passengers - on_time_passengers = 213 := by
  sorry

end NUMINAMATH_CALUDE_passengers_landed_late_l3200_320036


namespace NUMINAMATH_CALUDE_password_length_l3200_320088

/-- Represents the structure of Pat's password --/
structure PasswordStructure where
  lowercase_letters : Nat
  alternating_chars : Nat
  digits : Nat
  symbols : Nat

/-- Theorem stating that Pat's password contains 22 characters --/
theorem password_length (pw : PasswordStructure) 
  (h1 : pw.lowercase_letters = 10)
  (h2 : pw.alternating_chars = 6)
  (h3 : pw.digits = 4)
  (h4 : pw.symbols = 2) : 
  pw.lowercase_letters + pw.alternating_chars + pw.digits + pw.symbols = 22 := by
  sorry

#check password_length

end NUMINAMATH_CALUDE_password_length_l3200_320088


namespace NUMINAMATH_CALUDE_dave_initial_apps_l3200_320020

/-- The number of apps Dave deleted -/
def deleted_apps : ℕ := 8

/-- The number of apps Dave had left after deleting -/
def remaining_apps : ℕ := 8

/-- The initial number of apps Dave had -/
def initial_apps : ℕ := deleted_apps + remaining_apps

theorem dave_initial_apps : initial_apps = 16 := by
  sorry

end NUMINAMATH_CALUDE_dave_initial_apps_l3200_320020


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l3200_320072

theorem cyclic_sum_inequality (a b c : ℝ) (n : ℕ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_prod : a * b * c = 1) :
  (a + b + c^n) / (a^(2*n+3) + b^(2*n+3) + a*b) +
  (b + c + a^n) / (b^(2*n+3) + c^(2*n+3) + b*c) +
  (c + a + b^n) / (c^(2*n+3) + a^(2*n+3) + c*a) ≤
  a^(n+1) + b^(n+1) + c^(n+1) := by sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l3200_320072


namespace NUMINAMATH_CALUDE_simplify_expression_l3200_320067

theorem simplify_expression : 
  3 * Real.sqrt 48 - 9 * Real.sqrt (1/3) - Real.sqrt 3 * (2 - Real.sqrt 27) = 7 * Real.sqrt 3 + 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3200_320067


namespace NUMINAMATH_CALUDE_x_range_l3200_320071

theorem x_range (x : ℝ) : (|x + 1| + |x - 1| = 2) ↔ (-1 ≤ x ∧ x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_x_range_l3200_320071


namespace NUMINAMATH_CALUDE_jasons_books_count_l3200_320017

/-- The number of books each shelf can hold -/
def books_per_shelf : ℕ := 45

/-- The number of shelves Jason needs -/
def shelves_needed : ℕ := 7

/-- The total number of books Jason has -/
def total_books : ℕ := books_per_shelf * shelves_needed

theorem jasons_books_count : total_books = 315 := by
  sorry

end NUMINAMATH_CALUDE_jasons_books_count_l3200_320017


namespace NUMINAMATH_CALUDE_power_of_three_equation_l3200_320040

theorem power_of_three_equation (m : ℤ) : 
  3^2001 - 2 * 3^2000 - 3^1999 + 5 * 3^1998 = m * 3^1998 → m = 11 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_equation_l3200_320040


namespace NUMINAMATH_CALUDE_wall_building_time_l3200_320066

/-- Represents the time taken to build a wall given different workforce scenarios -/
theorem wall_building_time
  (original_men : ℕ)
  (original_days : ℕ)
  (new_total_men : ℕ)
  (fast_men : ℕ)
  (h1 : original_men = 20)
  (h2 : original_days = 6)
  (h3 : new_total_men = 30)
  (h4 : fast_men = 10)
  (h5 : fast_men ≤ new_total_men) :
  let effective_workforce := new_total_men - fast_men + 2 * fast_men
  let new_days := (original_men * original_days) / effective_workforce
  new_days = 3 := by sorry

end NUMINAMATH_CALUDE_wall_building_time_l3200_320066


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l3200_320015

def A (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def B (a : ℝ) : Set ℝ := {a-3, 2*a-1, a^2+1}

theorem intersection_implies_a_value (a : ℝ) :
  A a ∩ B a = {-3} → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l3200_320015


namespace NUMINAMATH_CALUDE_largest_class_size_l3200_320079

/-- Proves that in a school with 5 classes, where each class has 2 students less than the previous class,
    and the total number of students is 120, the largest class has 28 students. -/
theorem largest_class_size (n : ℕ) (h1 : n = 5) (total : ℕ) (h2 : total = 120) :
  ∃ x : ℕ, x = 28 ∧ 
    x + (x - 2) + (x - 4) + (x - 6) + (x - 8) = total :=
by sorry

end NUMINAMATH_CALUDE_largest_class_size_l3200_320079


namespace NUMINAMATH_CALUDE_no_solution_exists_l3200_320076

theorem no_solution_exists : ¬ ∃ n : ℤ, 10 ≤ n ∧ n ≤ 15 ∧ n ≡ 10467 [ZMOD 7] := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3200_320076


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l3200_320061

/-- An isosceles triangle with a given base and inscribed circle property -/
structure IsoscelesTriangleWithInscribedCircle where
  base : ℝ
  sideLength : ℝ
  inscribedCircleRadius : ℝ
  isIsosceles : sideLength > 0
  hasInscribedCircle : inscribedCircleRadius > 0

/-- The property of having three tangents to the inscribed circle -/
def HasThreeTangents (triangle : IsoscelesTriangleWithInscribedCircle) 
  (sumOfSmallerPerimeters : ℝ) : Prop :=
  ∃ (t1 t2 t3 : ℝ), 
    t1 > 0 ∧ t2 > 0 ∧ t3 > 0 ∧
    t1 + t2 + t3 = sumOfSmallerPerimeters ∧
    t1 + t2 + t3 = triangle.base + 2 * triangle.sideLength

theorem isosceles_triangle_side_length 
  (triangle : IsoscelesTriangleWithInscribedCircle) 
  (b : ℝ) 
  (h : HasThreeTangents triangle b) : 
  triangle.sideLength = (b - triangle.base) / 2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l3200_320061


namespace NUMINAMATH_CALUDE_stratified_sample_factory_a_l3200_320091

theorem stratified_sample_factory_a (total : ℕ) (factory_a : ℕ) (sample_size : ℕ)
  (h_total : total = 98)
  (h_factory_a : factory_a = 56)
  (h_sample_size : sample_size = 14) :
  (factory_a : ℚ) / total * sample_size = 8 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_factory_a_l3200_320091


namespace NUMINAMATH_CALUDE_arcsin_equation_solution_l3200_320090

theorem arcsin_equation_solution :
  ∃ y : ℝ, y = 1 / Real.sqrt 19 ∧ Real.arcsin y + Real.arcsin (3 * y) = π / 4 :=
by sorry

end NUMINAMATH_CALUDE_arcsin_equation_solution_l3200_320090


namespace NUMINAMATH_CALUDE_power_of_seven_l3200_320031

theorem power_of_seven (k : ℕ) (h : 7^k = 2) : 7^(4*k + 2) = 784 := by
  sorry

end NUMINAMATH_CALUDE_power_of_seven_l3200_320031


namespace NUMINAMATH_CALUDE_negation_is_false_l3200_320077

theorem negation_is_false : 
  ¬(∀ x y : ℝ, (x > 2 ∧ y > 3) → x + y > 5) = False := by sorry

end NUMINAMATH_CALUDE_negation_is_false_l3200_320077
