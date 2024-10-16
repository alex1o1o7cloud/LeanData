import Mathlib

namespace NUMINAMATH_CALUDE_my_algebra_theorem_l3882_388272

/-- A binary operation on a type -/
def BinOp (α : Type*) := α → α → α

/-- A type with a binary operation satisfying the given property -/
class MyAlgebra (X : Type*) where
  mul : BinOp X
  mul_property : ∀ x y : X, mul (mul x y) x = y

theorem my_algebra_theorem (X : Type*) [MyAlgebra X] :
  ∀ x y : X, MyAlgebra.mul x (MyAlgebra.mul y x) = y := by
  sorry

end NUMINAMATH_CALUDE_my_algebra_theorem_l3882_388272


namespace NUMINAMATH_CALUDE_chess_class_percentage_l3882_388241

theorem chess_class_percentage 
  (total_students : ℕ) 
  (swimming_students : ℕ) 
  (chess_to_swimming_ratio : ℚ) :
  total_students = 1000 →
  swimming_students = 125 →
  chess_to_swimming_ratio = 1/2 →
  (↑swimming_students : ℚ) / (chess_to_swimming_ratio * ↑total_students) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_chess_class_percentage_l3882_388241


namespace NUMINAMATH_CALUDE_count_special_integers_l3882_388244

theorem count_special_integers : 
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, 2 ≤ n ∧ n ≤ 2016 ∧ 
      (2 ∣ n^n - 1) ∧ (3 ∣ n^n - 1) ∧ (5 ∣ n^n - 1) ∧ (7 ∣ n^n - 1)) ∧
    S.card = 9 ∧
    (∀ n : ℕ, 2 ≤ n ∧ n ≤ 2016 ∧ 
      (2 ∣ n^n - 1) ∧ (3 ∣ n^n - 1) ∧ (5 ∣ n^n - 1) ∧ (7 ∣ n^n - 1) → n ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_count_special_integers_l3882_388244


namespace NUMINAMATH_CALUDE_tree_planting_cost_l3882_388227

/-- The cost of planting trees around a circular park -/
theorem tree_planting_cost
  (park_circumference : ℕ) -- Park circumference in meters
  (planting_interval : ℕ) -- Interval between trees in meters
  (tree_cost : ℕ) -- Cost per tree in mill
  (h1 : park_circumference = 1500)
  (h2 : planting_interval = 30)
  (h3 : tree_cost = 5000) :
  (park_circumference / planting_interval) * tree_cost = 250000 := by
sorry

end NUMINAMATH_CALUDE_tree_planting_cost_l3882_388227


namespace NUMINAMATH_CALUDE_ranch_cows_count_l3882_388260

/-- Represents the number of cows and horses a rancher has -/
structure RanchAnimals where
  horses : ℕ
  cows : ℕ

/-- Represents the conditions of the ranch -/
def ranchConditions (animals : RanchAnimals) : Prop :=
  animals.cows = 5 * animals.horses ∧ animals.cows + animals.horses = 168

theorem ranch_cows_count :
  ∃ (animals : RanchAnimals), ranchConditions animals ∧ animals.cows = 140 := by
  sorry

end NUMINAMATH_CALUDE_ranch_cows_count_l3882_388260


namespace NUMINAMATH_CALUDE_unique_common_one_position_l3882_388253

/-- A binary sequence of length n -/
def BinarySequence (n : ℕ) := Fin n → Bool

/-- The property that for any three sequences, there exists a position where all three have a 1 -/
def ThreeSequenceProperty (n : ℕ) (sequences : Finset (BinarySequence n)) : Prop :=
  ∀ s1 s2 s3 : BinarySequence n, s1 ∈ sequences → s2 ∈ sequences → s3 ∈ sequences →
    ∃ p : Fin n, s1 p = true ∧ s2 p = true ∧ s3 p = true

/-- The main theorem to be proved -/
theorem unique_common_one_position
  (n : ℕ) (sequences : Finset (BinarySequence n))
  (h_count : sequences.card = 2^(n-1))
  (h_three : ThreeSequenceProperty n sequences) :
  ∃! p : Fin n, ∀ s ∈ sequences, s p = true :=
sorry

end NUMINAMATH_CALUDE_unique_common_one_position_l3882_388253


namespace NUMINAMATH_CALUDE_man_rowing_speed_l3882_388243

/-- 
Given a man's rowing speed against the stream and his speed in still water,
calculate his speed with the stream.
-/
theorem man_rowing_speed 
  (speed_against_stream : ℝ) 
  (speed_still_water : ℝ) 
  (h1 : speed_against_stream = 4) 
  (h2 : speed_still_water = 6) : 
  speed_still_water + (speed_still_water - speed_against_stream) = 8 := by
  sorry

#check man_rowing_speed

end NUMINAMATH_CALUDE_man_rowing_speed_l3882_388243


namespace NUMINAMATH_CALUDE_ratio_b_to_a_is_one_l3882_388286

/-- An arithmetic sequence with first four terms a, b, x, and 2x - 1/2 -/
structure ArithmeticSequence (a b x : ℝ) : Prop where
  term1 : a = a
  term2 : b = b
  term3 : x = x
  term4 : 2 * x - 1/2 = 2 * x - 1/2
  is_arithmetic : ∃ (d : ℝ), b - a = d ∧ x - b = d ∧ (2 * x - 1/2) - x = d

/-- The ratio of b to a in the arithmetic sequence is 1 -/
theorem ratio_b_to_a_is_one {a b x : ℝ} (h : ArithmeticSequence a b x) : b / a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_b_to_a_is_one_l3882_388286


namespace NUMINAMATH_CALUDE_next_year_with_sum_4_year_2101_is_valid_year_2101_is_smallest_l3882_388288

def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

def isValidYear (year : Nat) : Prop :=
  year > 2020 ∧ sumOfDigits year = 4

theorem next_year_with_sum_4 :
  ∀ year, year > 2020 → sumOfDigits year = 4 → year ≥ 2101 :=
by sorry

theorem year_2101_is_valid :
  isValidYear 2101 :=
by sorry

theorem year_2101_is_smallest :
  ∀ year, isValidYear year → year ≥ 2101 :=
by sorry

end NUMINAMATH_CALUDE_next_year_with_sum_4_year_2101_is_valid_year_2101_is_smallest_l3882_388288


namespace NUMINAMATH_CALUDE_expression_value_l3882_388208

theorem expression_value : (fun x : ℝ => x^2 + 3*x - 4) 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3882_388208


namespace NUMINAMATH_CALUDE_multiplication_and_addition_l3882_388223

theorem multiplication_and_addition : 2 * (-2) + (-3) = -7 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_and_addition_l3882_388223


namespace NUMINAMATH_CALUDE_nicole_clothes_proof_l3882_388250

/-- Calculates the total number of clothing pieces Nicole ends up with --/
def nicole_total_clothes (nicole_initial : ℕ) : ℕ :=
  let sister1 := nicole_initial / 2
  let sister2 := nicole_initial + 2
  let sister3 := (nicole_initial + sister1 + sister2) / 3
  nicole_initial + sister1 + sister2 + sister3

/-- Proves that Nicole ends up with 36 pieces of clothing --/
theorem nicole_clothes_proof :
  nicole_total_clothes 10 = 36 := by
  sorry

#eval nicole_total_clothes 10

end NUMINAMATH_CALUDE_nicole_clothes_proof_l3882_388250


namespace NUMINAMATH_CALUDE_ethel_mental_math_l3882_388283

theorem ethel_mental_math (square_50 : 50^2 = 2500) :
  49^2 = 2500 - 99 := by
  sorry

end NUMINAMATH_CALUDE_ethel_mental_math_l3882_388283


namespace NUMINAMATH_CALUDE_complete_square_factorization_l3882_388285

theorem complete_square_factorization :
  ∀ (x : ℝ), x^2 + 2*x + 1 = (x + 1)^2 := by
  sorry

#check complete_square_factorization

end NUMINAMATH_CALUDE_complete_square_factorization_l3882_388285


namespace NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l3882_388290

theorem pentagon_rectangle_ratio : 
  let pentagon_perimeter : ℝ := 60
  let rectangle_perimeter : ℝ := 60
  let pentagon_side : ℝ := pentagon_perimeter / 5
  let rectangle_width : ℝ := rectangle_perimeter / 6
  pentagon_side / rectangle_width = 6 / 5 := by
sorry

end NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l3882_388290


namespace NUMINAMATH_CALUDE_fiftieth_rising_number_excludes_one_three_four_l3882_388225

/-- A rising number is a number where each digit is strictly greater than the previous digit. -/
def IsRisingNumber (n : ℕ) : Prop := sorry

/-- The set of digits used to construct the rising numbers. -/
def DigitSet : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

/-- The function that generates the nth four-digit rising number from the DigitSet. -/
def NthRisingNumber (n : ℕ) : Finset ℕ := sorry

/-- Theorem stating that the 50th rising number does not contain 1, 3, or 4. -/
theorem fiftieth_rising_number_excludes_one_three_four :
  1 ∉ NthRisingNumber 50 ∧ 3 ∉ NthRisingNumber 50 ∧ 4 ∉ NthRisingNumber 50 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_rising_number_excludes_one_three_four_l3882_388225


namespace NUMINAMATH_CALUDE_percentage_relation_l3882_388262

theorem percentage_relation (x y : ℕ) (N : ℚ) (hx : Prime x) (hy : Prime y) (hxy : x ≠ y) 
  (h : 70 = (x : ℚ) / 100 * N) : 
  (y : ℚ) / 100 * N = (y * 70 : ℚ) / x := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l3882_388262


namespace NUMINAMATH_CALUDE_factor_x8_minus_81_l3882_388210

theorem factor_x8_minus_81 (x : ℝ) : x^8 - 81 = (x^4 + 9) * (x^2 + 3) * (x + Real.sqrt 3) * (x - Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_x8_minus_81_l3882_388210


namespace NUMINAMATH_CALUDE_confectioner_customers_l3882_388256

/-- The number of regular customers for a confectioner -/
def regular_customers : ℕ := 28

/-- The total number of pastries -/
def total_pastries : ℕ := 392

/-- The number of customers in the alternative scenario -/
def alternative_customers : ℕ := 49

/-- The difference in pastries per customer between regular and alternative scenarios -/
def pastry_difference : ℕ := 6

theorem confectioner_customers :
  regular_customers = 28 ∧
  total_pastries = 392 ∧
  alternative_customers = 49 ∧
  pastry_difference = 6 ∧
  (total_pastries / regular_customers : ℚ) = 
    (total_pastries / alternative_customers : ℚ) + pastry_difference := by
  sorry

end NUMINAMATH_CALUDE_confectioner_customers_l3882_388256


namespace NUMINAMATH_CALUDE_distance_covered_l3882_388217

/-- Proves that the distance covered is 30 km given the conditions of the problem -/
theorem distance_covered (D : ℝ) (S : ℝ) : 
  (D / 5 = D / S + 2) →    -- Abhay takes 2 hours more than Sameer
  (D / 10 = D / S - 1) →   -- If Abhay doubles his speed, he takes 1 hour less than Sameer
  D = 30 := by             -- The distance covered is 30 km
sorry

end NUMINAMATH_CALUDE_distance_covered_l3882_388217


namespace NUMINAMATH_CALUDE_amithab_average_expenditure_l3882_388206

/-- Given Amithab's monthly expenses, prove the average expenditure for February to July. -/
theorem amithab_average_expenditure
  (jan_expense : ℕ)
  (jan_to_jun_avg : ℕ)
  (jul_expense : ℕ)
  (h1 : jan_expense = 1200)
  (h2 : jan_to_jun_avg = 4200)
  (h3 : jul_expense = 1500) :
  (6 * jan_to_jun_avg - jan_expense + jul_expense) / 6 = 4250 :=
by sorry

end NUMINAMATH_CALUDE_amithab_average_expenditure_l3882_388206


namespace NUMINAMATH_CALUDE_max_projection_area_specific_tetrahedron_l3882_388292

/-- Represents a tetrahedron with two adjacent equilateral triangular faces --/
structure Tetrahedron where
  side_length : ℝ
  dihedral_angle : ℝ

/-- Calculates the maximum projection area of a rotating tetrahedron --/
def max_projection_area (t : Tetrahedron) : ℝ :=
  sorry

/-- The theorem stating the maximum projection area of the specific tetrahedron --/
theorem max_projection_area_specific_tetrahedron :
  let t : Tetrahedron := { side_length := 1, dihedral_angle := π / 3 }
  max_projection_area t = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_projection_area_specific_tetrahedron_l3882_388292


namespace NUMINAMATH_CALUDE_disjoint_equal_sum_subsets_l3882_388278

theorem disjoint_equal_sum_subsets (S : Finset ℕ) 
  (h1 : S ⊆ Finset.range 2018)
  (h2 : S.card = 68) :
  ∃ (A B C : Finset ℕ), 
    A ⊆ S ∧ B ⊆ S ∧ C ⊆ S ∧
    A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ ∧
    A.card = B.card ∧ B.card = C.card ∧
    A.sum id = B.sum id ∧ B.sum id = C.sum id :=
by sorry

end NUMINAMATH_CALUDE_disjoint_equal_sum_subsets_l3882_388278


namespace NUMINAMATH_CALUDE_doudou_mother_age_l3882_388237

/-- Represents the ages of Doudou's family members -/
structure FamilyAges where
  doudou : ℕ
  brother : ℕ
  mother : ℕ
  father : ℕ

/-- The conditions of the problem -/
def problemConditions (ages : FamilyAges) : Prop :=
  ages.brother = ages.doudou + 3 ∧
  ages.mother = ages.father - 2 ∧
  ages.doudou + ages.brother + ages.mother + ages.father - 20 = 59 ∧
  ages.doudou + ages.brother + ages.mother + ages.father + 20 = 97

/-- The theorem to be proved -/
theorem doudou_mother_age (ages : FamilyAges) :
  problemConditions ages → ages.mother = 33 := by
  sorry


end NUMINAMATH_CALUDE_doudou_mother_age_l3882_388237


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3882_388266

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 5) * (Real.sqrt 5 / Real.sqrt 7) * (Real.sqrt 8 / Real.sqrt 11) =
  2 * Real.sqrt 462 / 77 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3882_388266


namespace NUMINAMATH_CALUDE_simplification_to_x_plus_one_l3882_388212

theorem simplification_to_x_plus_one (x : ℝ) (h : x ≠ 1) :
  (x^2 / (x - 1)) - (1 / (x - 1)) = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_simplification_to_x_plus_one_l3882_388212


namespace NUMINAMATH_CALUDE_village_population_l3882_388273

/-- Given that 90% of a population is 23040, prove that the total population is 25600 -/
theorem village_population (population : ℕ) : 
  (90 : ℚ) / 100 * population = 23040 → population = 25600 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l3882_388273


namespace NUMINAMATH_CALUDE_certain_number_value_l3882_388295

theorem certain_number_value : 
  ∀ x : ℝ,
  (28 + x + 42 + 78 + 104) / 5 = 62 →
  ∃ y : ℝ,
  (y + 62 + 98 + 124 + x) / 5 = 78 ∧
  y = 106 :=
by
  sorry

end NUMINAMATH_CALUDE_certain_number_value_l3882_388295


namespace NUMINAMATH_CALUDE_heavy_rain_time_is_14_minutes_l3882_388277

/-- Represents the weather conditions during the trip -/
inductive Weather
  | Sun
  | LightRain
  | HeavyRain

/-- Represents Shelby's scooter journey -/
structure Journey where
  totalDistance : ℝ
  totalTime : ℝ
  speedSun : ℝ
  speedLightRain : ℝ
  speedHeavyRain : ℝ

/-- Calculates the time spent in heavy rain given a journey -/
def timeInHeavyRain (j : Journey) : ℝ :=
  sorry

/-- Theorem stating that given the specific journey conditions, 
    the time spent in heavy rain is 14 minutes -/
theorem heavy_rain_time_is_14_minutes 
  (j : Journey)
  (h1 : j.totalDistance = 18)
  (h2 : j.totalTime = 50)
  (h3 : j.speedSun = 30)
  (h4 : j.speedLightRain = 20)
  (h5 : j.speedHeavyRain = 15)
  (h6 : timeInHeavyRain j = timeInHeavyRain j) -- Represents equal rain segments
  : timeInHeavyRain j = 14 := by
  sorry

end NUMINAMATH_CALUDE_heavy_rain_time_is_14_minutes_l3882_388277


namespace NUMINAMATH_CALUDE_solve_system_l3882_388298

theorem solve_system (x y : ℚ) 
  (eq1 : 3 * x + 4 * y = 0) 
  (eq2 : y - 3 = x) : 
  5 * y = 45 / 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l3882_388298


namespace NUMINAMATH_CALUDE_larry_wins_probability_l3882_388263

theorem larry_wins_probability (p : ℝ) (q : ℝ) (hp : p = 1/3) (hq : q = 1/4) :
  let win_prob := p / (1 - (1 - p) * (1 - q))
  win_prob = 2/3 := by sorry

end NUMINAMATH_CALUDE_larry_wins_probability_l3882_388263


namespace NUMINAMATH_CALUDE_units_digit_17_310_l3882_388229

/-- The units digit of 7^n for n ≥ 1 -/
def unitsDigit7 (n : ℕ) : ℕ :=
  match n % 4 with
  | 1 => 7
  | 2 => 9
  | 3 => 3
  | 0 => 1
  | _ => 0  -- This case should never occur

/-- The units digit of 17^n follows the same pattern as 7^n -/
axiom unitsDigit17 (n : ℕ) : n ≥ 1 → unitsDigit7 n = (17^n) % 10

theorem units_digit_17_310 : (17^310) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_17_310_l3882_388229


namespace NUMINAMATH_CALUDE_matrix_product_is_zero_l3882_388274

variable (a b c : ℝ)

def matrix1 : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 2*c, -2*b],
    ![-2*c, 0, 2*a],
    ![2*b, -2*a, 0]]

def matrix2 : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![a^2, 2*a*b, 2*a*c],
    ![2*a*b, b^2, 2*b*c],
    ![2*a*c, 2*b*c, c^2]]

theorem matrix_product_is_zero :
  matrix1 a b c * matrix2 a b c = 0 := by sorry

end NUMINAMATH_CALUDE_matrix_product_is_zero_l3882_388274


namespace NUMINAMATH_CALUDE_repeating_decimal_problem_l3882_388294

theorem repeating_decimal_problem (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) :
  72 * ((1 + (100 * a + 10 * b + c : ℕ) / 999 : ℚ) - (1 + (a / 10 + b / 100 + c / 1000 : ℚ))) = (3 / 5 : ℚ) →
  100 * a + 10 * b + c = 833 := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_problem_l3882_388294


namespace NUMINAMATH_CALUDE_cos_alpha_value_l3882_388276

theorem cos_alpha_value (α : Real) (h : Real.sin (5 * Real.pi / 2 + α) = 1 / 5) : 
  Real.cos α = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l3882_388276


namespace NUMINAMATH_CALUDE_dog_adoption_rate_is_half_l3882_388240

/-- Represents the animal shelter scenario --/
structure AnimalShelter where
  initialDogs : ℕ
  initialCats : ℕ
  initialLizards : ℕ
  catAdoptionRate : ℚ
  lizardAdoptionRate : ℚ
  newPetsPerMonth : ℕ
  totalPetsAfterMonth : ℕ

/-- Calculates the dog adoption rate given the shelter scenario --/
def dogAdoptionRate (shelter : AnimalShelter) : ℚ :=
  let totalInitial := shelter.initialDogs + shelter.initialCats + shelter.initialLizards
  let adoptedCats := shelter.catAdoptionRate * shelter.initialCats
  let adoptedLizards := shelter.lizardAdoptionRate * shelter.initialLizards
  let remainingPets := shelter.totalPetsAfterMonth - shelter.newPetsPerMonth
  ((totalInitial - remainingPets) - (adoptedCats + adoptedLizards)) / shelter.initialDogs

/-- Theorem stating that the dog adoption rate is 50% for the given scenario --/
theorem dog_adoption_rate_is_half (shelter : AnimalShelter) 
  (h1 : shelter.initialDogs = 30)
  (h2 : shelter.initialCats = 28)
  (h3 : shelter.initialLizards = 20)
  (h4 : shelter.catAdoptionRate = 1/4)
  (h5 : shelter.lizardAdoptionRate = 1/5)
  (h6 : shelter.newPetsPerMonth = 13)
  (h7 : shelter.totalPetsAfterMonth = 65) :
  dogAdoptionRate shelter = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_dog_adoption_rate_is_half_l3882_388240


namespace NUMINAMATH_CALUDE_number_difference_l3882_388252

theorem number_difference (a b c : ℝ) : 
  a = 2 * b ∧ 
  a = 3 * c ∧ 
  (a + b + c) / 3 = 88 → 
  a - c = 96 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l3882_388252


namespace NUMINAMATH_CALUDE_a_3_value_l3882_388222

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = -3

theorem a_3_value (a : ℕ → ℤ) :
  arithmetic_sequence a → a 1 = 7 → a 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_a_3_value_l3882_388222


namespace NUMINAMATH_CALUDE_arcsin_sqrt3_over_2_l3882_388204

theorem arcsin_sqrt3_over_2 : Real.arcsin (Real.sqrt 3 / 2) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_sqrt3_over_2_l3882_388204


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3882_388235

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- arithmetic sequence with common ratio q
  abs q > 1 →                   -- |q| > 1
  a 2 + a 7 = 2 →               -- a₂ + a₇ = 2
  a 4 * a 5 = -15 →             -- a₄a₅ = -15
  a 12 = -25 / 3 :=              -- a₁₂ = -25/3
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3882_388235


namespace NUMINAMATH_CALUDE_number_puzzle_l3882_388269

theorem number_puzzle (x : ℝ) : x / 3 = x - 42 → x = 63 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l3882_388269


namespace NUMINAMATH_CALUDE_sara_pears_l3882_388238

theorem sara_pears (total_pears sally_pears : ℕ) 
  (h1 : total_pears = 56)
  (h2 : sally_pears = 11) :
  total_pears - sally_pears = 45 := by
  sorry

end NUMINAMATH_CALUDE_sara_pears_l3882_388238


namespace NUMINAMATH_CALUDE_fraction_reduction_divisibility_l3882_388275

theorem fraction_reduction_divisibility
  (a b c d n : ℕ)
  (h1 : (a * n + b) % 2017 = 0)
  (h2 : (c * n + d) % 2017 = 0) :
  (a * d - b * c) % 2017 = 0 :=
by sorry

end NUMINAMATH_CALUDE_fraction_reduction_divisibility_l3882_388275


namespace NUMINAMATH_CALUDE_special_divisibility_property_l3882_388287

theorem special_divisibility_property (a b : ℕ) (ha : a ≥ 2) (hb : b ≥ 2) :
  (∀ n : ℕ, n > 0 → a^n - n^a ≠ 0 → (a^n - n^a) ∣ (b^n - n^b)) ↔
  ((a = 2 ∧ b = 2) ∨ (a = 2 ∧ b = 4) ∨ (a = b ∧ a > 1)) :=
by sorry

end NUMINAMATH_CALUDE_special_divisibility_property_l3882_388287


namespace NUMINAMATH_CALUDE_a_10_equals_505_l3882_388248

def sequence_a (n : ℕ) : ℕ :=
  if n = 1 then 1
  else
    let start := (n * (n - 1)) / 2 + 1
    (start + start + n - 1) * n / 2

theorem a_10_equals_505 : sequence_a 10 = 505 := by
  sorry

end NUMINAMATH_CALUDE_a_10_equals_505_l3882_388248


namespace NUMINAMATH_CALUDE_angle_A_value_max_area_l3882_388214

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = Real.sqrt 2 ∧
  (1/2) * t.b * t.c * Real.sin t.A = (Real.sqrt 2 / 2) * (t.c * Real.sin t.C + t.b * Real.sin t.B - t.a * Real.sin t.A)

-- Theorem for the value of angle A
theorem angle_A_value (t : Triangle) (h : triangle_conditions t) : t.A = π / 3 := by
  sorry

-- Theorem for the maximum area of triangle ABC
theorem max_area (t : Triangle) (h : triangle_conditions t) : 
  (∀ t' : Triangle, triangle_conditions t' → (1/2) * t'.b * t'.c * Real.sin t'.A ≤ Real.sqrt 3 / 2) ∧
  (∃ t' : Triangle, triangle_conditions t' ∧ (1/2) * t'.b * t'.c * Real.sin t'.A = Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_angle_A_value_max_area_l3882_388214


namespace NUMINAMATH_CALUDE_equations_solutions_l3882_388258

-- Define the equations
def equation1 (x : ℝ) : Prop := x^2 - 4*x - 1 = 0
def equation2 (x : ℝ) : Prop := (x + 3)^2 = x + 3

-- Define the solution sets
def solutions1 : Set ℝ := {2 + Real.sqrt 5, 2 - Real.sqrt 5}
def solutions2 : Set ℝ := {-3, -2}

-- Theorem statement
theorem equations_solutions :
  (∀ x : ℝ, equation1 x ↔ x ∈ solutions1) ∧
  (∀ x : ℝ, equation2 x ↔ x ∈ solutions2) := by
  sorry

end NUMINAMATH_CALUDE_equations_solutions_l3882_388258


namespace NUMINAMATH_CALUDE_midpoint_cut_equal_parts_l3882_388236

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ
  h_positive : 0 < length ∧ 0 < width
  h_length_gt_width : length > width

/-- Represents the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.length * r.width

/-- Represents a cut parallel to the shorter side of the rectangle -/
def parallel_cut (r : Rectangle) (x : ℝ) : ℝ := x * r.width

/-- Theorem stating that cutting a rectangle at its midpoint results in two equal parts -/
theorem midpoint_cut_equal_parts (r : Rectangle) :
  parallel_cut r (r.length / 2) = r.area / 2 := by sorry

end NUMINAMATH_CALUDE_midpoint_cut_equal_parts_l3882_388236


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l3882_388245

theorem circle_center_and_radius :
  ∀ (x y : ℝ), x^2 + y^2 - 4*x + 2*y = 0 →
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (2, -1) ∧
    radius = Real.sqrt 5 ∧
    (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l3882_388245


namespace NUMINAMATH_CALUDE_largest_solution_is_three_l3882_388289

theorem largest_solution_is_three :
  let f (x : ℝ) := (15 * x^2 - 40 * x + 18) / (4 * x - 3) + 7 * x
  ∃ (max : ℝ), max = 3 ∧ 
    (∀ x : ℝ, f x = 8 * x - 2 → x ≤ max) ∧
    (f max = 8 * max - 2) := by
  sorry

end NUMINAMATH_CALUDE_largest_solution_is_three_l3882_388289


namespace NUMINAMATH_CALUDE_probability_spade_then_king_diamonds_l3882_388207

-- Define the number of decks and cards per deck
def num_decks : ℕ := 6
def cards_per_deck : ℕ := 52

-- Define the total number of cards
def total_cards : ℕ := num_decks * cards_per_deck

-- Define the number of spades in the game
def num_spades : ℕ := num_decks * 13

-- Define the number of King of Diamonds in the game
def num_king_diamonds : ℕ := num_decks

-- Theorem statement
theorem probability_spade_then_king_diamonds :
  (num_spades : ℚ) / total_cards * num_king_diamonds / (total_cards - 1) = 3 / 622 := by
  sorry


end NUMINAMATH_CALUDE_probability_spade_then_king_diamonds_l3882_388207


namespace NUMINAMATH_CALUDE_excluded_students_average_mark_l3882_388216

theorem excluded_students_average_mark
  (N : ℕ)  -- Total number of students
  (A : ℚ)  -- Average mark of all students
  (E : ℕ)  -- Number of excluded students
  (AR : ℚ) -- Average mark of remaining students
  (h1 : N = 25)
  (h2 : A = 80)
  (h3 : E = 5)
  (h4 : AR = 90)
  : ∃ AE : ℚ, AE = 40 ∧ N * A - E * AE = (N - E) * AR :=
sorry

end NUMINAMATH_CALUDE_excluded_students_average_mark_l3882_388216


namespace NUMINAMATH_CALUDE_min_gennadys_required_l3882_388213

/-- Represents the number of people with a given name -/
structure Attendees :=
  (alexanders : Nat)
  (borises : Nat)
  (vasilys : Nat)
  (gennadys : Nat)

/-- Checks if the arrangement is valid (no two people with the same name are adjacent) -/
def isValidArrangement (a : Attendees) : Prop :=
  a.borises - 1 ≤ a.alexanders + a.vasilys + a.gennadys

/-- The given festival attendance -/
def festivalAttendance : Attendees :=
  { alexanders := 45
  , borises := 122
  , vasilys := 27
  , gennadys := 49 }

/-- Theorem stating that 49 is the minimum number of Gennadys required -/
theorem min_gennadys_required :
  isValidArrangement festivalAttendance ∧
  ∀ g : Nat, g < festivalAttendance.gennadys →
    ¬isValidArrangement { alexanders := festivalAttendance.alexanders
                        , borises := festivalAttendance.borises
                        , vasilys := festivalAttendance.vasilys
                        , gennadys := g } :=
by
  sorry

end NUMINAMATH_CALUDE_min_gennadys_required_l3882_388213


namespace NUMINAMATH_CALUDE_union_equals_real_complement_intersect_B_l3882_388211

-- Define sets A and B
def A : Set ℝ := {x | x - 2 ≥ 0}
def B : Set ℝ := {x | x < 5}

-- Theorem for A ∪ B = ℝ
theorem union_equals_real : A ∪ B = Set.univ := by sorry

-- Theorem for (∁ₐA) ∩ B = {x | x < 2}
theorem complement_intersect_B : 
  (Set.univ \ A) ∩ B = {x : ℝ | x < 2} := by sorry

end NUMINAMATH_CALUDE_union_equals_real_complement_intersect_B_l3882_388211


namespace NUMINAMATH_CALUDE_martha_ellen_age_ratio_l3882_388226

/-- The ratio of Martha's age to Ellen's age in six years -/
def age_ratio (martha_current_age ellen_current_age : ℕ) : ℚ :=
  (martha_current_age + 6) / (ellen_current_age + 6)

/-- Theorem stating the ratio of Martha's age to Ellen's age in six years -/
theorem martha_ellen_age_ratio :
  age_ratio 32 10 = 19 / 8 := by
  sorry

end NUMINAMATH_CALUDE_martha_ellen_age_ratio_l3882_388226


namespace NUMINAMATH_CALUDE_five_integers_average_l3882_388220

theorem five_integers_average (a b c d e : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  (a + b + c + d + e : ℚ) / 5 = 7 ∧
  ∀ (x y z w v : ℕ+), 
    x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ x ≠ v ∧
    y ≠ z ∧ y ≠ w ∧ y ≠ v ∧
    z ≠ w ∧ z ≠ v ∧
    w ≠ v ∧
    (x + y + z + w + v : ℚ) / 5 = 7 →
    (max a b - min a b : ℤ) ≥ (max x y - min x y : ℤ) ∧
    (max a c - min a c : ℤ) ≥ (max x z - min x z : ℤ) ∧
    (max a d - min a d : ℤ) ≥ (max x w - min x w : ℤ) ∧
    (max a e - min a e : ℤ) ≥ (max x v - min x v : ℤ) ∧
    (max b c - min b c : ℤ) ≥ (max y z - min y z : ℤ) ∧
    (max b d - min b d : ℤ) ≥ (max y w - min y w : ℤ) ∧
    (max b e - min b e : ℤ) ≥ (max y v - min y v : ℤ) ∧
    (max c d - min c d : ℤ) ≥ (max z w - min z w : ℤ) ∧
    (max c e - min c e : ℤ) ≥ (max z v - min z v : ℤ) ∧
    (max d e - min d e : ℤ) ≥ (max w v - min w v : ℤ) →
  (b + c + d : ℚ) / 3 = 3 := by
sorry

end NUMINAMATH_CALUDE_five_integers_average_l3882_388220


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l3882_388282

/-- An arithmetic sequence {aₙ} where a₂ = 2 and a₃ = 4 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  a 2 = 2 ∧ a 3 = 4 ∧ ∀ n : ℕ, a (n + 1) - a n = a 3 - a 2

theorem tenth_term_of_sequence (a : ℕ → ℝ) (h : arithmetic_sequence a) : a 10 = 18 :=
sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l3882_388282


namespace NUMINAMATH_CALUDE_residue_of_seven_power_l3882_388200

theorem residue_of_seven_power (n : ℕ) : 7^1234 ≡ 4 [ZMOD 13] := by
  sorry

end NUMINAMATH_CALUDE_residue_of_seven_power_l3882_388200


namespace NUMINAMATH_CALUDE_distribute_five_items_three_bags_l3882_388202

/-- The number of ways to distribute n distinct items into k identical bags --/
def distribute (n k : ℕ) : ℕ := sorry

/-- Theorem stating that distributing 5 distinct items into 3 identical bags results in 51 ways --/
theorem distribute_five_items_three_bags : distribute 5 3 = 51 := by sorry

end NUMINAMATH_CALUDE_distribute_five_items_three_bags_l3882_388202


namespace NUMINAMATH_CALUDE_intersection_of_quadratic_equations_l3882_388228

theorem intersection_of_quadratic_equations (p q : ℝ) : 
  (∃ M N : Set ℝ, 
    (∀ x, x ∈ M ↔ x^2 - p*x + 8 = 0) ∧ 
    (∀ x, x ∈ N ↔ x^2 - q*x + p = 0) ∧ 
    (M ∩ N = {1})) → 
  p + q = 19 := by
sorry

end NUMINAMATH_CALUDE_intersection_of_quadratic_equations_l3882_388228


namespace NUMINAMATH_CALUDE_problem_solution_l3882_388246

theorem problem_solution : 
  (∃ x : ℝ, 1/x < x + 1) ∧ 
  (¬(∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n)) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3882_388246


namespace NUMINAMATH_CALUDE_sqrt_3_simplest_l3882_388257

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℝ) : Prop := ∃ m : ℤ, n = m^2

-- Define what it means for a quadratic radical to be in its simplest form
def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  x > 0 ∧ ¬(is_perfect_square x) ∧ ∀ y z : ℝ, (y > 1 ∧ z > 1 ∧ x = y * z) → ¬(is_perfect_square y)

-- State the theorem
theorem sqrt_3_simplest :
  is_simplest_quadratic_radical 3 ∧
  ¬(is_simplest_quadratic_radical (1/2)) ∧
  ¬(is_simplest_quadratic_radical 8) ∧
  ¬(is_simplest_quadratic_radical 4) :=
sorry

end NUMINAMATH_CALUDE_sqrt_3_simplest_l3882_388257


namespace NUMINAMATH_CALUDE_continuous_at_8_l3882_388259

def f (x : ℝ) : ℝ := 5 * x^2 + 5

theorem continuous_at_8 :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 8| < δ → |f x - f 8| < ε := by
sorry

end NUMINAMATH_CALUDE_continuous_at_8_l3882_388259


namespace NUMINAMATH_CALUDE_special_polynomial_sum_l3882_388215

/-- A monic polynomial of degree 3 satisfying specific conditions -/
def special_polynomial (p : ℝ → ℝ) : Prop :=
  (∀ x y z : ℝ, p x = x^3 + y*x^2 + z*x + (7 - 6*y - 6*z)) ∧ 
  p 1 = 7 ∧ p 2 = 14 ∧ p 3 = 21

theorem special_polynomial_sum (p : ℝ → ℝ) (h : special_polynomial p) : 
  p 0 + p 5 = 53 := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_sum_l3882_388215


namespace NUMINAMATH_CALUDE_boys_percentage_l3882_388251

/-- Given a class with a 2:3 ratio of boys to girls and 30 total students,
    prove that 40% of the students are boys. -/
theorem boys_percentage (total_students : ℕ) (boy_girl_ratio : ℚ) : 
  total_students = 30 →
  boy_girl_ratio = 2 / 3 →
  (boy_girl_ratio / (1 + boy_girl_ratio)) * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_boys_percentage_l3882_388251


namespace NUMINAMATH_CALUDE_expression_value_l3882_388284

theorem expression_value (p q r s : ℝ) 
  (h1 : p^2 / q^3 = 4 / 5)
  (h2 : r^3 / s^2 = 7 / 9) :
  11 / (7 - r^3 / s^2) + (2 * q^3 - p^2) / (2 * q^3 + p^2) = 123 / 56 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3882_388284


namespace NUMINAMATH_CALUDE_officer_selection_count_l3882_388297

/-- Represents the number of members in the Robotics Club -/
def total_members : ℕ := 24

/-- Represents the number of officer positions to be filled -/
def officer_positions : ℕ := 4

/-- Represents the number of constrained pairs (Rachel and Samuel, Tim and Uma) -/
def constrained_pairs : ℕ := 2

/-- Calculates the number of ways to select officers given the constraints -/
def select_officers : ℕ := sorry

/-- Theorem stating that the number of ways to select officers is 126424 -/
theorem officer_selection_count :
  select_officers = 126424 := by sorry

end NUMINAMATH_CALUDE_officer_selection_count_l3882_388297


namespace NUMINAMATH_CALUDE_jack_recycling_earnings_l3882_388224

/-- The amount Jack gets per bottle in dollars -/
def bottle_amount : ℚ := sorry

/-- The amount Jack gets per can in dollars -/
def can_amount : ℚ := 5 / 100

/-- The number of bottles Jack recycled -/
def num_bottles : ℕ := 80

/-- The number of cans Jack recycled -/
def num_cans : ℕ := 140

/-- The total amount Jack made in dollars -/
def total_amount : ℚ := 15

theorem jack_recycling_earnings :
  bottle_amount * num_bottles + can_amount * num_cans = total_amount ∧
  bottle_amount = 1 / 10 := by sorry

end NUMINAMATH_CALUDE_jack_recycling_earnings_l3882_388224


namespace NUMINAMATH_CALUDE_evaluate_f_l3882_388264

def f (x : ℝ) : ℝ := 3 * x^2 + 5 * x - 7

theorem evaluate_f : 3 * f 2 - 2 * f (-2) = 55 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_f_l3882_388264


namespace NUMINAMATH_CALUDE_perpendicular_vectors_imply_k_zero_l3882_388293

/-- Given vectors a, b, and c in ℝ², prove that if a-c is perpendicular to b, then k = 0 -/
theorem perpendicular_vectors_imply_k_zero (a b c : ℝ × ℝ) (h : a.1 = 3 ∧ a.2 = 1) 
  (h' : b.1 = 1 ∧ b.2 = 3) (h'' : c.1 = k ∧ c.2 = 2) 
  (h''' : (a.1 - c.1) * b.1 + (a.2 - c.2) * b.2 = 0) : 
  k = 0 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_imply_k_zero_l3882_388293


namespace NUMINAMATH_CALUDE_complex_modulus_squared_l3882_388280

theorem complex_modulus_squared : Complex.abs (3/4 + 3*Complex.I)^2 = 153/16 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_squared_l3882_388280


namespace NUMINAMATH_CALUDE_correct_proposition_l3882_388254

-- Define proposition p₁
def p₁ : Prop := ∃ x₀ : ℝ, x₀^2 + x₀ + 1 < 0

-- Define proposition p₂
def p₂ : Prop := ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - 1 ≥ 0

-- Theorem to prove
theorem correct_proposition : (¬p₁) ∧ p₂ := by
  sorry

end NUMINAMATH_CALUDE_correct_proposition_l3882_388254


namespace NUMINAMATH_CALUDE_dihedral_angle_relationship_not_determined_l3882_388265

/-- Two dihedral angles with perpendicular half-planes -/
structure PerpendicularDihedralAngles where
  angle1 : ℝ
  angle2 : ℝ
  perpendicular_half_planes : Bool

/-- The relationship between the sizes of two dihedral angles with perpendicular half-planes is not determined -/
theorem dihedral_angle_relationship_not_determined (angles : PerpendicularDihedralAngles) :
  angles.perpendicular_half_planes →
  ¬(∀ a : PerpendicularDihedralAngles, a.angle1 = a.angle2) ∧
  ¬(∀ a : PerpendicularDihedralAngles, a.angle1 + a.angle2 = π) ∧
  ¬(∀ a : PerpendicularDihedralAngles, a.angle1 = a.angle2 ∨ a.angle1 + a.angle2 = π) :=
by sorry

end NUMINAMATH_CALUDE_dihedral_angle_relationship_not_determined_l3882_388265


namespace NUMINAMATH_CALUDE_knights_seating_probability_formula_l3882_388261

/-- The probability of three knights being seated at a round table with n chairs
    such that there is an empty chair on either side of each knight. -/
def knights_seating_probability (n : ℕ) : ℚ :=
  if n ≥ 6 then
    (n - 4) * (n - 5) / ((n - 1) * (n - 2))
  else
    0

/-- Theorem stating that the probability of three knights being seated at a round table
    with n chairs (where n ≥ 6) such that there is an empty chair on either side of
    each knight is equal to (n-4)(n-5) / ((n-1)(n-2)). -/
theorem knights_seating_probability_formula (n : ℕ) (h : n ≥ 6) :
  knights_seating_probability n = (n - 4) * (n - 5) / ((n - 1) * (n - 2)) :=
by sorry

end NUMINAMATH_CALUDE_knights_seating_probability_formula_l3882_388261


namespace NUMINAMATH_CALUDE_rosalina_gifts_l3882_388268

/-- The number of gifts Rosalina received from Emilio -/
def emilio_gifts : ℕ := 11

/-- The number of gifts Rosalina received from Jorge -/
def jorge_gifts : ℕ := 6

/-- The number of gifts Rosalina received from Pedro -/
def pedro_gifts : ℕ := 4

/-- The total number of gifts Rosalina received -/
def total_gifts : ℕ := emilio_gifts + jorge_gifts + pedro_gifts

theorem rosalina_gifts : total_gifts = 21 := by
  sorry

end NUMINAMATH_CALUDE_rosalina_gifts_l3882_388268


namespace NUMINAMATH_CALUDE_farm_chickens_count_l3882_388221

theorem farm_chickens_count (chicken_A duck_A chicken_B duck_B : ℕ) : 
  chicken_A + duck_A = 625 →
  chicken_B + duck_B = 748 →
  chicken_B = (chicken_A * 124) / 100 →
  duck_A = (duck_B * 85) / 100 →
  chicken_B = 248 := by
  sorry

end NUMINAMATH_CALUDE_farm_chickens_count_l3882_388221


namespace NUMINAMATH_CALUDE_function_sum_zero_at_five_sevenths_l3882_388249

-- Define the functions f and g
def f (x : ℝ) : ℝ := 5 * x - 6
def g (x : ℝ) : ℝ := 2 * x + 1

-- Theorem statement
theorem function_sum_zero_at_five_sevenths :
  ∃! a : ℝ, f a + g a = 0 ∧ a = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_function_sum_zero_at_five_sevenths_l3882_388249


namespace NUMINAMATH_CALUDE_sphere_segment_height_ratio_l3882_388230

/-- Given a sphere of radius R and a plane cutting a segment from it, 
    if the ratio of the segment's volume to the volume of a cone with 
    the same base and height is n, then the height h of the segment 
    is given by h = R / (3 - n), where n < 3 -/
theorem sphere_segment_height_ratio 
  (R : ℝ) 
  (n : ℝ) 
  (h : ℝ) 
  (hn : n < 3) 
  (hR : R > 0) :
  (π * R^2 * (h - R/3)) / ((1/3) * π * R^2 * h) = n → 
  h = R / (3 - n) :=
by sorry

end NUMINAMATH_CALUDE_sphere_segment_height_ratio_l3882_388230


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l3882_388234

def a : Fin 2 → ℝ := ![3, 2]

def v1 : Fin 2 → ℝ := ![3, -2]
def v2 : Fin 2 → ℝ := ![2, 3]
def v3 : Fin 2 → ℝ := ![-4, 6]
def v4 : Fin 2 → ℝ := ![-3, 2]

def dot_product (u v : Fin 2 → ℝ) : ℝ :=
  (u 0) * (v 0) + (u 1) * (v 1)

theorem perpendicular_vectors :
  dot_product a v1 ≠ 0 ∧
  dot_product a v2 ≠ 0 ∧
  dot_product a v3 = 0 ∧
  dot_product a v4 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l3882_388234


namespace NUMINAMATH_CALUDE_local_max_is_four_l3882_388231

/-- Given that x = 1 is a point of local minimum for f(x) = x³ - 3ax + 2,
    prove that the point of local maximum for f(x) is 4. -/
theorem local_max_is_four (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^3 - 3*a*x + 2
  (∀ h ∈ Set.Ioo (1 - ε) (1 + ε), f 1 ≤ f h) →
  ∃ ε > 0, ∀ h ∈ Set.Ioo (-1 - ε) (-1 + ε), f h ≤ f (-1) ∧ f (-1) = 4 :=
by sorry

end NUMINAMATH_CALUDE_local_max_is_four_l3882_388231


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_plus_four_l3882_388218

theorem square_plus_reciprocal_square_plus_four (m : ℝ) (h : m + 1/m = 10) :
  m^2 + 1/m^2 + 4 = 102 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_plus_four_l3882_388218


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l3882_388247

/-- A rectangle inscribed in a triangle -/
structure InscribedRectangle where
  /-- The length of the rectangle's side along the triangle's base -/
  base_length : ℝ
  /-- The width of the rectangle -/
  width : ℝ
  /-- The length of the triangle's base -/
  triangle_base : ℝ
  /-- The height of the triangle -/
  triangle_height : ℝ
  /-- The width is one-third of the base length -/
  width_constraint : width = base_length / 3
  /-- The triangle's base is 15 inches -/
  triangle_base_length : triangle_base = 15
  /-- The triangle's height is 12 inches -/
  triangle_height_value : triangle_height = 12

/-- The area of the inscribed rectangle -/
def area (r : InscribedRectangle) : ℝ := r.base_length * r.width

/-- Theorem: The area of the inscribed rectangle is 10800/289 square inches -/
theorem inscribed_rectangle_area (r : InscribedRectangle) : area r = 10800 / 289 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l3882_388247


namespace NUMINAMATH_CALUDE_pq_length_l3882_388242

/-- Given two lines and a point R that is the midpoint of a line segment PQ, 
    where P is on one line and Q is on the other, prove that the length of PQ 
    is √56512 / 33. -/
theorem pq_length (P Q R : ℝ × ℝ) : 
  R = (10, 8) →
  (∃ x, P = (x, 2*x)) →
  (∃ y, Q = (y, 4*y/11)) →
  R = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) →
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = Real.sqrt 56512 / 33 := by
  sorry

#check pq_length

end NUMINAMATH_CALUDE_pq_length_l3882_388242


namespace NUMINAMATH_CALUDE_angle_b_measure_l3882_388299

theorem angle_b_measure (A B C : ℝ) (h1 : A + B + C = 180) 
  (h2 : A = 3 * B) (h3 : B = 2 * C) : B = 40 := by
  sorry

end NUMINAMATH_CALUDE_angle_b_measure_l3882_388299


namespace NUMINAMATH_CALUDE_remaining_two_average_l3882_388279

theorem remaining_two_average (n₁ n₂ n₃ n₄ n₅ n₆ : ℝ) : 
  (n₁ + n₂ + n₃ + n₄ + n₅ + n₆) / 6 = 3.95 →
  (n₁ + n₂) / 2 = 3.4 →
  (n₃ + n₄) / 2 = 3.85 →
  (n₅ + n₆) / 2 = 4.6 := by
sorry

end NUMINAMATH_CALUDE_remaining_two_average_l3882_388279


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l3882_388203

/-- Given 50 observations with an initial mean, if one observation
    of 60 was wrongly recorded as 23, and the corrected mean is 36.5,
    then the initial mean is 35.76. -/
theorem initial_mean_calculation (n : ℕ) (M : ℝ) (wrong_value correct_value new_mean : ℝ) :
  n = 50 →
  wrong_value = 23 →
  correct_value = 60 →
  new_mean = 36.5 →
  ((n : ℝ) * M + (correct_value - wrong_value)) / n = new_mean →
  M = 35.76 := by
  sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l3882_388203


namespace NUMINAMATH_CALUDE_exponential_function_characterization_l3882_388281

/-- A function f is exponential if it satisfies f(x+y) = f(x)f(y) for all x and y -/
def IsExponential (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) = f x * f y

/-- A function f is monotonically increasing if f(x) ≤ f(y) whenever x ≤ y -/
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

theorem exponential_function_characterization (f : ℝ → ℝ) 
  (h_exp : IsExponential f) (h_mono : MonoIncreasing f) :
  ∃ a : ℝ, a > 1 ∧ (∀ x, f x = a^x) := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_characterization_l3882_388281


namespace NUMINAMATH_CALUDE_inequality_proof_l3882_388219

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a + 1/b > b + 1/a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3882_388219


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3882_388255

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x | -2 < x ∧ x < 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3882_388255


namespace NUMINAMATH_CALUDE_min_value_sequence_l3882_388201

theorem min_value_sequence (a : ℕ → ℝ) (m n : ℕ) :
  (∀ k, a k > 0) →  -- Positive sequence
  (∀ k, ∃ r, a (k + 1) = a k + r) →  -- Arithmetic progression
  (∀ k, ∃ q, a (k + 1) = a k * q) →  -- Geometric progression
  (a 7 = a 6 + 2 * a 5) →  -- Given condition
  (Real.sqrt (a m * a n) = 4 * a 1) →  -- Given condition
  (∃ min_val : ℝ, min_val = 1 + Real.sqrt 5 / 3 ∧
    ∀ p q : ℕ, 1 / p + 5 / q ≥ min_val) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sequence_l3882_388201


namespace NUMINAMATH_CALUDE_smallest_third_number_lcm_l3882_388209

/-- The lowest common multiple of a list of natural numbers -/
def lcm_list (l : List Nat) : Nat :=
  l.foldl Nat.lcm 1

/-- The theorem states that 10 is the smallest positive integer x
    such that the LCM of 24, 30, and x is 120 -/
theorem smallest_third_number_lcm :
  (∀ x : Nat, x > 0 → x < 10 → lcm_list [24, 30, x] ≠ 120) ∧
  lcm_list [24, 30, 10] = 120 := by
  sorry

end NUMINAMATH_CALUDE_smallest_third_number_lcm_l3882_388209


namespace NUMINAMATH_CALUDE_least_prime_factor_of_eight_cubed_minus_eight_squared_l3882_388270

theorem least_prime_factor_of_eight_cubed_minus_eight_squared :
  Nat.minFac (8^3 - 8^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_least_prime_factor_of_eight_cubed_minus_eight_squared_l3882_388270


namespace NUMINAMATH_CALUDE_shaded_region_area_l3882_388232

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a semicircle -/
structure Semicircle where
  center : Point
  radius : ℝ

/-- The shaded region formed by the intersection of three semicircles -/
def ShadedRegion (s1 s2 s3 : Semicircle) : Set Point := sorry

/-- The area of a set of points -/
def area (s : Set Point) : ℝ := sorry

/-- The midpoint of an arc -/
def arcMidpoint (s : Semicircle) : Point := sorry

theorem shaded_region_area 
  (s1 s2 s3 : Semicircle)
  (h1 : s1.radius = 2 ∧ s2.radius = 2 ∧ s3.radius = 2)
  (h2 : arcMidpoint s1 = s3.center)
  (h3 : arcMidpoint s2 = s3.center)
  (h4 : s3.center = arcMidpoint s3) :
  area (ShadedRegion s1 s2 s3) = 8 := by sorry

end NUMINAMATH_CALUDE_shaded_region_area_l3882_388232


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l3882_388291

/-- Given vectors in ℝ² -/
def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (2, -1)

/-- Function to check if two vectors are parallel -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), v.1 * w.2 = t * v.2 * w.1

/-- Main theorem -/
theorem parallel_vectors_k_value :
  ∃ (k : ℝ), are_parallel (a.1 + k * c.1, a.2 + k * c.2) (2 * b.1 - a.1, 2 * b.2 - a.2) ∧ k = 16 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l3882_388291


namespace NUMINAMATH_CALUDE_function_properties_l3882_388233

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := a^x - a + 1

def g (a : ℝ) (x : ℝ) : ℝ := f a (x + 1/2) - 1

def F (a m : ℝ) (x : ℝ) : ℝ := g a (2*x) - m * g a (x - 1)

def h (m : ℝ) : ℝ :=
  if m ≤ 1 then 1 - 2*m
  else if m < 2 then -m^2
  else 4 - 4*m

theorem function_properties (a : ℝ) (ha : a > 0 ∧ a ≠ 1) (hf : f a (1/2) = 2) :
  a = 1/2 ∧
  (∀ x, g a x = (1/2)^x) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 0, F a m x ≥ h m) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l3882_388233


namespace NUMINAMATH_CALUDE_intersection_complement_equal_l3882_388205

open Set

def U : Finset Nat := {1,2,3,4,5}
def M : Finset Nat := {1,4}
def N : Finset Nat := {1,3,5}

theorem intersection_complement_equal : N ∩ (U \ M) = {3,5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equal_l3882_388205


namespace NUMINAMATH_CALUDE_goose_eggs_theorem_l3882_388296

theorem goose_eggs_theorem (total_eggs : ℕ) : 
  (1 : ℚ) / 4 * (4 : ℚ) / 5 * (3 : ℚ) / 5 * total_eggs = 120 →
  total_eggs = 1000 := by
  sorry

end NUMINAMATH_CALUDE_goose_eggs_theorem_l3882_388296


namespace NUMINAMATH_CALUDE_theater_sales_result_l3882_388271

/-- Calculates the total amount collected from ticket sales for a theater performance -/
def theater_sales (adult_price child_price total_attendees children_attendees : ℕ) : ℕ :=
  let adults := total_attendees - children_attendees
  adult_price * adults + child_price * children_attendees

/-- Theorem stating that the theater collected $258 from ticket sales -/
theorem theater_sales_result : theater_sales 16 9 24 18 = 258 := by
  sorry

end NUMINAMATH_CALUDE_theater_sales_result_l3882_388271


namespace NUMINAMATH_CALUDE_soda_distribution_l3882_388239

theorem soda_distribution (boxes : Nat) (cans_per_box : Nat) (discarded : Nat) (cartons : Nat) :
  boxes = 7 →
  cans_per_box = 16 →
  discarded = 13 →
  cartons = 8 →
  (boxes * cans_per_box - discarded) % cartons = 3 := by
  sorry

end NUMINAMATH_CALUDE_soda_distribution_l3882_388239


namespace NUMINAMATH_CALUDE_counterexample_exists_l3882_388267

theorem counterexample_exists : ∃ n : ℕ+, ¬(Nat.Prime (6 * n - 1)) ∧ ¬(Nat.Prime (6 * n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l3882_388267
