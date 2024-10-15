import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_and_fractional_equations_l3893_389314

theorem quadratic_and_fractional_equations :
  (∃ x₁ x₂ : ℝ, x₁ = 1 + Real.sqrt 5 ∧ x₂ = 1 - Real.sqrt 5 ∧
    x₁^2 - 2*x₁ - 4 = 0 ∧ x₂^2 - 2*x₂ - 4 = 0) ∧
  (∀ x : ℝ, x ≠ 4 → ((x - 5) / (x - 4) = 1 - x / (4 - x)) ↔ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_and_fractional_equations_l3893_389314


namespace NUMINAMATH_CALUDE_power_multiplication_l3893_389331

theorem power_multiplication (x : ℝ) : x^2 * x^3 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3893_389331


namespace NUMINAMATH_CALUDE_power_sum_equality_l3893_389322

theorem power_sum_equality : (-2)^1999 + (-2)^2000 = 2^1999 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l3893_389322


namespace NUMINAMATH_CALUDE_clown_balloons_l3893_389344

theorem clown_balloons (initial_balloons : ℕ) : 
  initial_balloons + 13 = 60 → initial_balloons = 47 := by
  sorry

end NUMINAMATH_CALUDE_clown_balloons_l3893_389344


namespace NUMINAMATH_CALUDE_sum_reciprocal_n_n_plus_three_l3893_389300

open Real

/-- The sum of the infinite series Σ(1/(n(n+3))) from n=1 to infinity equals 11/18 -/
theorem sum_reciprocal_n_n_plus_three : 
  ∑' n : ℕ+, (1 : ℝ) / (n * (n + 3)) = 11/18 := by sorry

end NUMINAMATH_CALUDE_sum_reciprocal_n_n_plus_three_l3893_389300


namespace NUMINAMATH_CALUDE_one_millionth_digit_of_3_div_41_l3893_389396

/-- The decimal representation of a rational number -/
def decimal_representation (q : ℚ) : ℕ → ℕ := sorry

/-- The nth digit after the decimal point in the decimal representation of q -/
def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := 
  decimal_representation q n

/-- The one-millionth digit after the decimal point in 3/41 is 7 -/
theorem one_millionth_digit_of_3_div_41 : 
  nth_digit_after_decimal (3/41) 1000000 = 7 := by sorry

end NUMINAMATH_CALUDE_one_millionth_digit_of_3_div_41_l3893_389396


namespace NUMINAMATH_CALUDE_correct_yeast_experiment_methods_l3893_389399

/-- Represents the method used for counting yeast -/
inductive CountingMethod
| SamplingInspection
| Other

/-- Represents the action taken before extracting culture fluid -/
inductive PreExtractionAction
| GentlyShake
| Other

/-- Represents the measure taken when there are too many yeast cells -/
inductive OvercrowdingMeasure
| AppropriateDilution
| Other

/-- Represents the conditions of the yeast counting experiment -/
structure YeastExperiment where
  countingMethod : CountingMethod
  preExtractionAction : PreExtractionAction
  overcrowdingMeasure : OvercrowdingMeasure

/-- Theorem stating the correct methods and actions for the yeast counting experiment -/
theorem correct_yeast_experiment_methods :
  ∀ (experiment : YeastExperiment),
    experiment.countingMethod = CountingMethod.SamplingInspection ∧
    experiment.preExtractionAction = PreExtractionAction.GentlyShake ∧
    experiment.overcrowdingMeasure = OvercrowdingMeasure.AppropriateDilution :=
by sorry

end NUMINAMATH_CALUDE_correct_yeast_experiment_methods_l3893_389399


namespace NUMINAMATH_CALUDE_total_earnings_theorem_l3893_389385

/-- Represents the different car models --/
inductive CarModel
| A
| B
| C
| D

/-- Represents the different services offered --/
inductive Service
| OilChange
| Repair
| CarWash
| TireRotation

/-- Returns the price of a service for a given car model --/
def servicePrice (model : CarModel) (service : Service) : ℕ :=
  match model, service with
  | CarModel.A, Service.OilChange => 20
  | CarModel.A, Service.Repair => 30
  | CarModel.A, Service.CarWash => 5
  | CarModel.A, Service.TireRotation => 15
  | CarModel.B, Service.OilChange => 25
  | CarModel.B, Service.Repair => 40
  | CarModel.B, Service.CarWash => 8
  | CarModel.B, Service.TireRotation => 18
  | CarModel.C, Service.OilChange => 30
  | CarModel.C, Service.Repair => 50
  | CarModel.C, Service.CarWash => 10
  | CarModel.C, Service.TireRotation => 20
  | CarModel.D, Service.OilChange => 35
  | CarModel.D, Service.Repair => 60
  | CarModel.D, Service.CarWash => 12
  | CarModel.D, Service.TireRotation => 22

/-- Applies discount if the number of services is 3 or more --/
def applyDiscount (total : ℕ) (numServices : ℕ) : ℕ :=
  if numServices ≥ 3 then
    total - (total * 10 / 100)
  else
    total

/-- Calculates the total price for a car model with given services --/
def totalPrice (model : CarModel) (services : List Service) : ℕ :=
  let total := services.foldl (fun acc service => acc + servicePrice model service) 0
  applyDiscount total services.length

/-- The main theorem to prove --/
theorem total_earnings_theorem :
  let modelA_services := [Service.OilChange, Service.Repair, Service.CarWash]
  let modelB_services := [Service.OilChange, Service.Repair, Service.CarWash, Service.TireRotation]
  let modelC_services := [Service.OilChange, Service.Repair, Service.TireRotation, Service.CarWash]
  let modelD_services := [Service.OilChange, Service.Repair, Service.TireRotation]
  
  5 * (totalPrice CarModel.A modelA_services) +
  3 * (totalPrice CarModel.B modelB_services) +
  2 * (totalPrice CarModel.C modelC_services) +
  4 * (totalPrice CarModel.D modelD_services) = 111240 :=
by sorry


end NUMINAMATH_CALUDE_total_earnings_theorem_l3893_389385


namespace NUMINAMATH_CALUDE_rectangular_box_dimensions_l3893_389359

theorem rectangular_box_dimensions (X Y Z : ℝ) 
  (h1 : X * Y = 32)
  (h2 : X * Z = 50)
  (h3 : Y * Z = 80) :
  X + Y + Z = 25.5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_dimensions_l3893_389359


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l3893_389397

/-- A binary operation on nonzero real numbers satisfying certain properties -/
def diamond (a b : ℝ) : ℝ := sorry

/-- The binary operation satisfies a ♢ (b ♢ c) = (a ♢ b) · c -/
axiom diamond_assoc (a b c : ℝ) : a ≠ 0 → b ≠ 0 → c ≠ 0 → diamond a (diamond b c) = (diamond a b) * c

/-- The binary operation satisfies a ♢ a = 1 -/
axiom diamond_self (a : ℝ) : a ≠ 0 → diamond a a = 1

/-- The equation 1008 ♢ (12 ♢ x) = 50 is satisfied when x = 25/42 -/
theorem diamond_equation_solution :
  1008 ≠ 0 → 12 ≠ 0 → (25 : ℝ) / 42 ≠ 0 → diamond 1008 (diamond 12 ((25 : ℝ) / 42)) = 50 := by sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l3893_389397


namespace NUMINAMATH_CALUDE_x_divisibility_l3893_389347

def x : ℤ := 64 + 96 + 128 + 160 + 288 + 352 + 3232

theorem x_divisibility :
  (∃ k : ℤ, x = 4 * k) ∧
  (∃ k : ℤ, x = 8 * k) ∧
  (∃ k : ℤ, x = 16 * k) ∧
  (∃ k : ℤ, x = 32 * k) :=
by sorry

end NUMINAMATH_CALUDE_x_divisibility_l3893_389347


namespace NUMINAMATH_CALUDE_bottle_cost_difference_l3893_389369

/-- Represents a bottle of capsules -/
structure Bottle where
  capsules : ℕ
  cost : ℚ

/-- Calculate the cost per capsule for a given bottle -/
def costPerCapsule (b : Bottle) : ℚ :=
  b.cost / b.capsules

/-- The difference in cost per capsule between two bottles -/
def costDifference (b1 b2 : Bottle) : ℚ :=
  costPerCapsule b1 - costPerCapsule b2

theorem bottle_cost_difference :
  let bottleR : Bottle := { capsules := 250, cost := 25/4 }
  let bottleT : Bottle := { capsules := 100, cost := 3 }
  costDifference bottleT bottleR = 1/200 := by
sorry

end NUMINAMATH_CALUDE_bottle_cost_difference_l3893_389369


namespace NUMINAMATH_CALUDE_sine_domain_range_constraint_l3893_389338

theorem sine_domain_range_constraint (a b : Real) : 
  (∀ x ∈ Set.Icc a b, -1 ≤ Real.sin x ∧ Real.sin x ≤ 1/2) →
  (∃ x ∈ Set.Icc a b, Real.sin x = -1) →
  (∃ x ∈ Set.Icc a b, Real.sin x = 1/2) →
  b - a ≠ π/3 := by
  sorry

end NUMINAMATH_CALUDE_sine_domain_range_constraint_l3893_389338


namespace NUMINAMATH_CALUDE_exponent_sum_l3893_389351

theorem exponent_sum (a : ℝ) (m n : ℕ) (h1 : a^m = 2) (h2 : a^n = 8) : a^(m+n) = 16 := by
  sorry

end NUMINAMATH_CALUDE_exponent_sum_l3893_389351


namespace NUMINAMATH_CALUDE_divisible_by_36_sum_6_l3893_389317

/-- Represents a 7-digit number in the form 457q89f -/
def number (q f : Nat) : Nat :=
  457000 + q * 1000 + 89 * 10 + f

/-- Predicate to check if two natural numbers are distinct digits -/
def distinct_digits (a b : Nat) : Prop :=
  a ≠ b ∧ a < 10 ∧ b < 10

theorem divisible_by_36_sum_6 (q f : Nat) :
  distinct_digits q f →
  number q f % 36 = 0 →
  q + f = 6 := by
sorry

end NUMINAMATH_CALUDE_divisible_by_36_sum_6_l3893_389317


namespace NUMINAMATH_CALUDE_total_letters_is_68_l3893_389342

/-- The total number of letters in all siblings' names -/
def total_letters : ℕ :=
  let jonathan_first := 8
  let jonathan_last := 10
  let younger_sister_first := 5
  let younger_sister_last := 10
  let older_brother_first := 6
  let older_brother_last := 10
  let youngest_sibling_first := 4
  let youngest_sibling_last := 15
  (jonathan_first + jonathan_last) +
  (younger_sister_first + younger_sister_last) +
  (older_brother_first + older_brother_last) +
  (youngest_sibling_first + youngest_sibling_last)

/-- Theorem stating that the total number of letters in all siblings' names is 68 -/
theorem total_letters_is_68 : total_letters = 68 := by
  sorry

end NUMINAMATH_CALUDE_total_letters_is_68_l3893_389342


namespace NUMINAMATH_CALUDE_sphere_volume_is_10936_l3893_389375

/-- The volume of a small hemisphere container in liters -/
def small_hemisphere_volume : ℝ := 4

/-- The number of small hemisphere containers required -/
def num_hemispheres : ℕ := 2734

/-- The total volume of water in the sphere container in liters -/
def sphere_volume : ℝ := small_hemisphere_volume * num_hemispheres

/-- Theorem stating that the total volume of water in the sphere container is 10936 liters -/
theorem sphere_volume_is_10936 : sphere_volume = 10936 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_is_10936_l3893_389375


namespace NUMINAMATH_CALUDE_least_questions_for_probability_l3893_389327

theorem least_questions_for_probability (n : ℕ) : n ≥ 4 ↔ (1/2 : ℝ)^n < 1/10 := by sorry

end NUMINAMATH_CALUDE_least_questions_for_probability_l3893_389327


namespace NUMINAMATH_CALUDE_geometry_test_passing_l3893_389389

theorem geometry_test_passing (total_problems : ℕ) (passing_percentage : ℚ) 
  (hp : passing_percentage = 85 / 100) (ht : total_problems = 50) : 
  ∃ (max_missed : ℕ), 
    (((total_problems - max_missed : ℚ) / total_problems) ≥ passing_percentage ∧ 
     ∀ (n : ℕ), n > max_missed → 
       ((total_problems - n : ℚ) / total_problems) < passing_percentage) ∧
    max_missed = 7 :=
sorry

end NUMINAMATH_CALUDE_geometry_test_passing_l3893_389389


namespace NUMINAMATH_CALUDE_max_a_for_inequality_l3893_389363

theorem max_a_for_inequality : ∃ (a : ℝ), ∀ (x : ℝ), |x - 2| + |x - 8| ≥ a ∧ ∀ (b : ℝ), (∀ (y : ℝ), |y - 2| + |y - 8| ≥ b) → b ≤ a :=
by sorry

end NUMINAMATH_CALUDE_max_a_for_inequality_l3893_389363


namespace NUMINAMATH_CALUDE_cherry_pitting_time_l3893_389302

/-- Time required to pit cherries for a pie --/
theorem cherry_pitting_time
  (pounds_needed : ℕ)
  (cherries_per_pound : ℕ)
  (pitting_time : ℕ)
  (cherries_per_batch : ℕ)
  (h1 : pounds_needed = 3)
  (h2 : cherries_per_pound = 80)
  (h3 : pitting_time = 10)
  (h4 : cherries_per_batch = 20) :
  (pounds_needed * cherries_per_pound * pitting_time) / (cherries_per_batch * 60) = 2 :=
by sorry

end NUMINAMATH_CALUDE_cherry_pitting_time_l3893_389302


namespace NUMINAMATH_CALUDE_total_marbles_l3893_389349

def jungkook_marbles : ℕ := 3
def marble_difference : ℕ := 4

def jimin_marbles : ℕ := jungkook_marbles + marble_difference

theorem total_marbles :
  jungkook_marbles + jimin_marbles = 10 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_l3893_389349


namespace NUMINAMATH_CALUDE_solve_birthday_money_problem_l3893_389352

def birthday_money_problem (aunt uncle friend1 friend2 friend3 sister : ℝ)
  (mean : ℝ) (total_gifts : ℕ) (unknown_gift : ℝ) : Prop :=
  aunt = 9 ∧
  uncle = 9 ∧
  friend1 = 22 ∧
  friend2 = 22 ∧
  friend3 = 22 ∧
  sister = 7 ∧
  mean = 16.3 ∧
  total_gifts = 7 ∧
  (aunt + uncle + friend1 + unknown_gift + friend2 + friend3 + sister) / total_gifts = mean ∧
  unknown_gift = 23.1

theorem solve_birthday_money_problem :
  ∃ (aunt uncle friend1 friend2 friend3 sister : ℝ)
    (mean : ℝ) (total_gifts : ℕ) (unknown_gift : ℝ),
  birthday_money_problem aunt uncle friend1 friend2 friend3 sister mean total_gifts unknown_gift :=
by sorry

end NUMINAMATH_CALUDE_solve_birthday_money_problem_l3893_389352


namespace NUMINAMATH_CALUDE_twelve_months_probability_l3893_389372

/-- Represents the card game "Twelve Months" -/
structure TwelveMonths where
  /-- Number of columns -/
  n : Nat
  /-- Number of cards per column -/
  m : Nat
  /-- Total number of cards -/
  total_cards : Nat
  /-- Condition: total cards equals m * n -/
  h_total : total_cards = m * n

/-- The probability of all cards being flipped in the "Twelve Months" game -/
def probability_all_flipped (game : TwelveMonths) : ℚ :=
  1 / game.n

/-- Theorem stating the probability of all cards being flipped in the "Twelve Months" game -/
theorem twelve_months_probability (game : TwelveMonths) 
  (h_columns : game.n = 12) 
  (h_cards_per_column : game.m = 4) : 
  probability_all_flipped game = 1 / 12 := by
  sorry

#eval probability_all_flipped ⟨12, 4, 48, rfl⟩

end NUMINAMATH_CALUDE_twelve_months_probability_l3893_389372


namespace NUMINAMATH_CALUDE_special_tetrahedron_edges_l3893_389334

/-- A tetrahedron with congruent triangular faces, each having one 60° angle,
    inscribed in a sphere of diameter 23 cm. -/
structure SpecialTetrahedron where
  -- Edge lengths
  a : ℕ
  b : ℕ
  c : ℕ
  -- Conditions
  congruent_faces : a^2 + b^2 - a*b/2 = c^2
  circumsphere_diameter : a^2 + b^2 + c^2 = 2 * 23^2

/-- The edge lengths of the special tetrahedron are 16 cm, 19 cm, and 21 cm. -/
theorem special_tetrahedron_edges :
  ∃ (t : SpecialTetrahedron), t.a = 16 ∧ t.b = 21 ∧ t.c = 19 :=
by sorry

end NUMINAMATH_CALUDE_special_tetrahedron_edges_l3893_389334


namespace NUMINAMATH_CALUDE_blocks_differing_in_two_ways_l3893_389378

/-- Represents the properties of a block -/
structure Block where
  material : Fin 3
  size : Fin 3
  color : Fin 4
  shape : Fin 5

/-- The set of all blocks -/
def all_blocks : Finset Block := sorry

/-- The reference block (wood small blue hexagon) -/
def reference_block : Block := ⟨0, 0, 0, 1⟩

/-- The number of ways a block differs from the reference block -/
def diff_count (b : Block) : Nat := sorry

/-- Theorem: The number of blocks differing in exactly 2 ways from the reference block is 44 -/
theorem blocks_differing_in_two_ways :
  (all_blocks.filter (λ b => diff_count b = 2)).card = 44 := by sorry

end NUMINAMATH_CALUDE_blocks_differing_in_two_ways_l3893_389378


namespace NUMINAMATH_CALUDE_vacation_days_calculation_l3893_389354

theorem vacation_days_calculation (families : Nat) (people_per_family : Nat) 
  (towels_per_person_per_day : Nat) (towels_per_load : Nat) (total_loads : Nat) :
  families = 3 →
  people_per_family = 4 →
  towels_per_person_per_day = 1 →
  towels_per_load = 14 →
  total_loads = 6 →
  (total_loads * towels_per_load) / (families * people_per_family * towels_per_person_per_day) = 7 := by
  sorry

end NUMINAMATH_CALUDE_vacation_days_calculation_l3893_389354


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3893_389388

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}

theorem intersection_of_A_and_B :
  A ∩ B = {x | 0 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3893_389388


namespace NUMINAMATH_CALUDE_similar_polygons_perimeter_ratio_l3893_389330

/-- If the ratio of the areas of two similar polygons is 4:9, then the ratio of their perimeters is 2:3 -/
theorem similar_polygons_perimeter_ratio (A B : ℝ) (P Q : ℝ) 
  (h_area : A / B = 4 / 9) (h_positive : A > 0 ∧ B > 0 ∧ P > 0 ∧ Q > 0)
  (h_area_perimeter : A / B = (P / Q)^2) : P / Q = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_similar_polygons_perimeter_ratio_l3893_389330


namespace NUMINAMATH_CALUDE_apps_difference_is_three_l3893_389383

/-- The difference between apps added and deleted -/
def appsDifference (initial final added : ℕ) : ℕ :=
  added - (initial + added - final)

/-- Proof that the difference between apps added and deleted is 3 -/
theorem apps_difference_is_three : appsDifference 21 24 89 = 3 := by
  sorry

end NUMINAMATH_CALUDE_apps_difference_is_three_l3893_389383


namespace NUMINAMATH_CALUDE_two_digit_numbers_problem_l3893_389382

theorem two_digit_numbers_problem :
  ∃ (x y : ℕ), 10 ≤ x ∧ x < y ∧ y < 100 ∧
  (1000 * y + x) % (100 * x + y) = 590 ∧
  (1000 * y + x) / (100 * x + y) = 2 ∧
  2 * y + 3 * x = 72 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_numbers_problem_l3893_389382


namespace NUMINAMATH_CALUDE_f_log2_32_equals_17_l3893_389323

noncomputable def f (x : ℝ) : ℝ :=
  if x < 4 then Real.log 4 / Real.log 2
  else 1 + 2^(x - 1)

theorem f_log2_32_equals_17 : f (Real.log 32 / Real.log 2) = 17 := by
  sorry

end NUMINAMATH_CALUDE_f_log2_32_equals_17_l3893_389323


namespace NUMINAMATH_CALUDE_barbi_monthly_loss_is_one_point_five_l3893_389332

/-- Represents the weight loss scenario of Barbi and Luca -/
structure WeightLossScenario where
  barbi_monthly_loss : ℝ
  months_in_year : ℕ
  luca_yearly_loss : ℝ
  luca_years : ℕ
  difference : ℝ

/-- The weight loss scenario satisfies the given conditions -/
def satisfies_conditions (scenario : WeightLossScenario) : Prop :=
  scenario.months_in_year = 12 ∧
  scenario.luca_yearly_loss = 9 ∧
  scenario.luca_years = 11 ∧
  scenario.difference = 81 ∧
  scenario.luca_yearly_loss * scenario.luca_years = 
    scenario.barbi_monthly_loss * scenario.months_in_year + scenario.difference

/-- Theorem stating that under the given conditions, Barbi's monthly weight loss is 1.5 kg -/
theorem barbi_monthly_loss_is_one_point_five 
  (scenario : WeightLossScenario) 
  (h : satisfies_conditions scenario) : 
  scenario.barbi_monthly_loss = 1.5 := by
  sorry


end NUMINAMATH_CALUDE_barbi_monthly_loss_is_one_point_five_l3893_389332


namespace NUMINAMATH_CALUDE_tom_seashells_l3893_389335

/-- The number of seashells Tom found yesterday -/
def seashells_yesterday : ℕ := 7

/-- The number of seashells Tom found today -/
def seashells_today : ℕ := 4

/-- The total number of seashells Tom found -/
def total_seashells : ℕ := seashells_yesterday + seashells_today

/-- Proof that the total number of seashells Tom found is 11 -/
theorem tom_seashells : total_seashells = 11 := by
  sorry

end NUMINAMATH_CALUDE_tom_seashells_l3893_389335


namespace NUMINAMATH_CALUDE_chrysanthemum_pots_count_l3893_389343

/-- The total number of chrysanthemum pots -/
def total_pots : ℕ := 360

/-- The number of rows after transportation -/
def remaining_rows : ℕ := 9

/-- The number of pots in each row -/
def pots_per_row : ℕ := 20

/-- Theorem stating that the total number of chrysanthemum pots is 360 -/
theorem chrysanthemum_pots_count :
  total_pots = 2 * remaining_rows * pots_per_row :=
by sorry

end NUMINAMATH_CALUDE_chrysanthemum_pots_count_l3893_389343


namespace NUMINAMATH_CALUDE_single_loop_probability_six_threads_l3893_389339

/-- Represents the game with threads and pairings -/
structure ThreadGame where
  num_threads : ℕ
  num_pairs : ℕ

/-- Calculates the total number of possible pairings -/
def total_pairings (game : ThreadGame) : ℕ :=
  (2 * game.num_threads - 1) * (2 * game.num_threads - 3)

/-- Calculates the number of pairings that form a single loop -/
def single_loop_pairings (game : ThreadGame) : ℕ :=
  (2 * game.num_threads - 2) * (game.num_threads - 1)

/-- Theorem stating the probability of forming a single loop in the game with 6 threads -/
theorem single_loop_probability_six_threads :
  let game : ThreadGame := { num_threads := 6, num_pairs := 3 }
  (single_loop_pairings game : ℚ) / (total_pairings game) = 8 / 15 := by
  sorry


end NUMINAMATH_CALUDE_single_loop_probability_six_threads_l3893_389339


namespace NUMINAMATH_CALUDE_cube_painting_theorem_l3893_389324

/-- The number of faces on a cube -/
def num_faces : ℕ := 6

/-- The number of available colors -/
def num_colors : ℕ := 6

/-- The number of rotational symmetries of a cube -/
def cube_symmetries : ℕ := 24

/-- The number of ways to paint all faces of a cube the same color -/
def all_same_color : ℕ := num_colors

/-- The number of ways to paint 5 faces the same color and 1 face a different color -/
def five_same_one_different : ℕ := num_faces * (num_colors - 1)

/-- The number of ways to paint all faces of a cube different colors, considering rotational symmetry -/
def all_different_colors : ℕ := (Nat.factorial num_colors) / cube_symmetries

theorem cube_painting_theorem :
  all_same_color = 6 ∧
  five_same_one_different = 30 ∧
  all_different_colors = 30 := by
  sorry

end NUMINAMATH_CALUDE_cube_painting_theorem_l3893_389324


namespace NUMINAMATH_CALUDE_second_term_is_five_l3893_389376

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℝ) (d : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => arithmetic_sequence a d n + d

-- Theorem statement
theorem second_term_is_five
  (a d : ℝ)
  (h : arithmetic_sequence a d 0 + arithmetic_sequence a d 2 = 10) :
  arithmetic_sequence a d 1 = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_second_term_is_five_l3893_389376


namespace NUMINAMATH_CALUDE_diesel_cost_approximation_l3893_389305

/-- Calculates the approximate average cost of diesel per litre over three years -/
def average_diesel_cost (price1 price2 price3 yearly_spend : ℚ) : ℚ :=
  let litres1 := yearly_spend / price1
  let litres2 := yearly_spend / price2
  let litres3 := yearly_spend / price3
  let total_litres := litres1 + litres2 + litres3
  let total_spent := 3 * yearly_spend
  total_spent / total_litres

/-- Theorem stating that the average diesel cost is approximately 8.98 given the specified conditions -/
theorem diesel_cost_approximation :
  let price1 : ℚ := 8.5
  let price2 : ℚ := 9
  let price3 : ℚ := 9.5
  let yearly_spend : ℚ := 5000
  let result := average_diesel_cost price1 price2 price3 yearly_spend
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ |result - 8.98| < ε :=
sorry

end NUMINAMATH_CALUDE_diesel_cost_approximation_l3893_389305


namespace NUMINAMATH_CALUDE_sequence_converges_to_ones_l3893_389366

/-- The operation S applied to a sequence -/
def S (a : Fin (2^n) → Int) : Fin (2^n) → Int :=
  fun i => a i * a (i.succ)

/-- The result of applying S v times -/
def applyS (a : Fin (2^n) → Int) (v : Nat) : Fin (2^n) → Int :=
  match v with
  | 0 => a
  | v+1 => S (applyS a v)

theorem sequence_converges_to_ones 
  (n : Nat) (a : Fin (2^n) → Int) 
  (h : ∀ i, a i = 1 ∨ a i = -1) : 
  ∀ i, applyS a (2^n) i = 1 := by
  sorry

#check sequence_converges_to_ones

end NUMINAMATH_CALUDE_sequence_converges_to_ones_l3893_389366


namespace NUMINAMATH_CALUDE_cube_equation_solution_l3893_389320

theorem cube_equation_solution (c : ℤ) : 
  c^3 + 3*c + 3/c + 1/c^3 = 8 → c = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l3893_389320


namespace NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l3893_389318

theorem integer_pairs_satisfying_equation :
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (p : ℤ × ℤ), p ∈ s ↔ p.1 + p.2 = p.1 * p.2 - 2) ∧ 
    s.card = 6 := by
  sorry

end NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l3893_389318


namespace NUMINAMATH_CALUDE_additional_rate_calculation_l3893_389370

/-- Telephone company charging model -/
structure TelephoneCharge where
  initial_rate : ℚ  -- Rate for the first 1/5 minute in cents
  additional_rate : ℚ  -- Rate for each additional 1/5 minute in cents

/-- Calculate the total charge for a given duration -/
def total_charge (model : TelephoneCharge) (duration : ℚ) : ℚ :=
  model.initial_rate + (duration * 5 - 1) * model.additional_rate

theorem additional_rate_calculation (model : TelephoneCharge) 
  (h1 : model.initial_rate = 310/100)  -- 3.10 cents for the first 1/5 minute
  (h2 : total_charge model (8 : ℚ) = 1870/100)  -- 18.70 cents for 8 minutes
  : model.additional_rate = 40/100 := by
  sorry

end NUMINAMATH_CALUDE_additional_rate_calculation_l3893_389370


namespace NUMINAMATH_CALUDE_reappearance_line_is_lcm_l3893_389357

/-- The cycle length of the letter sequence -/
def letter_cycle_length : ℕ := 8

/-- The cycle length of the digit sequence -/
def digit_cycle_length : ℕ := 4

/-- The line number where the original sequences reappear -/
def reappearance_line : ℕ := 8

/-- Theorem stating that the reappearance line is the least common multiple of the cycle lengths -/
theorem reappearance_line_is_lcm :
  reappearance_line = Nat.lcm letter_cycle_length digit_cycle_length :=
by sorry

end NUMINAMATH_CALUDE_reappearance_line_is_lcm_l3893_389357


namespace NUMINAMATH_CALUDE_square_side_lengths_average_l3893_389321

theorem square_side_lengths_average (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 16) (h₂ : a₂ = 49) (h₃ : a₃ = 169) :
  (Real.sqrt a₁ + Real.sqrt a₂ + Real.sqrt a₃) / 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_side_lengths_average_l3893_389321


namespace NUMINAMATH_CALUDE_milk_students_l3893_389368

theorem milk_students (juice_students : ℕ) (juice_angle : ℝ) (total_angle : ℝ) :
  juice_students = 80 →
  juice_angle = 90 →
  total_angle = 360 →
  (juice_angle / total_angle) * (juice_students + (total_angle - juice_angle) / juice_angle * juice_students) = 240 :=
by sorry

end NUMINAMATH_CALUDE_milk_students_l3893_389368


namespace NUMINAMATH_CALUDE_project_completion_time_l3893_389373

theorem project_completion_time 
  (initial_people : ℕ) 
  (initial_days : ℕ) 
  (additional_people : ℕ) 
  (h1 : initial_people = 12)
  (h2 : initial_days = 15)
  (h3 : additional_people = 8) : 
  (initial_days + (initial_people * initial_days * 2) / (initial_people + additional_people)) = 33 :=
by sorry

end NUMINAMATH_CALUDE_project_completion_time_l3893_389373


namespace NUMINAMATH_CALUDE_savings_equality_l3893_389316

/-- Prove that A's savings equal B's savings given the conditions -/
theorem savings_equality (total_salary : ℝ) (a_salary : ℝ) (a_spend_rate : ℝ) (b_spend_rate : ℝ)
  (h1 : total_salary = 7000)
  (h2 : a_salary = 5250)
  (h3 : a_spend_rate = 0.95)
  (h4 : b_spend_rate = 0.85) :
  a_salary * (1 - a_spend_rate) = (total_salary - a_salary) * (1 - b_spend_rate) :=
by
  sorry

end NUMINAMATH_CALUDE_savings_equality_l3893_389316


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l3893_389365

/-- Properties of a quadratic equation -/
theorem quadratic_equation_properties
  (a b c : ℝ) (h_a : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  -- Statement 1
  (∀ x, f x = 0 ↔ x = 1 ∨ x = 2) → 2 * a - c = 0 ∧
  -- Statement 2
  (b = 2 * a + c → b^2 - 4 * a * c > 0) ∧
  -- Statement 3
  (∀ m, f m = 0 → b^2 - 4 * a * c = (2 * a * m + b)^2) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_equation_properties_l3893_389365


namespace NUMINAMATH_CALUDE_percentage_difference_l3893_389325

theorem percentage_difference : 
  (0.80 * 40) - ((4 / 5) * 20) = 16 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l3893_389325


namespace NUMINAMATH_CALUDE_tea_cost_price_l3893_389374

/-- The cost price per kg of the 80 kg of tea -/
def x : ℝ := 15

/-- The theorem stating that the cost price per kg of the 80 kg of tea is 15 -/
theorem tea_cost_price : 
  ∀ (quantity_1 quantity_2 cost_2 profit sale_price : ℝ),
  quantity_1 = 80 →
  quantity_2 = 20 →
  cost_2 = 20 →
  profit = 0.2 →
  sale_price = 19.2 →
  x = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_tea_cost_price_l3893_389374


namespace NUMINAMATH_CALUDE_tree_planting_solution_l3893_389303

/-- Represents the tree planting problem during Arbor Day -/
structure TreePlanting where
  students : ℕ
  typeA : ℕ
  typeB : ℕ

/-- The conditions of the tree planting problem -/
def valid_tree_planting (tp : TreePlanting) : Prop :=
  3 * tp.students + 20 = tp.typeA + tp.typeB ∧
  4 * tp.students = tp.typeA + tp.typeB + 25 ∧
  30 * tp.typeA + 40 * tp.typeB ≤ 5400

/-- The theorem stating the solution to the tree planting problem -/
theorem tree_planting_solution :
  ∃ (tp : TreePlanting), valid_tree_planting tp ∧ tp.students = 45 ∧ tp.typeA ≥ 80 :=
sorry

end NUMINAMATH_CALUDE_tree_planting_solution_l3893_389303


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_k_nonzero_l3893_389393

/-- Given a quadratic equation kx^2 - 2x + 1/2 = 0, if it has two distinct real roots, then k ≠ 0 -/
theorem quadratic_distinct_roots_k_nonzero (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 2*x + 1/2 = 0 ∧ k * y^2 - 2*y + 1/2 = 0) → k ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_k_nonzero_l3893_389393


namespace NUMINAMATH_CALUDE_total_rhino_weight_l3893_389301

/-- The weight of a white rhino in pounds -/
def white_rhino_weight : ℕ := 5100

/-- The weight of a black rhino in pounds -/
def black_rhino_weight : ℕ := 2000

/-- The number of white rhinos -/
def num_white_rhinos : ℕ := 7

/-- The number of black rhinos -/
def num_black_rhinos : ℕ := 8

/-- Theorem: The total weight of 7 white rhinos and 8 black rhinos is 51,700 pounds -/
theorem total_rhino_weight :
  num_white_rhinos * white_rhino_weight + num_black_rhinos * black_rhino_weight = 51700 := by
  sorry

end NUMINAMATH_CALUDE_total_rhino_weight_l3893_389301


namespace NUMINAMATH_CALUDE_xy_reciprocal_inequality_l3893_389358

theorem xy_reciprocal_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (1 + x) * (1 + y) = 2) : x * y + 1 / (x * y) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_xy_reciprocal_inequality_l3893_389358


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l3893_389340

/-- Given a scalene triangle with angles in the ratio 1:2:3 and the smallest angle being 30°,
    the largest angle is 90°. -/
theorem largest_angle_in_special_triangle :
  ∀ (a b c : ℝ), 
    0 < a ∧ 0 < b ∧ 0 < c →  -- angles are positive
    a < b ∧ b < c →  -- scalene triangle condition
    a + b + c = 180 →  -- sum of angles in a triangle
    b = 2*a ∧ c = 3*a →  -- ratio of angles is 1:2:3
    a = 30 →  -- smallest angle is 30°
    c = 90 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l3893_389340


namespace NUMINAMATH_CALUDE_movie_only_attendance_l3893_389361

/-- Represents the number of students attending different activities --/
structure ActivityAttendance where
  total : ℕ
  picnic : ℕ
  games : ℕ
  movie_and_picnic : ℕ
  movie_and_games : ℕ
  picnic_and_games : ℕ
  all_activities : ℕ

/-- The given conditions for the problem --/
def given_conditions : ActivityAttendance :=
  { total := 31
  , picnic := 20
  , games := 5
  , movie_and_picnic := 4
  , movie_and_games := 2
  , picnic_and_games := 0
  , all_activities := 2
  }

/-- Theorem stating that the number of students meeting for the movie only is 12 --/
theorem movie_only_attendance (conditions : ActivityAttendance) : 
  conditions.total - (conditions.picnic + conditions.games - conditions.movie_and_picnic - conditions.movie_and_games - conditions.picnic_and_games + conditions.all_activities) = 12 :=
by sorry

end NUMINAMATH_CALUDE_movie_only_attendance_l3893_389361


namespace NUMINAMATH_CALUDE_efficiency_increase_sakshi_to_tanya_l3893_389336

/-- The percentage increase in efficiency between two work rates -/
def efficiency_increase (rate1 rate2 : ℚ) : ℚ :=
  (rate2 - rate1) / rate1 * 100

/-- Sakshi's work rate in parts per day -/
def sakshi_rate : ℚ := 1 / 25

/-- Tanya's work rate in parts per day -/
def tanya_rate : ℚ := 1 / 20

theorem efficiency_increase_sakshi_to_tanya :
  efficiency_increase sakshi_rate tanya_rate = 25 := by
  sorry

end NUMINAMATH_CALUDE_efficiency_increase_sakshi_to_tanya_l3893_389336


namespace NUMINAMATH_CALUDE_max_a_squared_b_l3893_389353

theorem max_a_squared_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a * (a + b) = 27) :
  a^2 * b ≤ 54 := by
sorry

end NUMINAMATH_CALUDE_max_a_squared_b_l3893_389353


namespace NUMINAMATH_CALUDE_prism_volume_l3893_389308

/-- The volume of a right rectangular prism with face areas 10, 15, and 18 square inches is 30√3 cubic inches. -/
theorem prism_volume (l w h : ℝ) (h1 : l * w = 10) (h2 : w * h = 15) (h3 : l * h = 18) :
  l * w * h = 30 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l3893_389308


namespace NUMINAMATH_CALUDE_test_question_percentage_l3893_389315

theorem test_question_percentage (second_correct : ℝ) (neither_correct : ℝ) (both_correct : ℝ)
  (h1 : second_correct = 0.55)
  (h2 : neither_correct = 0.20)
  (h3 : both_correct = 0.50) :
  ∃ first_correct : ℝ,
    first_correct = 0.75 ∧
    first_correct + second_correct - both_correct + neither_correct = 1 :=
by sorry

end NUMINAMATH_CALUDE_test_question_percentage_l3893_389315


namespace NUMINAMATH_CALUDE_line_point_k_value_l3893_389387

/-- Given a line containing points (0, 10), (5, k), and (25, 0), prove that k = 8 -/
theorem line_point_k_value (k : ℝ) : 
  (∃ (m b : ℝ), m * 0 + b = 10 ∧ m * 5 + b = k ∧ m * 25 + b = 0) → k = 8 := by
  sorry

end NUMINAMATH_CALUDE_line_point_k_value_l3893_389387


namespace NUMINAMATH_CALUDE_solution_and_minimum_value_l3893_389380

-- Define the solution set of |2x-3| < x
def solution_set : Set ℝ := {x | 1 < x ∧ x < 3}

-- Define m and n based on the quadratic equation x^2 - mx + n = 0 with roots 1 and 3
def m : ℝ := 4
def n : ℝ := 3

-- Define the constraint for a, b, c
def abc_constraint (a b c : ℝ) : Prop := 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 ∧ 0 < c ∧ c < 1 ∧ a * b + b * c + a * c = 1

theorem solution_and_minimum_value :
  (m - n = 1) ∧
  (∀ a b c : ℝ, abc_constraint a b c → a + b + c ≥ Real.sqrt 3) ∧
  (∃ a b c : ℝ, abc_constraint a b c ∧ a + b + c = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_solution_and_minimum_value_l3893_389380


namespace NUMINAMATH_CALUDE_smallest_dual_palindrome_is_17_l3893_389309

/-- Checks if a number is a palindrome in the given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop :=
  let digits := Nat.digits base n
  digits = digits.reverse

/-- The smallest positive integer greater than 10 that is a palindrome in both base 2 and base 4 -/
def smallestDualPalindrome : ℕ := 17

theorem smallest_dual_palindrome_is_17 :
  (smallestDualPalindrome > 10) ∧
  (isPalindrome smallestDualPalindrome 2) ∧
  (isPalindrome smallestDualPalindrome 4) ∧
  (∀ n : ℕ, n > 10 ∧ n < smallestDualPalindrome →
    ¬(isPalindrome n 2 ∧ isPalindrome n 4)) :=
by sorry

#eval smallestDualPalindrome

end NUMINAMATH_CALUDE_smallest_dual_palindrome_is_17_l3893_389309


namespace NUMINAMATH_CALUDE_cube_difference_simplification_l3893_389367

theorem cube_difference_simplification (a b : ℝ) (ha_pos : a > 0) (hb_neg : b < 0)
  (ha_sq : a^2 = 9/25) (hb_sq : b^2 = (3 + Real.sqrt 2)^2 / 14) :
  (a - b)^3 = 88 * Real.sqrt 2 / 12750 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_simplification_l3893_389367


namespace NUMINAMATH_CALUDE_complex_number_i_properties_l3893_389326

/-- Given a complex number i such that i^2 = -1, prove the properties of i raised to different powers -/
theorem complex_number_i_properties (i : ℂ) (n : ℕ) (h : i^2 = -1) :
  i^(4*n + 1) = i ∧ i^(4*n + 2) = -1 ∧ i^(4*n + 3) = -i := by
  sorry

end NUMINAMATH_CALUDE_complex_number_i_properties_l3893_389326


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l3893_389329

theorem units_digit_of_expression : ∃ n : ℕ, (9 * 19 * 1989 - 9^4) % 10 = 8 ∧ n * 10 + 8 = 9 * 19 * 1989 - 9^4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l3893_389329


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3893_389390

-- Define the sets M and N
def M : Set ℕ := {0, 1, 3}
def N : Set ℕ := {0, 1, 7}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3893_389390


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l3893_389384

theorem cube_root_equation_solution :
  ∃ y : ℝ, y = 1/32 ∧ (5 - 1/y)^(1/3 : ℝ) = -3 :=
sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l3893_389384


namespace NUMINAMATH_CALUDE_tan_squared_to_sin_squared_l3893_389307

noncomputable def f (x : ℝ) : ℝ :=
  1 / ((x / (x - 1)))

theorem tan_squared_to_sin_squared (t : ℝ) (h1 : 0 ≤ t) (h2 : t ≤ π/2) :
  f (Real.tan t ^ 2) = Real.sin t ^ 2 :=
by
  sorry

end NUMINAMATH_CALUDE_tan_squared_to_sin_squared_l3893_389307


namespace NUMINAMATH_CALUDE_power_multiplication_division_equality_l3893_389304

theorem power_multiplication_division_equality : (12 : ℕ)^1 * 6^4 / 432 = 36 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_division_equality_l3893_389304


namespace NUMINAMATH_CALUDE_area_of_triangle_PF1F2_l3893_389337

-- Define the ellipse C1
def C1 (x y : ℝ) : Prop := x^2/6 + y^2/2 = 1

-- Define the hyperbola C2
def C2 (x y : ℝ) : Prop := x^2/3 - y^2 = 1

-- Define the foci F1 and F2
def F1 : ℝ × ℝ := (-2, 0)
def F2 : ℝ × ℝ := (2, 0)

-- Define a point P that satisfies both C1 and C2
def P : ℝ × ℝ := sorry

-- Assume P is on both C1 and C2
axiom P_on_C1 : C1 P.1 P.2
axiom P_on_C2 : C2 P.1 P.2

-- Define the distance function
def dist (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the area of a triangle given three points
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_of_triangle_PF1F2 : 
  triangle_area P F1 F2 = Real.sqrt 2 := sorry

end NUMINAMATH_CALUDE_area_of_triangle_PF1F2_l3893_389337


namespace NUMINAMATH_CALUDE_problem_statement_l3893_389312

/-- Given a function f(x) = ax^5 + bx^3 + cx - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem problem_statement (a b c : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = a * x^5 + b * x^3 + c * x - 8)
    (h2 : f (-2) = 10) : 
  f 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3893_389312


namespace NUMINAMATH_CALUDE_sin_cos_identity_l3893_389313

theorem sin_cos_identity (x : ℝ) (h : Real.sin (x + π/3) = 1/3) :
  Real.sin (5*π/3 - x) - Real.cos (2*x - π/3) = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l3893_389313


namespace NUMINAMATH_CALUDE_total_chocolates_in_month_l3893_389319

/-- Represents the number of chocolates Kantana buys for herself each Saturday -/
def self_chocolates_per_saturday : ℕ := 2

/-- Represents the number of chocolates Kantana buys for her sister each Saturday -/
def sister_chocolates_per_saturday : ℕ := 1

/-- Represents the number of Saturdays in a month -/
def saturdays_in_month : ℕ := 4

/-- Represents the number of chocolates Kantana bought for her friend Charlie -/
def charlie_chocolates : ℕ := 10

/-- Theorem stating the total number of chocolates Kantana bought in a month -/
theorem total_chocolates_in_month : 
  self_chocolates_per_saturday * saturdays_in_month + 
  sister_chocolates_per_saturday * saturdays_in_month + 
  charlie_chocolates = 22 := by
  sorry

end NUMINAMATH_CALUDE_total_chocolates_in_month_l3893_389319


namespace NUMINAMATH_CALUDE_number_manipulation_l3893_389392

theorem number_manipulation (x : ℝ) : (x - 5) / 7 = 7 → (x - 14) / 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_manipulation_l3893_389392


namespace NUMINAMATH_CALUDE_pens_left_after_giving_away_l3893_389377

/-- Given that a student's parents bought her 56 pens and she gave 22 pens to her friends,
    prove that the number of pens left for her to use is 34. -/
theorem pens_left_after_giving_away (total_pens : ℕ) (pens_given_away : ℕ) :
  total_pens = 56 → pens_given_away = 22 → total_pens - pens_given_away = 34 := by
  sorry

end NUMINAMATH_CALUDE_pens_left_after_giving_away_l3893_389377


namespace NUMINAMATH_CALUDE_goose_egg_hatch_fraction_l3893_389348

theorem goose_egg_hatch_fraction (total_eggs : ℕ) (survived_year : ℕ) 
  (h1 : total_eggs = 550)
  (h2 : survived_year = 110)
  (h3 : ∀ x : ℚ, x * total_eggs * (3/4 : ℚ) * (2/5 : ℚ) = survived_year → x = 2/3) :
  ∃ x : ℚ, x * total_eggs = (total_eggs : ℚ) * (2/3 : ℚ) := by sorry

end NUMINAMATH_CALUDE_goose_egg_hatch_fraction_l3893_389348


namespace NUMINAMATH_CALUDE_brick_weighs_32_kg_l3893_389350

-- Define the weight of one brick
def brick_weight : ℝ := sorry

-- Define the weight of one statue
def statue_weight : ℝ := sorry

-- Theorem stating the weight of one brick is 32 kg
theorem brick_weighs_32_kg : brick_weight = 32 :=
  by
  -- Condition 1: 5 bricks weigh the same as 4 statues
  have h1 : 5 * brick_weight = 4 * statue_weight := sorry
  -- Condition 2: 2 statues weigh 80 kg
  have h2 : 2 * statue_weight = 80 := sorry
  sorry -- Proof goes here


end NUMINAMATH_CALUDE_brick_weighs_32_kg_l3893_389350


namespace NUMINAMATH_CALUDE_existence_of_plane_only_properties_l3893_389371

-- Define abstract types for plane and solid geometry
def PlaneGeometry : Type := Unit
def SolidGeometry : Type := Unit

-- Define a property as a function that takes a geometry and returns a proposition
def GeometricProperty : Type := (PlaneGeometry ⊕ SolidGeometry) → Prop

-- Define a function to check if a property holds in plane geometry
def holdsInPlaneGeometry (prop : GeometricProperty) : Prop :=
  prop (Sum.inl ())

-- Define a function to check if a property holds in solid geometry
def holdsInSolidGeometry (prop : GeometricProperty) : Prop :=
  prop (Sum.inr ())

-- State the theorem
theorem existence_of_plane_only_properties :
  ∃ (prop : GeometricProperty),
    holdsInPlaneGeometry prop ∧ ¬holdsInSolidGeometry prop := by
  sorry

-- Examples of properties (these are just placeholders and not actual proofs)
def perpendicularLinesParallel : GeometricProperty := fun _ => True
def uniquePerpendicularLine : GeometricProperty := fun _ => True
def equalSidedQuadrilateralIsRhombus : GeometricProperty := fun _ => True

end NUMINAMATH_CALUDE_existence_of_plane_only_properties_l3893_389371


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l3893_389362

theorem smallest_integer_with_remainders : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 2 = 1 ∧ 
  n % 3 = 2 ∧ 
  n % 4 = 3 ∧ 
  n % 10 = 9 ∧ 
  ∀ m : ℕ, m > 0 ∧ m % 2 = 1 ∧ m % 3 = 2 ∧ m % 4 = 3 ∧ m % 10 = 9 → n ≤ m :=
by
  sorry

#eval 59 % 2  -- Expected output: 1
#eval 59 % 3  -- Expected output: 2
#eval 59 % 4  -- Expected output: 3
#eval 59 % 10 -- Expected output: 9

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l3893_389362


namespace NUMINAMATH_CALUDE_tan_monotonic_interval_l3893_389333

/-- The monotonic increasing interval of tan(x + π/4) -/
theorem tan_monotonic_interval (k : ℤ) :
  ∀ x : ℝ, (k * π - 3 * π / 4 < x ∧ x < k * π + π / 4) →
    Monotone (fun x => Real.tan (x + π / 4)) := by
  sorry

end NUMINAMATH_CALUDE_tan_monotonic_interval_l3893_389333


namespace NUMINAMATH_CALUDE_odometer_puzzle_l3893_389386

/-- Represents the odometer reading as a triple of digits -/
structure OdometerReading where
  hundreds : Nat
  tens : Nat
  ones : Nat

/-- Represents the trip details -/
structure TripDetails where
  initial : OdometerReading
  final : OdometerReading
  duration : Nat  -- in hours
  avgSpeed : Nat  -- in miles per hour

theorem odometer_puzzle (trip : TripDetails) :
  trip.initial.hundreds ≥ 2 ∧
  trip.initial.hundreds + trip.initial.tens + trip.initial.ones = 9 ∧
  trip.avgSpeed = 60 ∧
  trip.initial.hundreds = trip.final.ones ∧
  trip.initial.tens = trip.final.tens ∧
  trip.initial.ones = trip.final.hundreds →
  trip.initial.hundreds^2 + trip.initial.tens^2 + trip.initial.ones^2 = 33 := by
  sorry

end NUMINAMATH_CALUDE_odometer_puzzle_l3893_389386


namespace NUMINAMATH_CALUDE_die_throws_probability_l3893_389395

/-- The probability of rolling a number greater than 4 on a single die throw -/
def prob_high : ℚ := 1/3

/-- The probability of rolling a number less than or equal to 4 on a single die throw -/
def prob_low : ℚ := 2/3

/-- The probability of getting at least two numbers greater than 4 in two die throws -/
def prob_at_least_two_high : ℚ := prob_high * prob_high + 2 * prob_high * prob_low

theorem die_throws_probability :
  prob_at_least_two_high = 5/9 := by sorry

end NUMINAMATH_CALUDE_die_throws_probability_l3893_389395


namespace NUMINAMATH_CALUDE_intersection_with_complement_of_reals_l3893_389391

open Set

theorem intersection_with_complement_of_reals (A B : Set ℝ) 
  (hA : A = {x : ℝ | x > 0}) 
  (hB : B = {x : ℝ | x > 1}) : 
  A ∩ (Set.univ \ B) = {x : ℝ | 0 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_of_reals_l3893_389391


namespace NUMINAMATH_CALUDE_quadratic_function_value_l3893_389398

/-- A quadratic function with specified properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_function_value (a b c : ℝ) :
  (∀ x, QuadraticFunction a b c x ≤ 75) ∧ 
  (QuadraticFunction a b c (-3) = 0) ∧ 
  (QuadraticFunction a b c 3 = 0) →
  QuadraticFunction a b c 2 = 125/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l3893_389398


namespace NUMINAMATH_CALUDE_special_functions_properties_l3893_389346

/-- Two functions satisfying a specific functional equation -/
class SpecialFunctions (f g : ℝ → ℝ) : Prop where
  eq : ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * g y
  f_zero : f 0 = 0
  f_nonzero : ∃ x : ℝ, f x ≠ 0

/-- f is an odd function -/
def IsOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

/-- g is an even function -/
def IsEvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = g x

/-- Main theorem: f is odd and g is even -/
theorem special_functions_properties {f g : ℝ → ℝ} [SpecialFunctions f g] :
    IsOddFunction f ∧ IsEvenFunction g := by
  sorry

end NUMINAMATH_CALUDE_special_functions_properties_l3893_389346


namespace NUMINAMATH_CALUDE_union_M_N_equals_M_l3893_389328

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 = 0}
def N : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 0}

-- State the theorem
theorem union_M_N_equals_M : M ∪ N = M := by sorry

end NUMINAMATH_CALUDE_union_M_N_equals_M_l3893_389328


namespace NUMINAMATH_CALUDE_initial_marbles_count_l3893_389311

/-- The number of marbles Carla bought -/
def marbles_bought : ℕ := 134

/-- The total number of marbles Carla has now -/
def total_marbles_now : ℕ := 187

/-- The initial number of marbles Carla had -/
def initial_marbles : ℕ := total_marbles_now - marbles_bought

theorem initial_marbles_count : initial_marbles = 53 := by
  sorry

end NUMINAMATH_CALUDE_initial_marbles_count_l3893_389311


namespace NUMINAMATH_CALUDE_inverse_difference_l3893_389356

theorem inverse_difference (a : ℝ) (h : a + a⁻¹ = 6) : a - a⁻¹ = 4 * Real.sqrt 2 ∨ a - a⁻¹ = -4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_difference_l3893_389356


namespace NUMINAMATH_CALUDE_unique_root_formula_l3893_389381

/-- A quadratic polynomial with exactly one root -/
class UniqueRootQuadratic (g : ℝ → ℝ) : Prop where
  is_quadratic : ∃ a b c : ℝ, ∀ x, g x = a * x^2 + b * x + c
  unique_root : ∃! x : ℝ, g x = 0

/-- The property that g(ax + b) + g(cx + d) has exactly one root -/
def has_unique_combined_root (g : ℝ → ℝ) (a b c d : ℝ) : Prop :=
  ∃! x : ℝ, g (a * x + b) + g (c * x + d) = 0

theorem unique_root_formula 
  (g : ℝ → ℝ) (a b c d : ℝ) 
  [UniqueRootQuadratic g] 
  (h₁ : has_unique_combined_root g a b c d) 
  (h₂ : a ≠ c) : 
  ∃ x₀ : ℝ, (∀ x, g x = 0 ↔ x = x₀) ∧ x₀ = (a * d - b * c) / (a - c) := by
  sorry

end NUMINAMATH_CALUDE_unique_root_formula_l3893_389381


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3893_389306

/-- Given a geometric sequence {a_n} with sum of first n terms S_n, prove S_5/a_5 = 31 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n, a (n + 1) = q * a n) →  -- a_n is a geometric sequence
  (a 1 + a 3 = 5/2) →                    -- First condition
  (a 2 + a 4 = 5/4) →                    -- Second condition
  (∀ n, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1))) →  -- Definition of S_n
  (S 5 / a 5 = 31) :=                    -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3893_389306


namespace NUMINAMATH_CALUDE_rabbit_population_l3893_389360

theorem rabbit_population (breeding_rabbits : ℕ) (first_spring_ratio : ℕ) 
  (second_spring_kittens : ℕ) (second_spring_adopted : ℕ) (total_rabbits : ℕ) :
  breeding_rabbits = 10 →
  second_spring_kittens = 60 →
  second_spring_adopted = 4 →
  total_rabbits = 121 →
  breeding_rabbits + (first_spring_ratio * breeding_rabbits / 2 + 5) + 
    (second_spring_kittens - second_spring_adopted) = total_rabbits →
  first_spring_ratio = 10 := by
sorry

end NUMINAMATH_CALUDE_rabbit_population_l3893_389360


namespace NUMINAMATH_CALUDE_no_solution_for_system_l3893_389394

theorem no_solution_for_system :
  ¬∃ (x y z : ℝ), 
    (Real.sqrt (2 * x^2 + 2) = y - 1) ∧
    (Real.sqrt (2 * y^2 + 2) = z - 1) ∧
    (Real.sqrt (2 * z^2 + 2) = x - 1) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_system_l3893_389394


namespace NUMINAMATH_CALUDE_equal_fish_time_l3893_389379

def brent_fish (n : ℕ) : ℕ := 9 * 4^n

def gretel_fish (n : ℕ) : ℕ := 243 * 3^n

theorem equal_fish_time : ∃ (n : ℕ), n > 0 ∧ brent_fish n = gretel_fish n ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_equal_fish_time_l3893_389379


namespace NUMINAMATH_CALUDE_simplify_expression_l3893_389355

theorem simplify_expression (x : ℝ) : (5 - 4*x) - (7 + 5*x - x^2) = x^2 - 9*x - 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3893_389355


namespace NUMINAMATH_CALUDE_dogwood_tree_count_l3893_389310

/-- The number of dogwood trees in the park after planting and removal operations --/
def final_tree_count (initial_trees : ℕ) (planted_today : ℕ) (planted_tomorrow : ℕ) 
                     (removed_today : ℕ) (workers : ℕ) : ℕ :=
  initial_trees + planted_today + planted_tomorrow - removed_today

theorem dogwood_tree_count : 
  let initial_trees := 7
  let planted_today := 5
  let planted_tomorrow := 4
  let removed_today := 3
  let workers := 8
  final_tree_count initial_trees planted_today planted_tomorrow removed_today workers = 13 := by
  sorry

end NUMINAMATH_CALUDE_dogwood_tree_count_l3893_389310


namespace NUMINAMATH_CALUDE_nails_in_toolshed_l3893_389364

theorem nails_in_toolshed (initial_nails : ℕ) (nails_to_buy : ℕ) (total_nails : ℕ) :
  initial_nails = 247 →
  nails_to_buy = 109 →
  total_nails = 500 →
  total_nails = initial_nails + nails_to_buy + (total_nails - initial_nails - nails_to_buy) →
  total_nails - initial_nails - nails_to_buy = 144 :=
by sorry

end NUMINAMATH_CALUDE_nails_in_toolshed_l3893_389364


namespace NUMINAMATH_CALUDE_investment_average_rate_l3893_389341

/-- Proves that given a total investment split between two schemes with different rates,
    if the annual returns from both parts are equal, then the average rate of interest
    on the total investment is as calculated. -/
theorem investment_average_rate
  (total_investment : ℝ)
  (rate1 rate2 : ℝ)
  (h_total : total_investment = 5000)
  (h_rates : rate1 = 0.03 ∧ rate2 = 0.05)
  (h_equal_returns : ∃ (x : ℝ), x ≥ 0 ∧ x ≤ total_investment ∧
    rate1 * (total_investment - x) = rate2 * x) :
  (rate1 * (total_investment - x) + rate2 * x) / total_investment = 0.0375 :=
sorry

end NUMINAMATH_CALUDE_investment_average_rate_l3893_389341


namespace NUMINAMATH_CALUDE_class_average_l3893_389345

theorem class_average (total_students : ℕ) 
                      (top_scorers : ℕ) 
                      (zero_scorers : ℕ) 
                      (top_score : ℝ) 
                      (rest_average : ℝ) :
  total_students = 25 →
  top_scorers = 5 →
  zero_scorers = 3 →
  top_score = 95 →
  rest_average = 45 →
  let rest_students := total_students - top_scorers - zero_scorers
  let total_score := top_scorers * top_score + zero_scorers * 0 + rest_students * rest_average
  total_score / total_students = 49.6 := by
  sorry

end NUMINAMATH_CALUDE_class_average_l3893_389345
