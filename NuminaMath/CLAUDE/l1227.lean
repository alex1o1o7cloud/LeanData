import Mathlib

namespace NUMINAMATH_CALUDE_chocolate_bars_to_sell_l1227_122772

theorem chocolate_bars_to_sell (initial : ℕ) (sold_week1 : ℕ) (sold_week2 : ℕ) 
  (h1 : initial = 18)
  (h2 : sold_week1 = 5)
  (h3 : sold_week2 = 7) :
  initial - (sold_week1 + sold_week2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_to_sell_l1227_122772


namespace NUMINAMATH_CALUDE_multiple_birth_statistics_l1227_122798

theorem multiple_birth_statistics (total_babies : ℕ) 
  (twins triplets quadruplets quintuplets : ℕ) : 
  total_babies = 1250 →
  quadruplets = 3 * quintuplets →
  triplets = 2 * quadruplets →
  twins = 2 * triplets →
  2 * twins + 3 * triplets + 4 * quadruplets + 5 * quintuplets = total_babies →
  5 * quintuplets = 6250 / 59 := by
  sorry

end NUMINAMATH_CALUDE_multiple_birth_statistics_l1227_122798


namespace NUMINAMATH_CALUDE_no_natural_pair_satisfies_condition_l1227_122712

theorem no_natural_pair_satisfies_condition : 
  ¬∃ (a b : ℕ), a ≠ 0 ∧ b ≠ 0 ∧ (b^a ∣ a^b - 1) :=
by sorry

end NUMINAMATH_CALUDE_no_natural_pair_satisfies_condition_l1227_122712


namespace NUMINAMATH_CALUDE_marcy_cat_time_l1227_122768

def total_time (petting combing brushing playing feeding cleaning : ℚ) : ℚ :=
  petting + combing + brushing + playing + feeding + cleaning

theorem marcy_cat_time : 
  let petting : ℚ := 12
  let combing : ℚ := (1/3) * petting
  let brushing : ℚ := (1/4) * combing
  let playing : ℚ := (1/2) * petting
  let feeding : ℚ := 5
  let cleaning : ℚ := (2/5) * feeding
  total_time petting combing brushing playing feeding cleaning = 30 := by
sorry

end NUMINAMATH_CALUDE_marcy_cat_time_l1227_122768


namespace NUMINAMATH_CALUDE_otimes_three_four_l1227_122795

-- Define the ⊗ operation
def otimes (m : ℤ) (a b : ℕ) : ℚ :=
  (m * a + b) / (2 * a * b)

-- Theorem statement
theorem otimes_three_four (m : ℤ) :
  (∀ (a b : ℕ), a ≠ 0 → b ≠ 0 → otimes m a b = otimes m b a) →
  otimes m 1 4 = otimes m 2 3 →
  otimes m 3 4 = 11 / 12 := by
  sorry


end NUMINAMATH_CALUDE_otimes_three_four_l1227_122795


namespace NUMINAMATH_CALUDE_al_sandwich_count_l1227_122736

/-- The number of different types of bread available. -/
def num_bread : ℕ := 5

/-- The number of different types of meat available. -/
def num_meat : ℕ := 6

/-- The number of different types of cheese available. -/
def num_cheese : ℕ := 5

/-- Represents whether French bread is available. -/
def french_bread_available : Prop := True

/-- Represents whether turkey is available. -/
def turkey_available : Prop := True

/-- Represents whether Swiss cheese is available. -/
def swiss_cheese_available : Prop := True

/-- Represents whether white bread is available. -/
def white_bread_available : Prop := True

/-- Represents whether rye bread is available. -/
def rye_bread_available : Prop := True

/-- Represents whether chicken is available. -/
def chicken_available : Prop := True

/-- The number of sandwich combinations with turkey and Swiss cheese. -/
def turkey_swiss_combos : ℕ := num_bread

/-- The number of sandwich combinations with white bread and chicken. -/
def white_chicken_combos : ℕ := num_cheese

/-- The number of sandwich combinations with rye bread and turkey. -/
def rye_turkey_combos : ℕ := num_cheese

/-- The total number of sandwich combinations Al can order. -/
def al_sandwich_options : ℕ := num_bread * num_meat * num_cheese - turkey_swiss_combos - white_chicken_combos - rye_turkey_combos

theorem al_sandwich_count :
  french_bread_available ∧ 
  turkey_available ∧ 
  swiss_cheese_available ∧ 
  white_bread_available ∧
  rye_bread_available ∧
  chicken_available →
  al_sandwich_options = 135 := by
  sorry

end NUMINAMATH_CALUDE_al_sandwich_count_l1227_122736


namespace NUMINAMATH_CALUDE_hamburger_price_correct_l1227_122781

/-- The price of a hamburger that satisfies the given conditions -/
def hamburger_price : ℝ := 3.125

/-- The number of hamburgers already sold -/
def hamburgers_sold : ℕ := 12

/-- The number of additional hamburgers needed to be sold -/
def additional_hamburgers : ℕ := 4

/-- The total revenue target -/
def total_revenue : ℝ := 50

/-- Theorem stating that the hamburger price satisfies the given conditions -/
theorem hamburger_price_correct : 
  hamburger_price * (hamburgers_sold + additional_hamburgers) = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_hamburger_price_correct_l1227_122781


namespace NUMINAMATH_CALUDE_milk_delivery_calculation_l1227_122724

/-- Given a total amount of milk and a difference between two people's deliveries,
    calculate the amount delivered by the person delivering more milk. -/
theorem milk_delivery_calculation (total : ℕ) (difference : ℕ) (h1 : total = 2100) (h2 : difference = 200) :
  (total + difference) / 2 = 1150 := by
  sorry

end NUMINAMATH_CALUDE_milk_delivery_calculation_l1227_122724


namespace NUMINAMATH_CALUDE_sneakers_cost_l1227_122727

/-- The cost of Leonard's wallet -/
def wallet_cost : ℕ := 50

/-- The cost of Michael's backpack -/
def backpack_cost : ℕ := 100

/-- The cost of each pair of jeans -/
def jeans_cost : ℕ := 50

/-- The total amount spent by Leonard and Michael -/
def total_spent : ℕ := 450

/-- The number of pairs of sneakers bought -/
def num_sneakers : ℕ := 2

/-- The number of pairs of jeans bought -/
def num_jeans : ℕ := 2

theorem sneakers_cost (sneakers_price : ℕ) : 
  wallet_cost + num_sneakers * sneakers_price + backpack_cost + num_jeans * jeans_cost = total_spent →
  sneakers_price = 100 := by
sorry

end NUMINAMATH_CALUDE_sneakers_cost_l1227_122727


namespace NUMINAMATH_CALUDE_average_of_shifted_data_l1227_122749

/-- Given four positive real numbers with a specific variance, prove that the average of these numbers plus 3 is 5 -/
theorem average_of_shifted_data (x₁ x₂ x₃ x₄ : ℝ) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0)
  (h_var : (x₁^2 + x₂^2 + x₃^2 + x₄^2 - 16) / 4 = (x₁^2 + x₂^2 + x₃^2 + x₄^2) / 4 - ((x₁ + x₂ + x₃ + x₄) / 4)^2) :
  ((x₁ + 3) + (x₂ + 3) + (x₃ + 3) + (x₄ + 3)) / 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_average_of_shifted_data_l1227_122749


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_10_70_1_7th_l1227_122769

def arithmeticSeriesSum (a₁ aₙ d : ℚ) : ℚ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_series_sum_10_70_1_7th : 
  arithmeticSeriesSum 10 70 (1/7) = 16840 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_10_70_1_7th_l1227_122769


namespace NUMINAMATH_CALUDE_mary_fruit_cost_l1227_122731

/-- The total cost of fruits Mary bought -/
def total_cost (berries apples peaches grapes bananas pineapples : ℚ) : ℚ :=
  berries + apples + peaches + grapes + bananas + pineapples

/-- Theorem stating that the total cost of fruits Mary bought is $52.09 -/
theorem mary_fruit_cost :
  total_cost 11.08 14.33 9.31 7.50 5.25 4.62 = 52.09 := by
  sorry

end NUMINAMATH_CALUDE_mary_fruit_cost_l1227_122731


namespace NUMINAMATH_CALUDE_stratified_sampling_proportion_l1227_122773

/-- Represents the number of athletes selected in a stratified sampling -/
structure StratifiedSample where
  male : ℕ
  female : ℕ

/-- Represents the composition of the track and field team -/
def team : StratifiedSample :=
  { male := 56, female := 42 }

/-- Calculates the ratio of male to female athletes -/
def ratio (s : StratifiedSample) : ℚ :=
  s.male / s.female

/-- Theorem: In a stratified sampling, if 8 male athletes are selected,
    then 6 female athletes should be selected to maintain the same proportion -/
theorem stratified_sampling_proportion :
  ∀ (sample : StratifiedSample),
    sample.male = 8 →
    ratio sample = ratio team →
    sample.female = 6 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_proportion_l1227_122773


namespace NUMINAMATH_CALUDE_square_ad_perimeter_l1227_122725

theorem square_ad_perimeter (side_length : ℝ) (h : side_length = 8) : 
  4 * side_length = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_ad_perimeter_l1227_122725


namespace NUMINAMATH_CALUDE_digit_150_is_7_l1227_122797

/-- The decimal representation of 17/70 -/
def decimal_rep : ℚ := 17 / 70

/-- The length of the repeating sequence in the decimal representation of 17/70 -/
def repeat_length : ℕ := 6

/-- The nth digit after the decimal point in the decimal representation of 17/70 -/
def nth_digit (n : ℕ) : ℕ := sorry

/-- Theorem: The 150th digit after the decimal point in the decimal representation of 17/70 is 7 -/
theorem digit_150_is_7 : nth_digit 150 = 7 := by sorry

end NUMINAMATH_CALUDE_digit_150_is_7_l1227_122797


namespace NUMINAMATH_CALUDE_certain_number_addition_l1227_122745

theorem certain_number_addition (x : ℤ) : x + 36 = 71 → x + 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_addition_l1227_122745


namespace NUMINAMATH_CALUDE_m_range_l1227_122756

-- Define the propositions
def p (m : ℝ) : Prop := ∀ x, |x| + |x - 1| > m
def q (m : ℝ) : Prop := ∀ x y, x < y → (-(5 - 2*m)^x) > (-(5 - 2*m)^y)

-- Define the theorem
theorem m_range :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ m ∈ Set.Icc 1 2 ∧ m ≠ 2 :=
sorry

end NUMINAMATH_CALUDE_m_range_l1227_122756


namespace NUMINAMATH_CALUDE_train_length_calculation_l1227_122792

/-- Represents the properties of a train and its movement --/
structure Train where
  length : ℝ
  speed : ℝ
  platform_crossing_time : ℝ
  pole_crossing_time : ℝ
  platform_length : ℝ

/-- Theorem stating the length of the train given specific conditions --/
theorem train_length_calculation (t : Train)
  (h1 : t.platform_crossing_time = 39)
  (h2 : t.pole_crossing_time = 16)
  (h3 : t.platform_length = 431.25)
  (h4 : t.length = t.speed * t.pole_crossing_time)
  (h5 : t.length + t.platform_length = t.speed * t.platform_crossing_time) :
  t.length = 6890 / 23 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l1227_122792


namespace NUMINAMATH_CALUDE_expression_value_l1227_122782

theorem expression_value : 
  let x : ℤ := 2
  let y : ℤ := -3
  let z : ℤ := 7
  x^2 + y^2 + z^2 - 2*x*y = 74 := by sorry

end NUMINAMATH_CALUDE_expression_value_l1227_122782


namespace NUMINAMATH_CALUDE_cube_root_sum_reciprocal_cube_l1227_122714

theorem cube_root_sum_reciprocal_cube (x : ℝ) : 
  x = Real.rpow 4 (1/3) + Real.rpow 2 (1/3) + 1 → (1 + 1/x)^3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_sum_reciprocal_cube_l1227_122714


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l1227_122709

/-- Simple interest calculation -/
theorem simple_interest_calculation
  (principal : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : principal = 400)
  (h2 : rate = 17.5)
  (h3 : time = 2) :
  (principal * rate * time) / 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l1227_122709


namespace NUMINAMATH_CALUDE_simplify_expression_l1227_122790

theorem simplify_expression (b : ℝ) : 3*b*(3*b^2 + 2*b) - 4*b^2 = 9*b^3 + 2*b^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1227_122790


namespace NUMINAMATH_CALUDE_mans_walking_rate_l1227_122752

/-- The problem of finding a man's initial walking rate given certain conditions. -/
theorem mans_walking_rate (distance : ℝ) (early_speed : ℝ) (early_time : ℝ) (late_time : ℝ) :
  distance = 6.000000000000001 →
  early_speed = 6 →
  early_time = 5 / 60 →
  late_time = 7 / 60 →
  ∃ (initial_speed : ℝ),
    initial_speed = distance / (distance / early_speed + early_time + late_time) ∧
    initial_speed = 5 := by
  sorry

#eval 6.000000000000001 / (6.000000000000001 / 6 + 5 / 60 + 7 / 60)

end NUMINAMATH_CALUDE_mans_walking_rate_l1227_122752


namespace NUMINAMATH_CALUDE_solve_for_a_l1227_122766

theorem solve_for_a : ∃ a : ℝ, (1 : ℝ) - a * 2 = 3 ∧ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l1227_122766


namespace NUMINAMATH_CALUDE_leftover_space_is_one_l1227_122734

-- Define the wall length
def wall_length : ℝ := 15

-- Define the desk length
def desk_length : ℝ := 2

-- Define the bookcase length
def bookcase_length : ℝ := 1.5

-- Define the function to calculate the space left over
def space_left_over (n : ℕ) : ℝ :=
  wall_length - (n * desk_length + n * bookcase_length)

-- Theorem statement
theorem leftover_space_is_one :
  ∃ n : ℕ, n > 0 ∧ 
    space_left_over n = 1 ∧
    ∀ m : ℕ, m > n → space_left_over m < 1 :=
  sorry

end NUMINAMATH_CALUDE_leftover_space_is_one_l1227_122734


namespace NUMINAMATH_CALUDE_polynomial_product_l1227_122777

theorem polynomial_product (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (1 - x)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a + a₂ + a₄) * (a₁ + a₃ + a₅) = -256 := by
sorry

end NUMINAMATH_CALUDE_polynomial_product_l1227_122777


namespace NUMINAMATH_CALUDE_group_shopping_popularity_justified_l1227_122710

/-- Represents the practice of group shopping -/
structure GroupShopping where
  risks : ℕ  -- Number of risks associated with group shopping
  countries : ℕ  -- Number of countries where group shopping is practiced

/-- Factors contributing to group shopping popularity -/
structure PopularityFactors where
  cost_savings : ℝ  -- Percentage of cost savings
  quality_assessment : ℝ  -- Measure of quality assessment improvement
  trust_dynamics : ℝ  -- Measure of trust within the community

/-- Theorem stating that group shopping popularity is justified -/
theorem group_shopping_popularity_justified 
  (gs : GroupShopping) 
  (pf : PopularityFactors) : 
  gs.risks > 0 → 
  gs.countries > 10 → 
  pf.cost_savings > 20 → 
  pf.quality_assessment > 0.5 → 
  pf.trust_dynamics > 0.7 → 
  True := by
  sorry


end NUMINAMATH_CALUDE_group_shopping_popularity_justified_l1227_122710


namespace NUMINAMATH_CALUDE_intersection_and_union_for_negative_one_intersection_equals_B_iff_l1227_122711

def A : Set ℝ := {x | x ≤ -1 ∨ x ≥ 5}
def B (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x ≤ a+2}

theorem intersection_and_union_for_negative_one :
  (A ∩ B (-1) = {x | -2 ≤ x ∧ x ≤ -1}) ∧
  (A ∪ B (-1) = {x | x ≤ 1 ∨ x ≥ 5}) := by sorry

theorem intersection_equals_B_iff :
  ∀ a : ℝ, A ∩ B a = B a ↔ a ≤ -3 ∨ a > 2 := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_for_negative_one_intersection_equals_B_iff_l1227_122711


namespace NUMINAMATH_CALUDE_polynomial_irreducibility_l1227_122799

theorem polynomial_irreducibility (n : ℕ) (hn : n > 1) :
  Irreducible (Polynomial.X ^ n + 5 * Polynomial.X ^ (n - 1) + 3 : Polynomial ℤ) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_irreducibility_l1227_122799


namespace NUMINAMATH_CALUDE_mariel_dogs_count_l1227_122742

theorem mariel_dogs_count (total_legs : ℕ) (other_dogs : ℕ) (human_legs : ℕ) (dog_legs : ℕ) :
  total_legs = 36 →
  other_dogs = 3 →
  human_legs = 2 →
  dog_legs = 4 →
  ∃ (mariel_dogs : ℕ), mariel_dogs = 5 ∧
    total_legs = 2 * human_legs + other_dogs * dog_legs + mariel_dogs * dog_legs :=
by
  sorry

end NUMINAMATH_CALUDE_mariel_dogs_count_l1227_122742


namespace NUMINAMATH_CALUDE_equation_solution_l1227_122750

theorem equation_solution : ∃ c : ℝ, (c - 15) / 3 = (2 * c - 3) / 5 ∧ c = -66 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1227_122750


namespace NUMINAMATH_CALUDE_frog_jumps_even_leaves_l1227_122758

def frog_position (n : ℕ) (t : ℕ) : ℕ :=
  (t * (t + 1) / 2 - 1) % n

theorem frog_jumps_even_leaves (n : ℕ) (h1 : n > 1) :
  (∀ r, 0 ≤ r ∧ r < n → ∃ t, frog_position n t = r) →
  Even n :=
sorry

end NUMINAMATH_CALUDE_frog_jumps_even_leaves_l1227_122758


namespace NUMINAMATH_CALUDE_sum_of_square_areas_l1227_122765

theorem sum_of_square_areas (a b : ℕ) (ha : a = 8) (hb : b = 10) :
  a ^ 2 + b ^ 2 = 164 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_square_areas_l1227_122765


namespace NUMINAMATH_CALUDE_solve_equation_l1227_122788

theorem solve_equation : ∃ x : ℚ, 5 * (x - 9) = 6 * (3 - 3 * x) + 6 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1227_122788


namespace NUMINAMATH_CALUDE_min_packs_for_130_cans_l1227_122762

/-- Represents the number of cans in each pack type -/
inductive PackSize
  | small : PackSize
  | medium : PackSize
  | large : PackSize

/-- Returns the number of cans for a given pack size -/
def cansInPack (p : PackSize) : ℕ :=
  match p with
  | .small => 8
  | .medium => 15
  | .large => 30

/-- Represents a combination of packs -/
structure PackCombination where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Calculates the total number of cans in a pack combination -/
def totalCans (c : PackCombination) : ℕ :=
  c.small * cansInPack PackSize.small +
  c.medium * cansInPack PackSize.medium +
  c.large * cansInPack PackSize.large

/-- Calculates the total number of packs in a combination -/
def totalPacks (c : PackCombination) : ℕ :=
  c.small + c.medium + c.large

/-- Theorem: The minimum number of packs to buy 130 cans is 6 -/
theorem min_packs_for_130_cans :
  ∃ (c : PackCombination),
    totalCans c = 130 ∧
    totalPacks c = 6 ∧
    (∀ (d : PackCombination), totalCans d = 130 → totalPacks d ≥ 6) :=
by sorry

end NUMINAMATH_CALUDE_min_packs_for_130_cans_l1227_122762


namespace NUMINAMATH_CALUDE_expectation_linear_transformation_l1227_122784

variable (X : Type) [MeasurableSpace X]
variable (μ : Measure X)
variable (f : X → ℝ)

noncomputable def expectation (f : X → ℝ) : ℝ := ∫ x, f x ∂μ

theorem expectation_linear_transformation 
  (h : expectation μ f = 6) : 
  expectation μ (fun x => 3 * (f x - 2)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_expectation_linear_transformation_l1227_122784


namespace NUMINAMATH_CALUDE_yuna_has_biggest_number_l1227_122763

def yoongi_number : ℕ := 4
def yuna_number : ℕ := 5
def jungkook_number : ℕ := 6 - 3

theorem yuna_has_biggest_number :
  yuna_number > yoongi_number ∧ yuna_number > jungkook_number :=
by sorry

end NUMINAMATH_CALUDE_yuna_has_biggest_number_l1227_122763


namespace NUMINAMATH_CALUDE_son_age_proof_l1227_122754

theorem son_age_proof (son_age man_age : ℕ) : 
  man_age = son_age + 22 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 20 := by
sorry

end NUMINAMATH_CALUDE_son_age_proof_l1227_122754


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1227_122730

theorem perfect_square_condition (x : ℤ) : 
  ∃ (y : ℤ), x * (x + 1) * (x + 7) * (x + 8) = y^2 ↔ 
  x = -9 ∨ x = -8 ∨ x = -7 ∨ x = -4 ∨ x = -1 ∨ x = 0 ∨ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1227_122730


namespace NUMINAMATH_CALUDE_namjoon_has_14_pencils_l1227_122771

/-- Represents the number of pencils in a dozen -/
def dozen : ℕ := 12

/-- Represents the number of dozens Taehyung bought -/
def bought_dozens : ℕ := 2

/-- Represents the total number of pencils Taehyung bought -/
def total_pencils : ℕ := bought_dozens * dozen

/-- Represents the number of pencils Taehyung has -/
def taehyung_pencils : ℕ := total_pencils / 2

/-- Represents the number of pencils Namjoon has -/
def namjoon_pencils : ℕ := taehyung_pencils + 4

theorem namjoon_has_14_pencils : namjoon_pencils = 14 := by
  sorry

end NUMINAMATH_CALUDE_namjoon_has_14_pencils_l1227_122771


namespace NUMINAMATH_CALUDE_area_between_curves_l1227_122733

-- Define the curves
def curve1 (x y : ℝ) : Prop := y^2 = 4*x
def curve2 (x y : ℝ) : Prop := x^2 = 4*y

-- Define the bounded area
def bounded_area (A : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ A ↔ (curve1 x y ∧ x ≥ 0 ∧ y ≥ 0) ∨ (curve2 x y ∧ x ≥ 0 ∧ y ≥ 0)

-- State the theorem
theorem area_between_curves :
  ∃ (A : Set (ℝ × ℝ)), bounded_area A ∧ MeasureTheory.volume A = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_area_between_curves_l1227_122733


namespace NUMINAMATH_CALUDE_arithmetic_progression_nested_l1227_122701

/-- An arithmetic progression of distinct positive integers -/
def ArithmeticProgression (s : ℕ → ℕ) : Prop :=
  ∃ a b : ℤ, a ≠ 0 ∧ ∀ n : ℕ, s n = a * n + b

/-- The sequence is strictly increasing -/
def StrictlyIncreasing (s : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, m < n → s m < s n

/-- All elements in the sequence are positive -/
def AllPositive (s : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, 0 < s n

theorem arithmetic_progression_nested (s : ℕ → ℕ) :
  ArithmeticProgression s →
  StrictlyIncreasing s →
  AllPositive s →
  ArithmeticProgression (fun n ↦ s (s n)) ∧
  StrictlyIncreasing (fun n ↦ s (s n)) ∧
  AllPositive (fun n ↦ s (s n)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_nested_l1227_122701


namespace NUMINAMATH_CALUDE_puzzle_pieces_problem_l1227_122722

theorem puzzle_pieces_problem (pieces_first : ℕ) (pieces_second : ℕ) (pieces_third : ℕ) :
  pieces_second = pieces_third ∧
  pieces_second = (3 : ℕ) / 2 * pieces_first ∧
  pieces_first + pieces_second + pieces_third = 4000 →
  pieces_first = 1000 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_pieces_problem_l1227_122722


namespace NUMINAMATH_CALUDE_perpendicular_planes_l1227_122719

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes 
  (a b : Line) (α β : Plane) :
  perpendicular a β → 
  parallel a b → 
  contained_in b α → 
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_l1227_122719


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l1227_122774

theorem binomial_coefficient_two (n : ℕ) (h : n ≥ 2) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l1227_122774


namespace NUMINAMATH_CALUDE_triangle_properties_l1227_122783

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
  t.b = Real.sqrt 6 ∧ t.A = 2 * Real.pi / 3 ∧
  ((t.B = Real.pi / 4) ∨ (t.a = 3))

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h : triangle_conditions t) :
  t.c = (3 * Real.sqrt 2 - Real.sqrt 6) / 2 ∧
  (1 / 2 * t.b * t.c * Real.sin t.A) = (9 - 3 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1227_122783


namespace NUMINAMATH_CALUDE_collinear_points_y_value_l1227_122723

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem collinear_points_y_value :
  let A : Point := { x := 4, y := 8 }
  let B : Point := { x := 2, y := 4 }
  let C : Point := { x := 3, y := y }
  collinear A B C → y = 6 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_y_value_l1227_122723


namespace NUMINAMATH_CALUDE_total_nails_l1227_122796

/-- The number of nails each person has -/
structure NailCount where
  violet : ℕ
  tickletoe : ℕ
  sillysocks : ℕ

/-- The conditions of the nail counting problem -/
def nail_conditions (n : NailCount) : Prop :=
  n.violet = 2 * n.tickletoe + 3 ∧
  n.sillysocks = 3 * n.tickletoe - 2 ∧
  3 * n.tickletoe = 2 * n.violet ∧
  4 * n.tickletoe = 3 * n.sillysocks ∧
  n.violet = 27

/-- The theorem stating the total number of nails -/
theorem total_nails (n : NailCount) (h : nail_conditions n) : 
  n.violet + n.tickletoe + n.sillysocks = 73 := by
  sorry

end NUMINAMATH_CALUDE_total_nails_l1227_122796


namespace NUMINAMATH_CALUDE_equal_numbers_product_l1227_122770

theorem equal_numbers_product (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 20 ∧ 
  a = 22 ∧ 
  b = 34 ∧ 
  c = d → 
  c * d = 144 := by
sorry

end NUMINAMATH_CALUDE_equal_numbers_product_l1227_122770


namespace NUMINAMATH_CALUDE_container_volume_ratio_l1227_122741

theorem container_volume_ratio :
  ∀ (A B C : ℝ),
  A > 0 → B > 0 → C > 0 →
  (3/4 * A - 5/8 * B = 7/8 * C - 1/2 * C) →
  (A / C = 4/5) :=
by
  sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l1227_122741


namespace NUMINAMATH_CALUDE_circle_condition_relationship_l1227_122715

theorem circle_condition_relationship :
  (∀ x y : ℝ, x^2 + y^2 ≤ 1 → (x - 1)^2 + y^2 ≤ 4) ∧
  (∃ x y : ℝ, (x - 1)^2 + y^2 ≤ 4 ∧ x^2 + y^2 > 1) :=
by sorry

end NUMINAMATH_CALUDE_circle_condition_relationship_l1227_122715


namespace NUMINAMATH_CALUDE_adams_initial_money_l1227_122775

theorem adams_initial_money (initial_amount : ℚ) : 
  (initial_amount - 21) / 21 = 10 / 3 → initial_amount = 91 :=
by sorry

end NUMINAMATH_CALUDE_adams_initial_money_l1227_122775


namespace NUMINAMATH_CALUDE_complement_P_intersect_Q_range_of_a_when_P_subset_Q_l1227_122748

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2*a + 1}
def Q : Set ℝ := {x | x^2 - 3*x ≤ 10}

-- Statement for the first part of the problem
theorem complement_P_intersect_Q :
  (Set.univ \ P 3) ∩ Q = Set.Icc (-2) 4 := by sorry

-- Statement for the second part of the problem
theorem range_of_a_when_P_subset_Q :
  {a : ℝ | P a ∩ Q = P a} = Set.Iic 2 := by sorry

end NUMINAMATH_CALUDE_complement_P_intersect_Q_range_of_a_when_P_subset_Q_l1227_122748


namespace NUMINAMATH_CALUDE_john_initial_plays_l1227_122718

/-- The number of acts in each play -/
def acts_per_play : ℕ := 5

/-- The number of wigs John wears per act -/
def wigs_per_act : ℕ := 2

/-- The cost of each wig in dollars -/
def cost_per_wig : ℕ := 5

/-- The selling price of each wig from the dropped play in dollars -/
def selling_price_per_wig : ℕ := 4

/-- The total amount John spent in dollars -/
def total_spent : ℕ := 110

/-- The number of plays John was initially performing in -/
def initial_plays : ℕ := 3

theorem john_initial_plays :
  initial_plays * (acts_per_play * wigs_per_act * cost_per_wig) -
  (acts_per_play * wigs_per_act * selling_price_per_wig) = total_spent :=
by sorry

end NUMINAMATH_CALUDE_john_initial_plays_l1227_122718


namespace NUMINAMATH_CALUDE_max_sum_under_constraint_l1227_122726

theorem max_sum_under_constraint (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  16 * x * y * z = (x + y)^2 * (x + z)^2 →
  x + y + z ≤ 4 ∧ ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    16 * a * b * c = (a + b)^2 * (a + c)^2 ∧ a + b + c = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_under_constraint_l1227_122726


namespace NUMINAMATH_CALUDE_negation_equivalence_l1227_122759

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1227_122759


namespace NUMINAMATH_CALUDE_race_distance_proof_l1227_122751

/-- The distance between two runners at the end of a race --/
def distance_between_runners (race_length : ℕ) (arianna_position : ℕ) : ℕ :=
  race_length - arianna_position

theorem race_distance_proof :
  let race_length : ℕ := 1000  -- 1 km in meters
  let arianna_position : ℕ := 184
  distance_between_runners race_length arianna_position = 816 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_proof_l1227_122751


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l1227_122753

/-- The equation of a circle passing through three given points -/
def circle_equation (x y : ℝ) : ℝ := x^2 + y^2 - 2*x - 3*y - 3

/-- Point A coordinates -/
def A : ℝ × ℝ := (-1, 0)

/-- Point B coordinates -/
def B : ℝ × ℝ := (3, 0)

/-- Point C coordinates -/
def C : ℝ × ℝ := (1, 4)

/-- Theorem: The circle_equation passes through points A, B, and C -/
theorem circle_passes_through_points :
  circle_equation A.1 A.2 = 0 ∧
  circle_equation B.1 B.2 = 0 ∧
  circle_equation C.1 C.2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_l1227_122753


namespace NUMINAMATH_CALUDE_shirt_problem_l1227_122755

/-- Given the prices of sarees and shirts, prove the number of shirts that can be bought for $2400 -/
theorem shirt_problem (S T : ℚ) (h1 : 2 * S + 4 * T = 1600) (h2 : S + 6 * T = 1600) :
  ∃ X : ℚ, X * T = 2400 ∧ X = 12 := by
  sorry

end NUMINAMATH_CALUDE_shirt_problem_l1227_122755


namespace NUMINAMATH_CALUDE_seagull_problem_l1227_122739

theorem seagull_problem (initial : ℕ) : 
  (initial : ℚ) * (3/4) * (2/3) = 18 → initial = 36 := by
  sorry

end NUMINAMATH_CALUDE_seagull_problem_l1227_122739


namespace NUMINAMATH_CALUDE_shopkeeper_face_cards_l1227_122767

/-- The number of complete decks of playing cards the shopkeeper has -/
def num_decks : ℕ := 5

/-- The number of face cards in a standard deck of playing cards -/
def face_cards_per_deck : ℕ := 12

/-- The total number of face cards the shopkeeper has -/
def total_face_cards : ℕ := num_decks * face_cards_per_deck

theorem shopkeeper_face_cards : total_face_cards = 60 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_face_cards_l1227_122767


namespace NUMINAMATH_CALUDE_meena_cookie_sales_l1227_122757

/-- The number of dozens of cookies Meena sold to Mr. Stone -/
def cookies_sold_to_mr_stone (total_dozens : ℕ) (brock_cookies : ℕ) (katy_multiplier : ℕ) (cookies_left : ℕ) : ℕ :=
  let total_cookies := total_dozens * 12
  let katy_cookies := brock_cookies * katy_multiplier
  let sold_cookies := total_cookies - cookies_left
  let mr_stone_cookies := sold_cookies - (brock_cookies + katy_cookies)
  mr_stone_cookies / 12

theorem meena_cookie_sales : 
  cookies_sold_to_mr_stone 5 7 2 15 = 2 := by
  sorry

end NUMINAMATH_CALUDE_meena_cookie_sales_l1227_122757


namespace NUMINAMATH_CALUDE_expression_simplification_l1227_122721

theorem expression_simplification (a : ℚ) (h : a = 3) :
  (((a + 3) / (a - 1) - 1 / (a - 1)) / ((a^2 + 4*a + 4) / (a^2 - a))) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1227_122721


namespace NUMINAMATH_CALUDE_teacher_age_l1227_122716

/-- Given a class of students and their teacher, calculate the teacher's age based on how it affects the class average. -/
theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (new_avg_age : ℝ) : 
  num_students = 22 →
  student_avg_age = 21 →
  new_avg_age = student_avg_age + 1 →
  (num_students + 1) * new_avg_age - num_students * student_avg_age = 44 :=
by sorry

end NUMINAMATH_CALUDE_teacher_age_l1227_122716


namespace NUMINAMATH_CALUDE_tangent_circles_radius_l1227_122717

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
def externally_tangent (r1 r2 d : ℝ) : Prop := d = r1 + r2

theorem tangent_circles_radius (d r1 r2 : ℝ) :
  d = 8 → r1 = 3 → externally_tangent r1 r2 d → r2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_radius_l1227_122717


namespace NUMINAMATH_CALUDE_sales_decrease_equation_l1227_122778

/-- Represents the monthly decrease rate as a real number between 0 and 1 -/
def monthly_decrease_rate : ℝ := sorry

/-- The initial sales amount in August -/
def initial_sales : ℝ := 42

/-- The final sales amount in October -/
def final_sales : ℝ := 27

/-- The number of months between August and October -/
def months_elapsed : ℕ := 2

theorem sales_decrease_equation :
  initial_sales * (1 - monthly_decrease_rate) ^ months_elapsed = final_sales :=
sorry

end NUMINAMATH_CALUDE_sales_decrease_equation_l1227_122778


namespace NUMINAMATH_CALUDE_net_pay_rate_l1227_122746

-- Define the given conditions
def travel_time : ℝ := 3
def speed : ℝ := 60
def fuel_efficiency : ℝ := 30
def pay_rate : ℝ := 0.60
def gas_price : ℝ := 2.50

-- Define the theorem
theorem net_pay_rate : 
  let distance := travel_time * speed
  let gas_used := distance / fuel_efficiency
  let earnings := distance * pay_rate
  let gas_cost := gas_used * gas_price
  let net_earnings := earnings - gas_cost
  net_earnings / travel_time = 31 := by sorry

end NUMINAMATH_CALUDE_net_pay_rate_l1227_122746


namespace NUMINAMATH_CALUDE_extreme_value_condition_l1227_122785

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) - x - a * x

theorem extreme_value_condition (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≤ f a 1) ∨
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≥ f a 1) ↔
  a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_condition_l1227_122785


namespace NUMINAMATH_CALUDE_trigonometric_equation_l1227_122761

theorem trigonometric_equation (x : Real) :
  2 * Real.cos x - 3 * Real.sin x = 2 →
  Real.sin x + 3 * Real.cos x = 3 ∨ Real.sin x + 3 * Real.cos x = -31/13 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_l1227_122761


namespace NUMINAMATH_CALUDE_fair_distribution_theorem_l1227_122793

/-- Represents the outcome of a chess game -/
inductive GameOutcome
  | AWin
  | BWin

/-- Represents the state of the chess competition -/
structure ChessCompetition where
  total_games : Nat
  games_played : Nat
  a_wins : Nat
  b_wins : Nat
  prize_money : Nat

/-- Calculates the probability of player A winning the competition -/
def probability_a_wins (comp : ChessCompetition) : Rat :=
  sorry

/-- Calculates the fair distribution of prize money -/
def fair_distribution (comp : ChessCompetition) : Nat × Nat :=
  sorry

/-- Theorem stating the fair distribution of prize money -/
theorem fair_distribution_theorem (comp : ChessCompetition) 
  (h1 : comp.total_games = 7)
  (h2 : comp.games_played = 5)
  (h3 : comp.a_wins = 3)
  (h4 : comp.b_wins = 2)
  (h5 : comp.prize_money = 10000) :
  fair_distribution comp = (7500, 2500) :=
sorry

end NUMINAMATH_CALUDE_fair_distribution_theorem_l1227_122793


namespace NUMINAMATH_CALUDE_expression_evaluation_l1227_122713

theorem expression_evaluation (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x^(y+1) * y^(x-1)) / (y^y * x^x) = x^(y-x+1) * y^(x-y-1) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1227_122713


namespace NUMINAMATH_CALUDE_sum_110_is_neg_110_l1227_122787

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  /-- Sum of the first 10 terms -/
  sum_10 : ℤ
  /-- Sum of the first 100 terms -/
  sum_100 : ℤ
  /-- Property: sum of first 10 terms is 100 -/
  prop_10 : sum_10 = 100
  /-- Property: sum of first 100 terms is 10 -/
  prop_100 : sum_100 = 10

/-- Theorem: For the given arithmetic sequence, the sum of the first 110 terms is -110 -/
theorem sum_110_is_neg_110 (seq : ArithmeticSequence) : ℤ :=
  -110

#check sum_110_is_neg_110

end NUMINAMATH_CALUDE_sum_110_is_neg_110_l1227_122787


namespace NUMINAMATH_CALUDE_triangle_properties_l1227_122706

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Given conditions
  (a * Real.cos B - b * Real.cos A = c - b) →
  (Real.tan A + Real.tan B + Real.tan C - Real.sqrt 3 * Real.tan B * Real.tan C = 0) →
  ((1/2) * a * (b * Real.sin B + c * Real.sin C - a * Real.sin A) = (1/2) * a * b * Real.sin C) →
  -- Conclusions
  (A = π/3) ∧
  (a = 8 → (1/2) * a * b * Real.sin C = 11 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1227_122706


namespace NUMINAMATH_CALUDE_single_elimination_tournament_games_l1227_122747

/-- The number of games played in a single-elimination tournament. -/
def gamesPlayed (n : ℕ) : ℕ :=
  n - 1

/-- Theorem: A single-elimination tournament with 21 teams requires 20 games. -/
theorem single_elimination_tournament_games :
  gamesPlayed 21 = 20 := by
  sorry

end NUMINAMATH_CALUDE_single_elimination_tournament_games_l1227_122747


namespace NUMINAMATH_CALUDE_ratio_of_13th_terms_l1227_122700

/-- Two arithmetic sequences with sums U_n and V_n for the first n terms -/
def arithmetic_sequences (U V : ℕ → ℚ) : Prop :=
  ∃ (a b c d : ℚ), ∀ n : ℕ,
    U n = n * (2 * a + (n - 1) * b) / 2 ∧
    V n = n * (2 * c + (n - 1) * d) / 2

/-- The ratio condition for U_n and V_n -/
def ratio_condition (U V : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, U n * (3 * n + 17) = V n * (5 * n + 3)

/-- The 13th term of an arithmetic sequence -/
def term_13 (seq : ℕ → ℚ) : ℚ :=
  seq 13 - seq 12

/-- Main theorem -/
theorem ratio_of_13th_terms
  (U V : ℕ → ℚ)
  (h1 : arithmetic_sequences U V)
  (h2 : ratio_condition U V) :
  term_13 U / term_13 V = 52 / 89 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_13th_terms_l1227_122700


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1227_122708

theorem geometric_sequence_product (a b : ℝ) :
  (∃ r : ℝ, r ≠ 0 ∧ -1 = -1 ∧ a = -1 * r ∧ b = a * r ∧ 2 = b * r) →
  a * b = -2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1227_122708


namespace NUMINAMATH_CALUDE_simplify_expression_l1227_122760

theorem simplify_expression (a b : ℝ) (h : a + b ≠ 1) :
  1 - (1 / (1 + (a + b) / (1 - a - b))) = a + b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1227_122760


namespace NUMINAMATH_CALUDE_shaded_squares_correct_l1227_122735

/-- Given a square grid with odd side length, calculates the number of shaded squares along the two diagonals -/
def shadedSquares (n : ℕ) : ℕ :=
  2 * n - 1

theorem shaded_squares_correct (n : ℕ) (h : Odd n) :
  shadedSquares n = 2 * n - 1 := by
  sorry

#eval shadedSquares 7  -- Expected: 13
#eval shadedSquares 101  -- Expected: 201

end NUMINAMATH_CALUDE_shaded_squares_correct_l1227_122735


namespace NUMINAMATH_CALUDE_average_of_numbers_is_ten_l1227_122764

def numbers : List ℝ := [6, 8, 9, 11, 16]

theorem average_of_numbers_is_ten :
  (List.sum numbers) / (List.length numbers) = 10 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_is_ten_l1227_122764


namespace NUMINAMATH_CALUDE_eighteen_times_two_minus_four_l1227_122791

theorem eighteen_times_two_minus_four (x : ℝ) : x * 2 = 18 → x - 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_times_two_minus_four_l1227_122791


namespace NUMINAMATH_CALUDE_smallest_product_l1227_122740

def S : Set Int := {-10, -3, 0, 2, 6}

theorem smallest_product (a b : Int) (ha : a ∈ S) (hb : b ∈ S) :
  ∃ (x y : Int), x ∈ S ∧ y ∈ S ∧ x * y ≤ a * b ∧ x * y = -60 :=
sorry

end NUMINAMATH_CALUDE_smallest_product_l1227_122740


namespace NUMINAMATH_CALUDE_average_of_last_three_l1227_122780

theorem average_of_last_three (numbers : Fin 6 → ℝ) 
  (h1 : (numbers 0 + numbers 1 + numbers 2 + numbers 3 + numbers 4 + numbers 5) / 6 = 30)
  (h2 : (numbers 0 + numbers 1 + numbers 2 + numbers 3) / 4 = 25)
  (h3 : numbers 3 = 25) :
  (numbers 3 + numbers 4 + numbers 5) / 3 = 35 := by
sorry

end NUMINAMATH_CALUDE_average_of_last_three_l1227_122780


namespace NUMINAMATH_CALUDE_point_not_on_line_l1227_122705

theorem point_not_on_line (a c : ℝ) (h : a * c > 0) :
  ¬(∃ (x y : ℝ), x = 2500 ∧ y = 0 ∧ y = a * x + c) :=
by sorry

end NUMINAMATH_CALUDE_point_not_on_line_l1227_122705


namespace NUMINAMATH_CALUDE_rachel_painting_time_l1227_122738

/-- Prove that Rachel's painting time is 13 hours -/
theorem rachel_painting_time : ℝ → ℝ → ℝ → Prop :=
  fun matt_time patty_time rachel_time =>
    matt_time = 12 ∧
    patty_time = matt_time / 3 ∧
    rachel_time = 2 * patty_time + 5 →
    rachel_time = 13

/-- Proof of the theorem -/
lemma rachel_painting_time_proof : rachel_painting_time 12 4 13 := by
  sorry


end NUMINAMATH_CALUDE_rachel_painting_time_l1227_122738


namespace NUMINAMATH_CALUDE_sequence_problem_l1227_122737

def arithmetic_sequence (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def geometric_sequence (a b c d : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c

theorem sequence_problem (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (h1 : arithmetic_sequence 1 a₁ a₂ a₃)
  (h2 : arithmetic_sequence a₁ a₂ a₃ 9)
  (h3 : geometric_sequence (-9) b₁ b₂ b₃)
  (h4 : geometric_sequence b₁ b₂ b₃ (-1)) :
  b₂ / (a₁ + a₃) = -3/10 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l1227_122737


namespace NUMINAMATH_CALUDE_bailey_chew_toys_l1227_122786

theorem bailey_chew_toys (dog_treats rawhide_bones credit_cards items_per_charge : ℕ) 
  (h1 : dog_treats = 8)
  (h2 : rawhide_bones = 10)
  (h3 : credit_cards = 4)
  (h4 : items_per_charge = 5) :
  credit_cards * items_per_charge - (dog_treats + rawhide_bones) = 2 := by
  sorry

end NUMINAMATH_CALUDE_bailey_chew_toys_l1227_122786


namespace NUMINAMATH_CALUDE_smallest_valid_n_l1227_122703

def is_valid (n : ℕ) : Prop :=
  ∃ k₁ k₂ : ℕ, 1 ≤ k₁ ∧ k₁ ≤ n ∧ 1 ≤ k₂ ∧ k₂ ≤ n ∧
  (n^2 + n) % k₁ = 0 ∧ (n^2 + n) % k₂ ≠ 0

theorem smallest_valid_n :
  is_valid 4 ∧ ∀ m : ℕ, 0 < m ∧ m < 4 → ¬is_valid m :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_n_l1227_122703


namespace NUMINAMATH_CALUDE_eighteenth_permutation_l1227_122744

def FourDigitPermutation : Type := Fin 4 → Fin 10

def isValidPermutation (p : FourDigitPermutation) : Prop :=
  (p 0 = 1 ∨ p 0 = 2 ∨ p 0 = 5 ∨ p 0 = 6) ∧
  (p 1 = 1 ∨ p 1 = 2 ∨ p 1 = 5 ∨ p 1 = 6) ∧
  (p 2 = 1 ∨ p 2 = 2 ∨ p 2 = 5 ∨ p 2 = 6) ∧
  (p 3 = 1 ∨ p 3 = 2 ∨ p 3 = 5 ∨ p 3 = 6) ∧
  (p 0 ≠ p 1) ∧ (p 0 ≠ p 2) ∧ (p 0 ≠ p 3) ∧
  (p 1 ≠ p 2) ∧ (p 1 ≠ p 3) ∧ (p 2 ≠ p 3)

def toInteger (p : FourDigitPermutation) : ℕ :=
  1000 * (p 0).val + 100 * (p 1).val + 10 * (p 2).val + (p 3).val

def isOrdered (p q : FourDigitPermutation) : Prop :=
  toInteger p ≤ toInteger q

theorem eighteenth_permutation :
  ∃ (perms : List FourDigitPermutation),
    (∀ p ∈ perms, isValidPermutation p) ∧
    (perms.length = 24) ∧
    (∀ i j, i < j → isOrdered (perms.get ⟨i, by sorry⟩) (perms.get ⟨j, by sorry⟩)) ∧
    (toInteger (perms.get ⟨17, by sorry⟩) = 5621) :=
  sorry

end NUMINAMATH_CALUDE_eighteenth_permutation_l1227_122744


namespace NUMINAMATH_CALUDE_product_of_real_parts_l1227_122702

theorem product_of_real_parts (x : ℂ) : 
  x^2 + 4*x = -1 + Complex.I → 
  (∃ x₁ x₂ : ℂ, x₁ ≠ x₂ ∧ x₁^2 + 4*x₁ = -1 + Complex.I ∧ x₂^2 + 4*x₂ = -1 + Complex.I ∧ 
    (x₁.re * x₂.re = (1 + 3 * Real.sqrt 10) / 2)) := by
  sorry

end NUMINAMATH_CALUDE_product_of_real_parts_l1227_122702


namespace NUMINAMATH_CALUDE_divisibility_by_ten_l1227_122743

theorem divisibility_by_ten (x y : Nat) : 
  x < 10 → y < 10 → x + y = 2 → (65300 + 10 * x + y) % 10 = 0 → x = 2 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_ten_l1227_122743


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l1227_122729

/-- The largest possible angle in a triangle with two sides of length 2 and the third side greater than 4 --/
theorem largest_angle_in_special_triangle :
  ∀ (a b c : ℝ) (C : ℝ),
    a = 2 →
    b = 2 →
    c > 4 →
    C = Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) →
    ∀ ε > 0, C < 180 - ε := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l1227_122729


namespace NUMINAMATH_CALUDE_triangle_exists_l1227_122732

/-- Represents a point in 2D space with integer coordinates -/
structure Point :=
  (x : Int) (y : Int)

/-- Calculates the square of the distance between two points -/
def distanceSquared (p q : Point) : Int :=
  (p.x - q.x)^2 + (p.y - q.y)^2

/-- Calculates the area of a triangle given its three vertices -/
def triangleArea (a b c : Point) : Rat :=
  let det := a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)
  Rat.ofInt (abs det) / 2

/-- Theorem stating the existence of a triangle with the specified properties -/
theorem triangle_exists : ∃ (a b c : Point),
  (triangleArea a b c < 1) ∧
  (distanceSquared a b > 4) ∧
  (distanceSquared b c > 4) ∧
  (distanceSquared c a > 4) :=
sorry

end NUMINAMATH_CALUDE_triangle_exists_l1227_122732


namespace NUMINAMATH_CALUDE_i_times_one_plus_i_l1227_122707

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem i_times_one_plus_i : i * (1 + i) = i - 1 := by
  sorry

end NUMINAMATH_CALUDE_i_times_one_plus_i_l1227_122707


namespace NUMINAMATH_CALUDE_constant_term_theorem_l1227_122720

theorem constant_term_theorem (m : ℝ) : 
  (∀ x, (x - m) * (x + 7) = x^2 + (7 - m) * x - 7 * m) →
  -7 * m = 14 →
  m = -2 := by
sorry

end NUMINAMATH_CALUDE_constant_term_theorem_l1227_122720


namespace NUMINAMATH_CALUDE_distinct_triangles_in_grid_l1227_122794

/-- The number of points in a 3 x 2 grid -/
def total_points : ℕ := 6

/-- The number of points needed to form a triangle -/
def points_per_triangle : ℕ := 3

/-- The number of rows in the grid -/
def num_rows : ℕ := 3

/-- Function to calculate combinations -/
def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else (List.range n).foldl (λ acc i => acc * (n - i) / (i + 1)) 1

/-- The number of degenerate cases (collinear points in rows) -/
def degenerate_cases : ℕ := num_rows

/-- Theorem: The number of distinct triangles in a 3 x 2 grid is 17 -/
theorem distinct_triangles_in_grid :
  choose total_points points_per_triangle - degenerate_cases = 17 := by
  sorry


end NUMINAMATH_CALUDE_distinct_triangles_in_grid_l1227_122794


namespace NUMINAMATH_CALUDE_ben_win_probability_l1227_122728

theorem ben_win_probability (p_loss p_tie : ℚ) 
  (h_loss : p_loss = 5/12)
  (h_tie : p_tie = 1/6)
  (h_sum : p_loss + p_tie + (1 - p_loss - p_tie) = 1) :
  1 - p_loss - p_tie = 5/12 := by
sorry

end NUMINAMATH_CALUDE_ben_win_probability_l1227_122728


namespace NUMINAMATH_CALUDE_solution_set_f_leq_6_range_of_m_l1227_122776

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| + |2*x - 3|

-- Theorem for the solution set of f(x) ≤ 6
theorem solution_set_f_leq_6 :
  {x : ℝ | f x ≤ 6} = Set.Icc (-1/2) (5/2) := by sorry

-- Theorem for the range of m
theorem range_of_m :
  {m : ℝ | ∀ x, 6*m^2 - 4*m < f x} = Set.Ioo (-1/3) 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_6_range_of_m_l1227_122776


namespace NUMINAMATH_CALUDE_arithmetic_sequence_constant_multiple_l1227_122779

/-- An arithmetic sequence with common difference d -/
def ArithmeticSequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a (n + 1) - a n = d

/-- The constant multiple of a sequence -/
def ConstantMultiple (a : ℕ → ℝ) (c : ℝ) : ℕ → ℝ :=
  fun n => c * a n

theorem arithmetic_sequence_constant_multiple
  (a : ℕ → ℝ) (d c : ℝ) (hc : c ≠ 0) (ha : ArithmeticSequence a d) :
  ArithmeticSequence (ConstantMultiple a c) (c * d) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_constant_multiple_l1227_122779


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1227_122789

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 + 1) + abs x

theorem inequality_solution_set (x : ℝ) : 
  f (2*x - 1) > f (x + 1) ↔ x < 0 ∨ x > 2 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1227_122789


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1227_122704

-- Define a triangle with side lengths a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_inequality_ab : a < b + c
  triangle_inequality_bc : b < a + c
  triangle_inequality_ca : c < a + b

-- Define what it means for an angle to be acute in terms of side lengths
def is_angle_A_acute (t : Triangle) : Prop :=
  t.b ^ 2 + t.c ^ 2 > t.a ^ 2

-- Define the condition a ≤ (b + c) / 2
def condition (t : Triangle) : Prop :=
  t.a ≤ (t.b + t.c) / 2

-- Theorem statement
theorem condition_sufficient_not_necessary :
  (∀ t : Triangle, condition t → is_angle_A_acute t) ∧
  ¬(∀ t : Triangle, is_angle_A_acute t → condition t) :=
sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1227_122704
