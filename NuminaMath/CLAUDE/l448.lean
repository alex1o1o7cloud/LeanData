import Mathlib

namespace NUMINAMATH_CALUDE_number_of_students_l448_44811

/-- The number of storybooks available for distribution -/
def total_books : ℕ := 60

/-- Predicate to check if there are books left over after initial distribution -/
def has_leftover (n : ℕ) : Prop := n < total_books

/-- Predicate to check if remaining books can be evenly distributed with 2 students sharing 1 book -/
def can_evenly_distribute_remainder (n : ℕ) : Prop :=
  ∃ k : ℕ, total_books - n = 2 * k

/-- The theorem stating the number of students in the class -/
theorem number_of_students :
  ∃ n : ℕ, n = 40 ∧ 
    has_leftover n ∧ 
    can_evenly_distribute_remainder n :=
sorry

end NUMINAMATH_CALUDE_number_of_students_l448_44811


namespace NUMINAMATH_CALUDE_jar_red_marble_difference_l448_44836

-- Define the ratios for each jar
def jar_a_ratio : Rat := 5 / 3
def jar_b_ratio : Rat := 3 / 2

-- Define the total number of white marbles
def total_white_marbles : ℕ := 70

-- Theorem statement
theorem jar_red_marble_difference :
  ∃ (total_marbles : ℕ) (jar_a_red jar_a_white jar_b_red jar_b_white : ℕ),
    -- Both jars have equal number of marbles
    jar_a_red + jar_a_white = total_marbles ∧
    jar_b_red + jar_b_white = total_marbles ∧
    -- Ratio conditions
    jar_a_red / jar_a_white = jar_a_ratio ∧
    jar_b_red / jar_b_white = jar_b_ratio ∧
    -- Total white marbles condition
    jar_a_white + jar_b_white = total_white_marbles ∧
    -- Difference in red marbles
    jar_a_red - jar_b_red = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_jar_red_marble_difference_l448_44836


namespace NUMINAMATH_CALUDE_parrots_per_cage_l448_44827

theorem parrots_per_cage (num_cages : ℕ) (parakeets_per_cage : ℕ) (total_birds : ℕ) :
  num_cages = 8 →
  parakeets_per_cage = 7 →
  total_birds = 72 →
  ∃ (parrots_per_cage : ℕ),
    parrots_per_cage * num_cages + parakeets_per_cage * num_cages = total_birds ∧
    parrots_per_cage = 2 :=
by sorry

end NUMINAMATH_CALUDE_parrots_per_cage_l448_44827


namespace NUMINAMATH_CALUDE_kevins_weekly_revenue_l448_44815

/-- Calculates the total weekly revenue for a fruit vendor --/
def total_weekly_revenue (total_crates price_grapes price_mangoes price_passion 
                          crates_grapes crates_mangoes : ℕ) : ℕ :=
  let crates_passion := total_crates - crates_grapes - crates_mangoes
  let revenue_grapes := crates_grapes * price_grapes
  let revenue_mangoes := crates_mangoes * price_mangoes
  let revenue_passion := crates_passion * price_passion
  revenue_grapes + revenue_mangoes + revenue_passion

/-- Theorem stating that Kevin's total weekly revenue is $1020 --/
theorem kevins_weekly_revenue : 
  total_weekly_revenue 50 15 20 25 13 20 = 1020 := by
  sorry

#eval total_weekly_revenue 50 15 20 25 13 20

end NUMINAMATH_CALUDE_kevins_weekly_revenue_l448_44815


namespace NUMINAMATH_CALUDE_james_puzzles_l448_44859

/-- Calculates the number of puzzles James bought given the puzzle size, completion rate, and total time --/
theorem james_puzzles (puzzle_size : ℕ) (pieces_per_interval : ℕ) (interval_minutes : ℕ) (total_minutes : ℕ) :
  puzzle_size = 2000 →
  pieces_per_interval = 100 →
  interval_minutes = 10 →
  total_minutes = 400 →
  (total_minutes / interval_minutes) * pieces_per_interval / puzzle_size = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_james_puzzles_l448_44859


namespace NUMINAMATH_CALUDE_ellipse_theorem_proof_l448_44801

noncomputable def ellipse_theorem (a b : ℝ) (h : a > b ∧ b > 0) : Prop :=
  let e := Real.sqrt (a^2 - b^2)
  ∀ (M : ℝ × ℝ),
    M.1^2 / a^2 + M.2^2 / b^2 = 1 →
    let F₁ : ℝ × ℝ := (-e, 0)
    let F₂ : ℝ × ℝ := (e, 0)
    ∃ (A B : ℝ × ℝ),
      A.1^2 / a^2 + A.2^2 / b^2 = 1 ∧
      B.1^2 / a^2 + B.2^2 / b^2 = 1 ∧
      (∃ t : ℝ, A = M + t • (M - F₁)) ∧
      (∃ s : ℝ, B = M + s • (M - F₂)) →
      (b^2 / a^2) * (‖M - F₁‖ / ‖F₁ - A‖ + ‖M - F₂‖ / ‖F₂ - B‖ + 2) = 4

theorem ellipse_theorem_proof (a b : ℝ) (h : a > b ∧ b > 0) :
  ellipse_theorem a b h := by
  sorry

end NUMINAMATH_CALUDE_ellipse_theorem_proof_l448_44801


namespace NUMINAMATH_CALUDE_parallelogram_side_length_l448_44881

theorem parallelogram_side_length (s : ℝ) : 
  s > 0 → -- Ensure s is positive
  let side1 := 3 * s
  let side2 := s
  let angle := 30 * π / 180 -- Convert 30 degrees to radians
  let area := side1 * side2 * Real.sin angle
  area = 9 * Real.sqrt 3 → s = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_side_length_l448_44881


namespace NUMINAMATH_CALUDE_inequality_proof_l448_44860

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : (a + b) * (b + c) * (c + a) = 1) : 
  (a^2 / (1 + Real.sqrt (b * c))) + (b^2 / (1 + Real.sqrt (c * a))) + (c^2 / (1 + Real.sqrt (a * b))) ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l448_44860


namespace NUMINAMATH_CALUDE_equal_to_mac_ratio_l448_44818

/-- Represents the survey results of computer brand preferences among college students. -/
structure SurveyResults where
  total : ℕ
  mac_preference : ℕ
  no_preference : ℕ
  windows_preference : ℕ

/-- Calculates the number of students who equally preferred both brands. -/
def equal_preference (s : SurveyResults) : ℕ :=
  s.total - (s.mac_preference + s.no_preference + s.windows_preference)

/-- Represents a ratio as a pair of natural numbers. -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Theorem stating the ratio of students who equally preferred both brands
    to students who preferred Mac to Windows. -/
theorem equal_to_mac_ratio (s : SurveyResults)
  (h_total : s.total = 210)
  (h_mac : s.mac_preference = 60)
  (h_no_pref : s.no_preference = 90)
  (h_windows : s.windows_preference = 40) :
  ∃ (r : Ratio), r.numerator = 1 ∧ r.denominator = 3 ∧
  r.numerator * s.mac_preference = r.denominator * equal_preference s :=
sorry

end NUMINAMATH_CALUDE_equal_to_mac_ratio_l448_44818


namespace NUMINAMATH_CALUDE_yanna_kept_36_apples_l448_44803

/-- The number of apples Yanna bought -/
def total_apples : ℕ := 60

/-- The number of apples Yanna gave to Zenny -/
def apples_to_zenny : ℕ := 18

/-- The number of apples Yanna gave to Andrea -/
def apples_to_andrea : ℕ := 6

/-- The number of apples Yanna kept -/
def apples_kept : ℕ := total_apples - (apples_to_zenny + apples_to_andrea)

theorem yanna_kept_36_apples : apples_kept = 36 := by
  sorry

end NUMINAMATH_CALUDE_yanna_kept_36_apples_l448_44803


namespace NUMINAMATH_CALUDE_square_land_equation_l448_44893

theorem square_land_equation (a p : ℝ) (h1 : p = 36) : 
  (∃ s : ℝ, s > 0 ∧ a = s^2 ∧ p = 4*s) → 
  (5*a = 10*p + 45) := by
sorry

end NUMINAMATH_CALUDE_square_land_equation_l448_44893


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l448_44856

-- Define the points A, B, and C
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (4, 2)
def C : ℝ × ℝ := (1, -7)

-- Define circle M passing through A, B, and C
def circle_M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 2)^2 = 25}

-- Define the y-axis
def y_axis : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 0}

-- Define the line on which the center of circle N moves
def center_line : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1 - p.2 + 6 = 0}

-- Define circle N with radius 10 and center (a, 2a + 6)
def circle_N (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + (p.2 - (2 * a + 6))^2 = 100}

-- Define the theorem
theorem circle_intersection_theorem :
  ∃ (P Q : ℝ × ℝ) (a : Set ℝ),
    P ∈ circle_M ∧ P ∈ y_axis ∧
    Q ∈ circle_M ∧ Q ∈ y_axis ∧
    (Q.2 - P.2)^2 = 96 ∧
    (∀ x ∈ a, ∃ y, (x, y) ∈ center_line ∧ (circle_N x ∩ circle_M).Nonempty) ∧
    a = {x : ℝ | -3 - Real.sqrt 41 ≤ x ∧ x ≤ -4 ∨ -2 ≤ x ∧ x ≤ -3 + Real.sqrt 41} :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l448_44856


namespace NUMINAMATH_CALUDE_ellipse_and_outer_point_properties_l448_44883

/-- Definition of an ellipse C with given properties -/
structure Ellipse :=
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : a^2 - b^2 = 5)
  (h4 : a = 3)

/-- Definition of a point P outside the ellipse -/
structure OuterPoint (C : Ellipse) :=
  (x₀ y₀ : ℝ)
  (h5 : x₀^2 / C.a^2 + y₀^2 / C.b^2 > 1)

/-- Theorem stating the properties of the ellipse and outer point -/
theorem ellipse_and_outer_point_properties (C : Ellipse) (P : OuterPoint C) :
  (∀ x y, x^2 / 9 + y^2 / 4 = 1 ↔ x^2 / C.a^2 + y^2 / C.b^2 = 1) ∧
  (P.x₀^2 + P.y₀^2 = 13) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_outer_point_properties_l448_44883


namespace NUMINAMATH_CALUDE_probability_blue_after_removal_l448_44861

/-- Probability of pulling a blue ball after removal -/
theorem probability_blue_after_removal (initial_total : ℕ) (initial_blue : ℕ) (removed_blue : ℕ) :
  initial_total = 18 →
  initial_blue = 6 →
  removed_blue = 3 →
  (initial_blue - removed_blue : ℚ) / (initial_total - removed_blue) = 1 / 5 := by
  sorry


end NUMINAMATH_CALUDE_probability_blue_after_removal_l448_44861


namespace NUMINAMATH_CALUDE_outfit_combinations_l448_44849

theorem outfit_combinations (shirts : Nat) (ties : Nat) (hats : Nat) :
  shirts = 8 → ties = 6 → hats = 4 → shirts * ties * hats = 192 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l448_44849


namespace NUMINAMATH_CALUDE_print_shop_price_X_l448_44809

/-- The price per color copy at print shop Y -/
def price_Y : ℝ := 1.70

/-- The number of copies in the comparison -/
def num_copies : ℕ := 70

/-- The price difference between shops Y and X for 70 copies -/
def price_difference : ℝ := 35

/-- The price per color copy at print shop X -/
def price_X : ℝ := 1.20

theorem print_shop_price_X :
  price_X = (price_Y * num_copies - price_difference) / num_copies :=
by sorry

end NUMINAMATH_CALUDE_print_shop_price_X_l448_44809


namespace NUMINAMATH_CALUDE_barney_situp_time_l448_44843

/-- The number of sit-ups Barney can perform in one minute -/
def barney_situps_per_minute : ℕ := 45

/-- The number of minutes Barney does sit-ups -/
def barney_minutes : ℕ := 1

/-- The number of minutes Carrie does sit-ups -/
def carrie_minutes : ℕ := 2

/-- The number of minutes Jerrie does sit-ups -/
def jerrie_minutes : ℕ := 3

/-- The total number of sit-ups performed by all three -/
def total_situps : ℕ := 510

/-- Theorem stating that given the conditions, Barney did sit-ups for 1 minute -/
theorem barney_situp_time : 
  barney_situps_per_minute * barney_minutes + 
  (2 * barney_situps_per_minute) * carrie_minutes + 
  (2 * barney_situps_per_minute + 5) * jerrie_minutes = total_situps :=
by sorry

end NUMINAMATH_CALUDE_barney_situp_time_l448_44843


namespace NUMINAMATH_CALUDE_stating_race_result_l448_44832

/-- Represents a runner in the race -/
inductive Runner
| Primus
| Secundus
| Tertius

/-- Represents the order of runners -/
def RunnerOrder := List Runner

/-- The number of place changes between pairs of runners -/
structure PlaceChanges where
  primus_secundus : Nat
  secundus_tertius : Nat
  primus_tertius : Nat

/-- The initial order of runners -/
def initial_order : RunnerOrder := [Runner.Primus, Runner.Secundus, Runner.Tertius]

/-- The place changes during the race -/
def race_changes : PlaceChanges := {
  primus_secundus := 9,
  secundus_tertius := 10,
  primus_tertius := 11
}

/-- The final order of runners -/
def final_order : RunnerOrder := [Runner.Secundus, Runner.Tertius, Runner.Primus]

/-- 
Theorem stating that given the initial order and place changes,
the final order is [Secundus, Tertius, Primus]
-/
theorem race_result (order : RunnerOrder) (changes : PlaceChanges) :
  order = initial_order ∧ changes = race_changes →
  final_order = [Runner.Secundus, Runner.Tertius, Runner.Primus] :=
by sorry

end NUMINAMATH_CALUDE_stating_race_result_l448_44832


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l448_44890

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - k * x - 2 * x + 12 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - k * y - 2 * y + 12 = 0 → y = x) ↔ 
  (k = 10 ∨ k = -14) := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l448_44890


namespace NUMINAMATH_CALUDE_henry_twice_jill_age_l448_44800

/-- Represents the number of years ago when Henry was twice Jill's age. -/
def years_ago : ℕ := 9

/-- Henry's present age -/
def henry_age : ℕ := 29

/-- Jill's present age -/
def jill_age : ℕ := 19

theorem henry_twice_jill_age :
  (henry_age + jill_age = 48) →
  (henry_age - years_ago = 2 * (jill_age - years_ago)) := by
  sorry

end NUMINAMATH_CALUDE_henry_twice_jill_age_l448_44800


namespace NUMINAMATH_CALUDE_rug_area_l448_44839

theorem rug_area (w l : ℝ) (h1 : l = w + 8) 
  (h2 : (w + 16) * (l + 16) - w * l = 704) : w * l = 180 := by
  sorry

end NUMINAMATH_CALUDE_rug_area_l448_44839


namespace NUMINAMATH_CALUDE_wilsons_theorem_l448_44847

theorem wilsons_theorem (p : ℕ) (h : p ≥ 2) :
  Nat.Prime p ↔ (Nat.factorial (p - 1) % p = p - 1) := by
  sorry

end NUMINAMATH_CALUDE_wilsons_theorem_l448_44847


namespace NUMINAMATH_CALUDE_product_equality_l448_44825

theorem product_equality : 
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l448_44825


namespace NUMINAMATH_CALUDE_pentagon_largest_angle_l448_44845

theorem pentagon_largest_angle (P Q R S T : ℝ) : 
  P = 75 →
  Q = 110 →
  R = S →
  T = 3 * R - 20 →
  P + Q + R + S + T = 540 →
  max P (max Q (max R (max S T))) = 217 :=
sorry

end NUMINAMATH_CALUDE_pentagon_largest_angle_l448_44845


namespace NUMINAMATH_CALUDE_fixed_internet_charge_l448_44865

/-- Represents Elvin's monthly telephone bill structure -/
structure MonthlyBill where
  callCharge : ℝ
  internetCharge : ℝ
  totalCharge : ℝ
  total_eq : totalCharge = callCharge + internetCharge

/-- Theorem stating the fixed monthly internet charge -/
theorem fixed_internet_charge 
  (jan : MonthlyBill) 
  (feb : MonthlyBill) 
  (jan_total : jan.totalCharge = 46)
  (feb_total : feb.totalCharge = 76)
  (feb_call_charge : feb.callCharge = 2 * jan.callCharge)
  : jan.internetCharge = 16 := by
  sorry

end NUMINAMATH_CALUDE_fixed_internet_charge_l448_44865


namespace NUMINAMATH_CALUDE_zero_sponsorship_prob_high_sponsorship_prob_l448_44896

-- Define the number of students and experts
def num_students : ℕ := 3
def num_experts : ℕ := 2

-- Define the probability of a "support" review
def support_prob : ℚ := 1/2

-- Define the function to calculate the probability of k successes in n trials
def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

-- Theorem for the probability of zero total sponsorship
theorem zero_sponsorship_prob :
  binomial_prob (num_students * num_experts) 0 support_prob = 1/64 := by sorry

-- Theorem for the probability of sponsorship exceeding 150,000 yuan
theorem high_sponsorship_prob :
  (binomial_prob (num_students * num_experts) 4 support_prob +
   binomial_prob (num_students * num_experts) 5 support_prob +
   binomial_prob (num_students * num_experts) 6 support_prob) = 11/32 := by sorry

end NUMINAMATH_CALUDE_zero_sponsorship_prob_high_sponsorship_prob_l448_44896


namespace NUMINAMATH_CALUDE_gravel_calculation_l448_44819

/-- The amount of gravel bought by a construction company -/
def gravel_amount : ℝ := 14.02 - 8.11

/-- The total amount of material bought by the construction company -/
def total_material : ℝ := 14.02

/-- The amount of sand bought by the construction company -/
def sand_amount : ℝ := 8.11

theorem gravel_calculation :
  gravel_amount = 5.91 ∧
  total_material = gravel_amount + sand_amount :=
sorry

end NUMINAMATH_CALUDE_gravel_calculation_l448_44819


namespace NUMINAMATH_CALUDE_tan_y_plus_pi_third_l448_44841

theorem tan_y_plus_pi_third (y : ℝ) (h : Real.tan y = -1) : 
  Real.tan (y + π/3) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_y_plus_pi_third_l448_44841


namespace NUMINAMATH_CALUDE_school_purchase_cost_l448_44889

/-- The total cost of purchasing sweaters and sports shirts -/
def total_cost (sweater_price : ℕ) (sweater_quantity : ℕ) 
               (shirt_price : ℕ) (shirt_quantity : ℕ) : ℕ :=
  sweater_price * sweater_quantity + shirt_price * shirt_quantity

/-- Theorem stating that the total cost for the given quantities and prices is 5400 yuan -/
theorem school_purchase_cost : 
  total_cost 98 25 59 50 = 5400 := by
  sorry

end NUMINAMATH_CALUDE_school_purchase_cost_l448_44889


namespace NUMINAMATH_CALUDE_new_person_weight_l448_44814

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 4.5 →
  replaced_weight = 65 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 101 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l448_44814


namespace NUMINAMATH_CALUDE_forecast_variation_determinants_l448_44822

/-- Represents a variable in regression analysis -/
inductive RegressionVariable
  | Forecast
  | Explanatory
  | Residual

/-- Represents the components that determine the variation of a variable -/
structure VariationDeterminants where
  components : List RegressionVariable

/-- Axiom: In regression analysis, the variation of the forecast variable
    is determined by both explanatory and residual variables -/
axiom regression_variation_determinants :
  VariationDeterminants.components (VariationDeterminants.mk [RegressionVariable.Explanatory, RegressionVariable.Residual]) =
  (VariationDeterminants.mk [RegressionVariable.Explanatory, RegressionVariable.Residual]).components

/-- Theorem: The variation of the forecast variable in regression analysis
    is determined by both explanatory and residual variables -/
theorem forecast_variation_determinants :
  VariationDeterminants.components (VariationDeterminants.mk [RegressionVariable.Explanatory, RegressionVariable.Residual]) =
  (VariationDeterminants.mk [RegressionVariable.Explanatory, RegressionVariable.Residual]).components :=
by sorry

end NUMINAMATH_CALUDE_forecast_variation_determinants_l448_44822


namespace NUMINAMATH_CALUDE_negation_of_exists_positive_power_l448_44852

theorem negation_of_exists_positive_power (x : ℝ) : 
  (¬ (∃ x < 0, 2^x > 0)) ↔ (∀ x < 0, 2^x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_exists_positive_power_l448_44852


namespace NUMINAMATH_CALUDE_rationalize_denominator_sqrt3_minus1_l448_44824

theorem rationalize_denominator_sqrt3_minus1 : 
  (1 : ℝ) / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2 := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_sqrt3_minus1_l448_44824


namespace NUMINAMATH_CALUDE_book_pages_l448_44855

/-- The number of pages Hallie read on the first day -/
def pages_day1 : ℕ := 63

/-- The number of pages Hallie read on the second day -/
def pages_day2 : ℕ := 2 * pages_day1

/-- The number of pages Hallie read on the third day -/
def pages_day3 : ℕ := pages_day2 + 10

/-- The number of pages Hallie read on the fourth day -/
def pages_day4 : ℕ := 29

/-- The total number of pages in the book -/
def total_pages : ℕ := pages_day1 + pages_day2 + pages_day3 + pages_day4

theorem book_pages : total_pages = 354 := by sorry

end NUMINAMATH_CALUDE_book_pages_l448_44855


namespace NUMINAMATH_CALUDE_pencil_count_l448_44867

theorem pencil_count (initial : ℕ) (added : ℕ) (total : ℕ) : 
  initial = 27 → added = 45 → total = initial + added → total = 72 := by sorry

end NUMINAMATH_CALUDE_pencil_count_l448_44867


namespace NUMINAMATH_CALUDE_min_value_z3_l448_44891

open Complex

theorem min_value_z3 (z₁ z₂ z₃ : ℂ) 
  (h_im : (z₁ / z₂).im ≠ 0 ∧ (z₁ / z₂).re = 0)
  (h_mag_z1 : abs z₁ = 1)
  (h_mag_z2 : abs z₂ = 1)
  (h_sum : abs (z₁ + z₂ + z₃) = 1) :
  abs z₃ ≥ Real.sqrt 2 - 1 := by
sorry

end NUMINAMATH_CALUDE_min_value_z3_l448_44891


namespace NUMINAMATH_CALUDE_f_properties_l448_44816

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x^2 + a*x|

-- Define the property of being monotonically increasing on [0,1]
def monotone_increasing_on_unit_interval (g : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ y ∧ y ≤ 1 → g x ≤ g y

-- Define M(a) as the maximum value of f(x) on [0,1]
noncomputable def M (a : ℝ) : ℝ :=
  ⨆ (x : ℝ) (h : x ∈ Set.Icc 0 1), f a x

-- State the theorem
theorem f_properties (a : ℝ) :
  (monotone_increasing_on_unit_interval (f a) ↔ a ≤ -2 ∨ a ≥ 0) ∧
  (∃ (a_min : ℝ), ∀ (a : ℝ), M a_min ≤ M a ∧ M a_min = 3 - 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l448_44816


namespace NUMINAMATH_CALUDE_production_volume_proof_l448_44823

/-- Represents the production volume equation over three years -/
def production_equation (x : ℝ) : Prop :=
  200 + 200 * (1 + x) + 200 * (1 + x)^2 = 1400

/-- 
  Proves that the given equation correctly represents the total production volume
  over three years, given an initial production of 200 units and a constant
  percentage increase x for two consecutive years, resulting in a total of 1400 units.
-/
theorem production_volume_proof (x : ℝ) : production_equation x := by
  sorry

end NUMINAMATH_CALUDE_production_volume_proof_l448_44823


namespace NUMINAMATH_CALUDE_hotel_rooms_for_couples_l448_44810

theorem hotel_rooms_for_couples :
  let single_rooms : ℕ := 14
  let bubble_bath_per_bath : ℕ := 10
  let total_bubble_bath : ℕ := 400
  let baths_per_single_room : ℕ := 1
  let baths_per_couple_room : ℕ := 2
  ∃ couple_rooms : ℕ,
    couple_rooms = 13 ∧
    total_bubble_bath = bubble_bath_per_bath * (single_rooms * baths_per_single_room + couple_rooms * baths_per_couple_room) :=
by
  sorry

end NUMINAMATH_CALUDE_hotel_rooms_for_couples_l448_44810


namespace NUMINAMATH_CALUDE_tissue_cost_theorem_l448_44870

/-- Calculates the total cost of tissue boxes given the number of boxes, packs per box, tissues per pack, and cost per tissue. -/
def total_cost (boxes : ℕ) (packs_per_box : ℕ) (tissues_per_pack : ℕ) (cost_per_tissue : ℚ) : ℚ :=
  (boxes * packs_per_box * tissues_per_pack : ℚ) * cost_per_tissue

/-- Proves that the total cost of 10 boxes of tissues is $1000 given the specified conditions. -/
theorem tissue_cost_theorem :
  let boxes : ℕ := 10
  let packs_per_box : ℕ := 20
  let tissues_per_pack : ℕ := 100
  let cost_per_tissue : ℚ := 5 / 100
  total_cost boxes packs_per_box tissues_per_pack cost_per_tissue = 1000 := by
  sorry

#eval total_cost 10 20 100 (5 / 100)

end NUMINAMATH_CALUDE_tissue_cost_theorem_l448_44870


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_power_of_two_l448_44892

theorem negation_of_forall_positive_power_of_two (P : ℝ → Prop) :
  (¬ ∀ x > 0, 2^x > 0) ↔ (∃ x > 0, 2^x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_power_of_two_l448_44892


namespace NUMINAMATH_CALUDE_chess_group_size_l448_44899

-- Define the number of players in the chess group
def num_players : ℕ := 30

-- Define the total number of games played
def total_games : ℕ := 435

-- Theorem stating that the number of players is correct given the conditions
theorem chess_group_size :
  (num_players.choose 2 = total_games) ∧ (num_players > 0) := by
  sorry

end NUMINAMATH_CALUDE_chess_group_size_l448_44899


namespace NUMINAMATH_CALUDE_lottery_not_guaranteed_win_l448_44802

theorem lottery_not_guaranteed_win (total_tickets : ℕ) (winning_rate : ℝ) (bought_tickets : ℕ) : 
  total_tickets = 1000000 →
  winning_rate = 0.001 →
  bought_tickets = 1000 →
  ∃ p : ℝ, p > 0 ∧ p = (1 - winning_rate) ^ bought_tickets := by
  sorry

end NUMINAMATH_CALUDE_lottery_not_guaranteed_win_l448_44802


namespace NUMINAMATH_CALUDE_angle_with_complement_quarter_supplement_l448_44898

theorem angle_with_complement_quarter_supplement (x : ℝ) :
  (90 - x = (1 / 4) * (180 - x)) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_with_complement_quarter_supplement_l448_44898


namespace NUMINAMATH_CALUDE_coin_toss_probability_l448_44875

/-- The probability of getting a specific sequence of heads and tails in 10 coin tosses -/
theorem coin_toss_probability : 
  let n : ℕ := 10  -- number of tosses
  let p : ℚ := 1/2  -- probability of heads (or tails) in a single toss
  (p ^ n : ℚ) = 1/1024 := by sorry

end NUMINAMATH_CALUDE_coin_toss_probability_l448_44875


namespace NUMINAMATH_CALUDE_line_through_points_and_equal_intercepts_l448_44842

-- Define points
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (-2, 0)
def P : ℝ × ℝ := (-1, 3)

-- Define line equations
def line_eq_1 (x y : ℝ) : Prop := 2 * x - 5 * y + 4 = 0
def line_eq_2 (x y : ℝ) : Prop := x + y = 2

-- Define a function to check if a point lies on a line
def point_on_line (p : ℝ × ℝ) (line : ℝ → ℝ → Prop) : Prop :=
  line p.1 p.2

-- Define equal intercepts
def equal_intercepts (line : ℝ → ℝ → Prop) : Prop :=
  ∃ m : ℝ, line m 0 ∧ line 0 m

theorem line_through_points_and_equal_intercepts :
  (point_on_line A line_eq_1 ∧ point_on_line B line_eq_1) ∧
  (point_on_line P line_eq_2 ∧ equal_intercepts line_eq_2) :=
sorry

end NUMINAMATH_CALUDE_line_through_points_and_equal_intercepts_l448_44842


namespace NUMINAMATH_CALUDE_raymonds_dimes_proof_l448_44894

/-- The number of dimes Raymond has left after spending at the arcade -/
def raymonds_remaining_dimes : ℕ :=
  let initial_amount : ℕ := 250 -- $2.50 in cents
  let petes_spent : ℕ := 20 -- 4 nickels * 5 cents
  let total_spent : ℕ := 200
  let raymonds_spent : ℕ := total_spent - petes_spent
  let raymonds_remaining : ℕ := initial_amount - raymonds_spent
  raymonds_remaining / 10 -- divide by 10 cents per dime

theorem raymonds_dimes_proof :
  raymonds_remaining_dimes = 7 := by
  sorry

end NUMINAMATH_CALUDE_raymonds_dimes_proof_l448_44894


namespace NUMINAMATH_CALUDE_rectangular_shape_x_value_l448_44888

/-- A shape formed entirely of rectangles with all internal angles 90 degrees -/
structure RectangularShape where
  top_lengths : List ℝ
  bottom_lengths : List ℝ

/-- The sum of lengths in a list -/
def sum_lengths (lengths : List ℝ) : ℝ := lengths.sum

/-- The property that the sum of top lengths equals the sum of bottom lengths -/
def equal_total_length (shape : RectangularShape) : Prop :=
  sum_lengths shape.top_lengths = sum_lengths shape.bottom_lengths

theorem rectangular_shape_x_value (shape : RectangularShape) 
  (h1 : shape.top_lengths = [2, 3, 4, X])
  (h2 : shape.bottom_lengths = [1, 2, 4, 6])
  (h3 : equal_total_length shape) :
  X = 4 := by
  sorry

#check rectangular_shape_x_value

end NUMINAMATH_CALUDE_rectangular_shape_x_value_l448_44888


namespace NUMINAMATH_CALUDE_unique_common_roots_l448_44837

/-- Two cubic polynomials with two distinct common roots -/
def has_two_common_roots (p q : ℝ) : Prop :=
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧
    r₁^3 + p*r₁^2 + 8*r₁ + 10 = 0 ∧
    r₁^3 + q*r₁^2 + 17*r₁ + 15 = 0 ∧
    r₂^3 + p*r₂^2 + 8*r₂ + 10 = 0 ∧
    r₂^3 + q*r₂^2 + 17*r₂ + 15 = 0

/-- The unique solution for p and q -/
theorem unique_common_roots :
  ∃! (p q : ℝ), has_two_common_roots p q ∧ p = 19 ∧ q = 28 := by
  sorry

end NUMINAMATH_CALUDE_unique_common_roots_l448_44837


namespace NUMINAMATH_CALUDE_alcohol_concentration_proof_l448_44871

/-- Proves that adding 7.5 litres of pure alcohol to a 10 litre solution
    that is 30% alcohol results in a 60% alcohol solution -/
theorem alcohol_concentration_proof :
  let initial_volume : ℝ := 10
  let initial_concentration : ℝ := 0.30
  let added_alcohol : ℝ := 7.5
  let final_concentration : ℝ := 0.60
  let final_volume : ℝ := initial_volume + added_alcohol
  let final_alcohol : ℝ := initial_volume * initial_concentration + added_alcohol
  final_alcohol / final_volume = final_concentration :=
by sorry


end NUMINAMATH_CALUDE_alcohol_concentration_proof_l448_44871


namespace NUMINAMATH_CALUDE_addition_inequality_l448_44826

theorem addition_inequality (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_addition_inequality_l448_44826


namespace NUMINAMATH_CALUDE_max_value_fraction_l448_44872

theorem max_value_fraction (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -3) (hy : 1 ≤ y ∧ y ≤ 5) :
  (∀ x' y', -6 ≤ x' ∧ x' ≤ -3 → 1 ≤ y' ∧ y' ≤ 5 → (x' + y') / x' ≤ (x + y) / x) →
  (x + y) / x = 2/3 :=
sorry

end NUMINAMATH_CALUDE_max_value_fraction_l448_44872


namespace NUMINAMATH_CALUDE_slices_left_over_l448_44858

/-- The number of initial pizza slices -/
def initial_slices : ℕ := 34

/-- The number of slices eaten by Dean -/
def dean_slices : ℕ := 7

/-- The number of slices eaten by Frank -/
def frank_slices : ℕ := 3

/-- The number of slices eaten by Sammy -/
def sammy_slices : ℕ := 4

/-- The number of slices eaten by Nancy -/
def nancy_slices : ℕ := 3

/-- The number of slices eaten by Olivia -/
def olivia_slices : ℕ := 3

/-- The total number of slices eaten -/
def total_eaten : ℕ := dean_slices + frank_slices + sammy_slices + nancy_slices + olivia_slices

/-- Theorem: The number of pizza slices left over is 14 -/
theorem slices_left_over : initial_slices - total_eaten = 14 := by
  sorry

end NUMINAMATH_CALUDE_slices_left_over_l448_44858


namespace NUMINAMATH_CALUDE_function_property_l448_44835

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x, (x - 1) * (deriv f x) ≤ 0)
variable (h2 : ∀ x, f (x + 1) = f (-x + 1))

-- Define the theorem
theorem function_property (x₁ x₂ : ℝ) (h3 : |x₁ - 1| < |x₂ - 1|) :
  f (2 - x₁) ≥ f (2 - x₂) := by
  sorry

end NUMINAMATH_CALUDE_function_property_l448_44835


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l448_44853

def M : Set Int := {-1, 0, 1, 3, 5}
def N : Set Int := {-2, 1, 2, 3, 5}

theorem intersection_of_M_and_N : M ∩ N = {1, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l448_44853


namespace NUMINAMATH_CALUDE_f_composition_equals_three_l448_44851

noncomputable def f (x : ℂ) : ℂ :=
  if x.im = 0 then 1 + x else (1 - Complex.I) / Complex.abs Complex.I * x

theorem f_composition_equals_three :
  f (f (1 + Complex.I)) = 3 := by sorry

end NUMINAMATH_CALUDE_f_composition_equals_three_l448_44851


namespace NUMINAMATH_CALUDE_system_two_solutions_l448_44850

/-- The system of inequalities has exactly two solutions if and only if a = 7 -/
theorem system_two_solutions (a : ℝ) : 
  (∃! x y z : ℝ, x ≠ y ∧ 
    (abs z + abs (z - x) ≤ a - abs (x - 1)) ∧
    ((z - 4) * (z + 3) ≥ (4 - x) * (3 + x)) ∧
    (abs z + abs (z - y) ≤ a - abs (y - 1)) ∧
    ((z - 4) * (z + 3) ≥ (4 - y) * (3 + y)))
  ↔ a = 7 := by sorry

end NUMINAMATH_CALUDE_system_two_solutions_l448_44850


namespace NUMINAMATH_CALUDE_divisible_by_twenty_l448_44838

theorem divisible_by_twenty (n : ℕ) : ∃ k : ℤ, 9^(8*n+4) - 7^(8*n+4) = 20*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_twenty_l448_44838


namespace NUMINAMATH_CALUDE_age_difference_l448_44820

def sachin_age : ℕ := 49

theorem age_difference (rahul_age : ℕ) 
  (h1 : sachin_age < rahul_age)
  (h2 : sachin_age * 9 = rahul_age * 7) : 
  rahul_age - sachin_age = 14 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l448_44820


namespace NUMINAMATH_CALUDE_subset_increase_l448_44897

theorem subset_increase (m k : ℕ) (hm : m > 0) (hk : k ≥ 2) :
  let original_subsets := 2^m
  let new_subsets_one := 2^(m+1)
  let new_subsets_k := 2^(m+k)
  (new_subsets_one - original_subsets = 2^m) ∧
  (new_subsets_k - original_subsets = (2^k - 1) * 2^m) := by
  sorry

end NUMINAMATH_CALUDE_subset_increase_l448_44897


namespace NUMINAMATH_CALUDE_first_half_total_score_l448_44828

/-- Represents the scores of a team in a basketball game --/
structure TeamScores where
  q1 : ℝ
  q2 : ℝ
  q3 : ℝ
  q4 : ℝ

/-- The game conditions --/
def GameConditions (alpha : TeamScores) (beta : TeamScores) : Prop :=
  -- Tied after first quarter
  alpha.q1 = beta.q1
  -- Alpha's scores form a geometric sequence
  ∧ ∃ r : ℝ, r > 1 ∧ alpha.q2 = alpha.q1 * r ∧ alpha.q3 = alpha.q2 * r ∧ alpha.q4 = alpha.q3 * r
  -- Beta's scores form an arithmetic sequence
  ∧ ∃ d : ℝ, d > 0 ∧ beta.q2 = beta.q1 + d ∧ beta.q3 = beta.q2 + d ∧ beta.q4 = beta.q3 + d
  -- Alpha won by 3 points
  ∧ alpha.q1 + alpha.q2 + alpha.q3 + alpha.q4 = beta.q1 + beta.q2 + beta.q3 + beta.q4 + 3
  -- No team scored more than 120 points
  ∧ alpha.q1 + alpha.q2 + alpha.q3 + alpha.q4 ≤ 120
  ∧ beta.q1 + beta.q2 + beta.q3 + beta.q4 ≤ 120

/-- The theorem to be proved --/
theorem first_half_total_score (alpha : TeamScores) (beta : TeamScores) 
  (h : GameConditions alpha beta) : 
  alpha.q1 + alpha.q2 + beta.q1 + beta.q2 = 35.5 := by
  sorry

end NUMINAMATH_CALUDE_first_half_total_score_l448_44828


namespace NUMINAMATH_CALUDE_floor_range_l448_44834

theorem floor_range (x : ℝ) : 
  Int.floor x = -3 → -3 ≤ x ∧ x < -2 := by sorry

end NUMINAMATH_CALUDE_floor_range_l448_44834


namespace NUMINAMATH_CALUDE_tan_ratio_from_sin_sum_diff_l448_44830

theorem tan_ratio_from_sin_sum_diff (a b : ℝ) 
  (h1 : Real.sin (a + b) = 5/8) 
  (h2 : Real.sin (a - b) = 1/4) : 
  Real.tan a / Real.tan b = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_from_sin_sum_diff_l448_44830


namespace NUMINAMATH_CALUDE_smallest_possible_b_l448_44848

theorem smallest_possible_b (a b : ℝ) : 
  (2 < a ∧ a < b) →
  (2 + a ≤ b) →
  (1/b + 1/a ≤ 2) →
  b ≥ 2 + Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_possible_b_l448_44848


namespace NUMINAMATH_CALUDE_expected_boy_girl_adjacencies_l448_44808

/-- The expected number of boy-girl adjacencies in a random arrangement of boys and girls -/
theorem expected_boy_girl_adjacencies
  (num_boys : ℕ)
  (num_girls : ℕ)
  (total : ℕ)
  (h_total : total = num_boys + num_girls)
  (h_boys : num_boys = 7)
  (h_girls : num_girls = 13) :
  (total - 1 : ℚ) * (num_boys * num_girls : ℚ) / (total * (total - 1) / 2 : ℚ) = 91 / 10 :=
sorry

end NUMINAMATH_CALUDE_expected_boy_girl_adjacencies_l448_44808


namespace NUMINAMATH_CALUDE_line_intersection_x_axis_l448_44877

/-- A line passing through two points (8, 2) and (4, 6) intersects the x-axis at (10, 0) -/
theorem line_intersection_x_axis :
  let p1 : ℝ × ℝ := (8, 2)
  let p2 : ℝ × ℝ := (4, 6)
  let m : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
  let b : ℝ := p1.2 - m * p1.1
  let line (x : ℝ) : ℝ := m * x + b
  ∃ x : ℝ, line x = 0 ∧ x = 10
:= by sorry

end NUMINAMATH_CALUDE_line_intersection_x_axis_l448_44877


namespace NUMINAMATH_CALUDE_fish_tank_problem_l448_44854

/-- Represents the number of gallons needed for each of the smaller tanks -/
def smaller_tank_gallons (total_weekly_gallons : ℕ) : ℕ :=
  (total_weekly_gallons - 2 * 8) / 2

/-- Represents the difference in gallons between larger and smaller tanks -/
def gallon_difference (total_weekly_gallons : ℕ) : ℕ :=
  8 - smaller_tank_gallons total_weekly_gallons

theorem fish_tank_problem (total_gallons : ℕ) 
  (h1 : total_gallons = 112) 
  (h2 : total_gallons % 4 = 0) : 
  gallon_difference (total_gallons / 4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fish_tank_problem_l448_44854


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l448_44862

theorem complex_fraction_evaluation (c d : ℂ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : c^2 + c*d + d^2 = 0) : 
  (c^12 + d^12) / (c + d)^12 = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l448_44862


namespace NUMINAMATH_CALUDE_problem_solution_l448_44887

theorem problem_solution (y : ℝ) (h1 : y > 0) (h2 : y * y / 100 = 9) : y = 30 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l448_44887


namespace NUMINAMATH_CALUDE_min_orchard_space_l448_44879

/-- The space required for planting trees in an orchard. -/
def orchard_space (apple apricot plum : ℕ) : ℕ :=
  apple^2 + 5*apricot + plum^3

/-- The minimum space required for planting 10 trees, including at least one of each type. -/
theorem min_orchard_space :
  ∃ (apple apricot plum : ℕ),
    apple + apricot + plum = 10 ∧
    apple ≥ 1 ∧ apricot ≥ 1 ∧ plum ≥ 1 ∧
    ∀ (a b c : ℕ),
      a + b + c = 10 →
      a ≥ 1 → b ≥ 1 → c ≥ 1 →
      orchard_space apple apricot plum ≤ orchard_space a b c ∧
      orchard_space apple apricot plum = 37 :=
by sorry

end NUMINAMATH_CALUDE_min_orchard_space_l448_44879


namespace NUMINAMATH_CALUDE_ship_travel_ratio_l448_44806

/-- Proves that the ratio of the distance traveled on day 2 to day 1 is 3:1 given the ship's travel conditions --/
theorem ship_travel_ratio : 
  ∀ (day1_distance day2_distance day3_distance : ℝ),
  day1_distance = 100 →
  day3_distance = day2_distance + 110 →
  day1_distance + day2_distance + day3_distance = 810 →
  day2_distance / day1_distance = 3 := by
sorry


end NUMINAMATH_CALUDE_ship_travel_ratio_l448_44806


namespace NUMINAMATH_CALUDE_third_to_second_package_ratio_is_half_l448_44846

/-- Represents the delivery driver's work for a day -/
structure DeliveryDay where
  miles_first_package : ℕ
  miles_second_package : ℕ
  total_pay : ℕ
  pay_per_mile : ℕ

/-- Calculates the ratio of the distance for the third package to the second package -/
def third_to_second_package_ratio (day : DeliveryDay) : ℚ :=
  let total_miles := day.total_pay / day.pay_per_mile
  let miles_third_package := total_miles - day.miles_first_package - day.miles_second_package
  miles_third_package / day.miles_second_package

/-- Theorem stating the ratio of the third package distance to the second package distance -/
theorem third_to_second_package_ratio_is_half (day : DeliveryDay) 
    (h1 : day.miles_first_package = 10)
    (h2 : day.miles_second_package = 28)
    (h3 : day.total_pay = 104)
    (h4 : day.pay_per_mile = 2) :
    third_to_second_package_ratio day = 1/2 := by
  sorry

#eval third_to_second_package_ratio { 
  miles_first_package := 10, 
  miles_second_package := 28, 
  total_pay := 104, 
  pay_per_mile := 2 
}

end NUMINAMATH_CALUDE_third_to_second_package_ratio_is_half_l448_44846


namespace NUMINAMATH_CALUDE_fraction_meaningful_l448_44829

theorem fraction_meaningful (a : ℝ) : 
  (∃ x : ℝ, x = (a + 1) / (2 * a - 1)) ↔ a ≠ 1/2 :=
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l448_44829


namespace NUMINAMATH_CALUDE_mila_visible_area_l448_44812

/-- The area visible to Mila as she walks around a square -/
theorem mila_visible_area (side_length : ℝ) (visibility_radius : ℝ) : 
  side_length = 4 →
  visibility_radius = 1 →
  (side_length - 2 * visibility_radius)^2 + 
  4 * side_length * visibility_radius + 
  π * visibility_radius^2 = 28 + π := by
  sorry

end NUMINAMATH_CALUDE_mila_visible_area_l448_44812


namespace NUMINAMATH_CALUDE_quadratic_function_k_value_l448_44885

/-- Quadratic function g(x) = ax^2 + bx + c -/
def g (a b c : ℤ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_k_value (a b c : ℤ) :
  g a b c 2 = 0 →
  90 < g a b c 9 ∧ g a b c 9 < 100 →
  120 < g a b c 10 ∧ g a b c 10 < 130 →
  (∃ k : ℤ, 7000 * k < g a b c 150 ∧ g a b c 150 < 7000 * (k + 1)) →
  (∃ k : ℤ, 7000 * k < g a b c 150 ∧ g a b c 150 < 7000 * (k + 1) ∧ k = 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_k_value_l448_44885


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l448_44874

/-- A positive geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

/-- The theorem statement -/
theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_2016 : a 2016 = a 2015 + 2 * a 2014)
  (h_mn : ∃ m n : ℕ, a m * a n = 16 * (a 1)^2) :
  ∃ m n : ℕ, (4 / m + 1 / n : ℝ) ≥ 3/2 ∧
    ∀ k l : ℕ, (4 / k + 1 / l : ℝ) ≥ 3/2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l448_44874


namespace NUMINAMATH_CALUDE_remainder_proof_l448_44833

theorem remainder_proof (a b : ℕ) (h : a > b) : 
  220070 % (a + b) = 220070 - (a + b) * (2 * (a - b)) :=
by
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l448_44833


namespace NUMINAMATH_CALUDE_sin_240_degrees_l448_44813

theorem sin_240_degrees : Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_240_degrees_l448_44813


namespace NUMINAMATH_CALUDE_linear_equation_solution_l448_44804

theorem linear_equation_solution (a b : ℝ) : 
  (2 * a + (-1) * b = -1) → (1 + 2 * a - b = 0) := by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l448_44804


namespace NUMINAMATH_CALUDE_second_number_problem_l448_44805

theorem second_number_problem (a b c : ℚ) : 
  a + b + c = 264 →
  a = 2 * b →
  c = (1 / 3) * a →
  b = 72 := by sorry

end NUMINAMATH_CALUDE_second_number_problem_l448_44805


namespace NUMINAMATH_CALUDE_polygon_sides_from_interior_angle_l448_44873

theorem polygon_sides_from_interior_angle (interior_angle : ℝ) (n : ℕ) :
  interior_angle = 108 →
  (n : ℝ) * (180 - interior_angle) = 360 →
  n = 5 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_interior_angle_l448_44873


namespace NUMINAMATH_CALUDE_calculator_squaring_l448_44866

theorem calculator_squaring (initial : ℕ) (target : ℕ) : 
  (initial = 3 ∧ target = 2000) → 
  (∃ n : ℕ, initial^(2^n) > target ∧ ∀ m : ℕ, m < n → initial^(2^m) ≤ target) → 
  (∃ n : ℕ, n = 3 ∧ initial^(2^n) > target ∧ ∀ m : ℕ, m < n → initial^(2^m) ≤ target) :=
by sorry

end NUMINAMATH_CALUDE_calculator_squaring_l448_44866


namespace NUMINAMATH_CALUDE_f_composition_value_l448_44807

def f (x : ℝ) : ℝ := 4 * x^3 - 6 * x + 2

theorem f_composition_value : f (f 2) = 42462 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l448_44807


namespace NUMINAMATH_CALUDE_quadratic_equation_one_l448_44831

theorem quadratic_equation_one (x : ℝ) : (x - 2)^2 = 9 ↔ x = 5 ∨ x = -1 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_one_l448_44831


namespace NUMINAMATH_CALUDE_total_bottles_bought_l448_44868

-- Define the variables
def bottles_per_day : ℕ := 9
def days_lasted : ℕ := 17

-- Define the theorem
theorem total_bottles_bought : 
  bottles_per_day * days_lasted = 153 := by
  sorry

end NUMINAMATH_CALUDE_total_bottles_bought_l448_44868


namespace NUMINAMATH_CALUDE_emily_coloring_books_l448_44863

/-- 
Given Emily's initial number of coloring books, the number she gave away,
and her current total, prove that she bought 14 coloring books.
-/
theorem emily_coloring_books 
  (initial : ℕ) 
  (given_away : ℕ) 
  (current_total : ℕ) 
  (h1 : initial = 7)
  (h2 : given_away = 2)
  (h3 : current_total = 19) :
  current_total - (initial - given_away) = 14 := by
  sorry

end NUMINAMATH_CALUDE_emily_coloring_books_l448_44863


namespace NUMINAMATH_CALUDE_shaded_probability_three_fourths_l448_44821

-- Define the right-angled triangle
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  right_angle : leg1^2 + leg2^2 = hypotenuse^2

-- Define the game board
structure GameBoard where
  triangle : RightTriangle
  total_regions : ℕ
  shaded_regions : ℕ
  regions_by_altitudes : total_regions = 4
  shaded_count : shaded_regions = 3

-- Define the probability function
def probability_shaded (board : GameBoard) : ℚ :=
  board.shaded_regions / board.total_regions

-- Theorem statement
theorem shaded_probability_three_fourths 
  (board : GameBoard) 
  (h1 : board.triangle.leg1 = 6) 
  (h2 : board.triangle.leg2 = 8) : 
  probability_shaded board = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_shaded_probability_three_fourths_l448_44821


namespace NUMINAMATH_CALUDE_students_above_90_l448_44840

/-- Represents a normal distribution of test scores -/
structure ScoreDistribution where
  mean : ℝ
  variance : ℝ
  is_normal : Bool

/-- Represents the class and score information -/
structure ClassScores where
  total_students : ℕ
  distribution : ScoreDistribution
  between_mean_and_plus_10 : ℕ

/-- Theorem stating the number of students scoring above 90 -/
theorem students_above_90 (c : ClassScores) 
  (h1 : c.total_students = 48)
  (h2 : c.distribution.mean = 80)
  (h3 : c.distribution.is_normal = true)
  (h4 : c.between_mean_and_plus_10 = 16) :
  c.total_students / 2 - c.between_mean_and_plus_10 = 8 := by
  sorry


end NUMINAMATH_CALUDE_students_above_90_l448_44840


namespace NUMINAMATH_CALUDE_next_perfect_square_l448_44864

theorem next_perfect_square (x : ℕ) (h : ∃ k : ℕ, x = k^2) :
  ∃ n : ℕ, n > x ∧ (∃ m : ℕ, n = m^2) ∧ 
  ∀ y : ℕ, y > x → (∃ l : ℕ, y = l^2) → y ≥ n :=
by
  sorry

end NUMINAMATH_CALUDE_next_perfect_square_l448_44864


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l448_44817

/-- Theorem about a specific triangle ABC --/
theorem triangle_abc_properties :
  ∀ (a b c : ℝ) (A B C : ℝ),
  A = π / 3 →  -- 60° in radians
  b = 1 →
  c = 4 →
  (a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A) →  -- Cosine rule
  (a = Real.sqrt 13 ∧ 
   (1 / 2) * b * c * Real.sin A = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l448_44817


namespace NUMINAMATH_CALUDE_prob_green_ball_is_five_ninths_l448_44878

structure Container where
  red_balls : ℕ
  green_balls : ℕ

def total_balls (c : Container) : ℕ := c.red_balls + c.green_balls

def prob_green (c : Container) : ℚ :=
  c.green_balls / (total_balls c)

def containers : List Container := [
  ⟨8, 4⟩,  -- Container I
  ⟨2, 4⟩,  -- Container II
  ⟨2, 4⟩   -- Container III
]

theorem prob_green_ball_is_five_ninths :
  (containers.map prob_green).sum / containers.length = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_green_ball_is_five_ninths_l448_44878


namespace NUMINAMATH_CALUDE_square_sum_implies_product_zero_l448_44880

theorem square_sum_implies_product_zero (n : ℝ) :
  (n - 2022)^2 + (2023 - n)^2 = 1 → (2022 - n) * (n - 2023) = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_implies_product_zero_l448_44880


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l448_44876

theorem quadratic_equations_solutions :
  let eq1 : ℝ → Prop := λ x => 4 * x^2 - 9 = 0
  let eq2 : ℝ → Prop := λ x => 2 * x^2 - 3 * x - 5 = 0
  let solutions1 : Set ℝ := {3/2, -3/2}
  let solutions2 : Set ℝ := {1, 5/2}
  (∀ x : ℝ, eq1 x ↔ x ∈ solutions1) ∧
  (∀ x : ℝ, eq2 x ↔ x ∈ solutions2) :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_equations_solutions_l448_44876


namespace NUMINAMATH_CALUDE_quadratic_integer_root_existence_l448_44869

/-- A quadratic polynomial with integer coefficients -/
structure QuadraticPolynomial where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Check if a quadratic polynomial has an integer root -/
def has_integer_root (p : QuadraticPolynomial) : Prop :=
  ∃ x : ℤ, p.a * x * x + p.b * x + p.c = 0

/-- Calculate the cost of changing from one polynomial to another -/
def change_cost (p q : QuadraticPolynomial) : ℕ :=
  (Int.natAbs (p.a - q.a)) + (Int.natAbs (p.b - q.b)) + (Int.natAbs (p.c - q.c))

/-- The main theorem -/
theorem quadratic_integer_root_existence (p : QuadraticPolynomial) 
    (h : p.a + p.b + p.c = 2000) :
    ∃ q : QuadraticPolynomial, has_integer_root q ∧ change_cost p q ≤ 1022 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_integer_root_existence_l448_44869


namespace NUMINAMATH_CALUDE_ab_pos_necessary_not_sufficient_l448_44857

theorem ab_pos_necessary_not_sufficient (a b : ℝ) :
  (∃ a b : ℝ, (b / a + a / b > 2) ∧ (a * b > 0)) ∧
  (∃ a b : ℝ, (a * b > 0) ∧ ¬(b / a + a / b > 2)) ∧
  (∀ a b : ℝ, (b / a + a / b > 2) → (a * b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_ab_pos_necessary_not_sufficient_l448_44857


namespace NUMINAMATH_CALUDE_square_side_length_l448_44895

theorem square_side_length (perimeter : ℝ) (h : perimeter = 17.8) :
  perimeter / 4 = 4.45 := by sorry

end NUMINAMATH_CALUDE_square_side_length_l448_44895


namespace NUMINAMATH_CALUDE_problem_solution_l448_44882

theorem problem_solution (a b : ℕ) (ha : a = 3) (hb : b = 2) : 
  (a^(b+1))^a + (b^(a+1))^b = 19939 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l448_44882


namespace NUMINAMATH_CALUDE_original_triangle_area_l448_44886

/-- Given a triangle whose dimensions are quadrupled to form a new triangle with area 144 square feet,
    prove that the area of the original triangle is 9 square feet. -/
theorem original_triangle_area (original_area new_area : ℝ) : 
  (∀ side, new_area = (4 * side)^2 / 2 * (original_area / (side^2 / 2))) → 
  new_area = 144 → 
  original_area = 9 := by
  sorry

end NUMINAMATH_CALUDE_original_triangle_area_l448_44886


namespace NUMINAMATH_CALUDE_geometric_series_relation_l448_44884

/-- Given real numbers c and d satisfying an infinite geometric series condition,
    prove that another related infinite geometric series equals 5/7. -/
theorem geometric_series_relation (c d : ℝ) 
    (h : (c/d) / (1 - 1/d) = 5) : 
    (c/(c+2*d)) / (1 - 1/(c+2*d)) = 5/7 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_relation_l448_44884


namespace NUMINAMATH_CALUDE_max_x_minus_y_l448_44844

theorem max_x_minus_y (x y : ℝ) (h : 2 * (x^2 + y^2) = x + y + 2*x*y) : 
  ∃ (M : ℝ), M = 2 ∧ ∀ (a b : ℝ), 2 * (a^2 + b^2) = a + b + 2*a*b → a - b ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l448_44844
