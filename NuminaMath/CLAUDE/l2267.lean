import Mathlib

namespace NUMINAMATH_CALUDE_maurice_previous_rides_l2267_226786

/-- Represents the horseback riding scenario of Maurice and Matt -/
structure RidingScenario where
  maurice_prev_rides : ℕ
  maurice_prev_horses : ℕ
  matt_total_horses : ℕ
  maurice_visit_rides : ℕ
  matt_rides_with_maurice : ℕ
  matt_solo_rides : ℕ
  matt_solo_horses : ℕ

/-- The specific scenario described in the problem -/
def problem_scenario : RidingScenario :=
  { maurice_prev_rides := 0,  -- to be determined
    maurice_prev_horses := 2,
    matt_total_horses := 4,
    maurice_visit_rides := 8,
    matt_rides_with_maurice := 8,
    matt_solo_rides := 16,
    matt_solo_horses := 2 }

/-- Theorem stating the number of Maurice's previous rides -/
theorem maurice_previous_rides (s : RidingScenario) :
  s.maurice_prev_horses = 2 ∧
  s.matt_total_horses = 4 ∧
  s.maurice_visit_rides = 8 ∧
  s.matt_rides_with_maurice = 8 ∧
  s.matt_solo_rides = 16 ∧
  s.matt_solo_horses = 2 ∧
  s.maurice_visit_rides = s.maurice_prev_rides ∧
  (s.matt_rides_with_maurice + s.matt_solo_rides) = 3 * s.maurice_prev_rides →
  s.maurice_prev_rides = 8 := by
  sorry

#check maurice_previous_rides problem_scenario

end NUMINAMATH_CALUDE_maurice_previous_rides_l2267_226786


namespace NUMINAMATH_CALUDE_fifteen_percent_of_600_is_90_l2267_226713

theorem fifteen_percent_of_600_is_90 : (15 / 100) * 600 = 90 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_percent_of_600_is_90_l2267_226713


namespace NUMINAMATH_CALUDE_infinite_geometric_series_ratio_l2267_226757

/-- Proves that for an infinite geometric series with first term 400 and sum 2500, the common ratio is 21/25 -/
theorem infinite_geometric_series_ratio : ∃ (r : ℝ), 
  let a : ℝ := 400
  let S : ℝ := 2500
  r > 0 ∧ r < 1 ∧ S = a / (1 - r) ∧ r = 21 / 25 := by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_ratio_l2267_226757


namespace NUMINAMATH_CALUDE_cost_of_48_doughnuts_l2267_226781

/-- The cost of buying a specified number of doughnuts -/
def doughnutCost (n : ℕ) : ℚ :=
  1 + 6 * ((n - 1) / 12)

/-- Theorem stating the cost of 48 doughnuts -/
theorem cost_of_48_doughnuts : doughnutCost 48 = 25 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_48_doughnuts_l2267_226781


namespace NUMINAMATH_CALUDE_subset_iff_a_in_range_l2267_226793

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def B (a : ℝ) : Set ℝ := {x | 2^(1-x) + a ≤ 0 ∧ x^2 - 2*(a+7)*x + 5 ≤ 0}

-- State the theorem
theorem subset_iff_a_in_range :
  ∀ a : ℝ, A ⊆ B a ↔ -4 ≤ a ∧ a ≤ -1 := by sorry

end NUMINAMATH_CALUDE_subset_iff_a_in_range_l2267_226793


namespace NUMINAMATH_CALUDE_l_shaped_area_l2267_226791

/-- The area of an L-shaped region formed by subtracting three non-overlapping squares
    from a larger square -/
theorem l_shaped_area (side_length : ℝ) (small_square1 : ℝ) (small_square2 : ℝ) (small_square3 : ℝ)
    (h1 : side_length = 6)
    (h2 : small_square1 = 2)
    (h3 : small_square2 = 4)
    (h4 : small_square3 = 2)
    (h5 : small_square1 + small_square2 + small_square3 ≤ side_length) :
    side_length^2 - (small_square1^2 + small_square2^2 + small_square3^2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_l_shaped_area_l2267_226791


namespace NUMINAMATH_CALUDE_cakes_sold_l2267_226773

theorem cakes_sold (made bought left : ℕ) 
  (h1 : made = 173)
  (h2 : bought = 103)
  (h3 : left = 190) :
  made + bought - left = 86 := by
sorry

end NUMINAMATH_CALUDE_cakes_sold_l2267_226773


namespace NUMINAMATH_CALUDE_managers_wage_l2267_226756

/-- Proves that the manager's hourly wage is $7.50 given the wage relationships between manager, chef, and dishwasher -/
theorem managers_wage (manager chef dishwasher : ℝ) 
  (h1 : chef = dishwasher * 1.2)
  (h2 : dishwasher = manager / 2)
  (h3 : chef = manager - 3) :
  manager = 7.5 := by
sorry

end NUMINAMATH_CALUDE_managers_wage_l2267_226756


namespace NUMINAMATH_CALUDE_dairy_water_mixture_l2267_226719

theorem dairy_water_mixture (original_price selling_price : ℝ) 
  (h1 : selling_price = original_price * 1.25) : 
  (selling_price - original_price) / selling_price = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_dairy_water_mixture_l2267_226719


namespace NUMINAMATH_CALUDE_games_next_month_l2267_226783

/-- Calculates the number of games Jason plans to attend next month -/
theorem games_next_month 
  (games_this_month : ℕ) 
  (games_last_month : ℕ) 
  (total_games : ℕ) 
  (h1 : games_this_month = 11)
  (h2 : games_last_month = 17)
  (h3 : total_games = 44) :
  total_games - (games_this_month + games_last_month) = 16 := by
sorry

end NUMINAMATH_CALUDE_games_next_month_l2267_226783


namespace NUMINAMATH_CALUDE_solution_replacement_concentration_l2267_226790

/-- Calculates the new concentration of a solution after partial replacement -/
def new_concentration (initial_conc : ℝ) (replacement_conc : ℝ) (replaced_fraction : ℝ) : ℝ :=
  (1 - replaced_fraction) * initial_conc + replaced_fraction * replacement_conc

/-- Theorem stating that replacing 7/9 of a 70% solution with a 25% solution results in a 35% solution -/
theorem solution_replacement_concentration :
  new_concentration 0.7 0.25 (7/9) = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_solution_replacement_concentration_l2267_226790


namespace NUMINAMATH_CALUDE_negation_of_proposition_cubic_inequality_negation_l2267_226778

theorem negation_of_proposition (p : ℝ → Prop) : 
  (¬∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬(p x)) :=
by sorry

theorem cubic_inequality_negation : 
  (¬∀ x : ℝ, x^3 + 2 < 0) ↔ (∃ x : ℝ, x^3 + 2 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_cubic_inequality_negation_l2267_226778


namespace NUMINAMATH_CALUDE_enclosed_area_is_four_l2267_226746

-- Define the functions
def f (x : ℝ) : ℝ := 4 * x
def g (x : ℝ) : ℝ := x^3

-- Define the region
def region := {x : ℝ | 0 ≤ x ∧ x ≤ 2 ∧ g x ≤ f x}

-- State the theorem
theorem enclosed_area_is_four : 
  ∫ x in region, (f x - g x) = 4 := by sorry

end NUMINAMATH_CALUDE_enclosed_area_is_four_l2267_226746


namespace NUMINAMATH_CALUDE_cone_cylinder_volume_relation_l2267_226712

/-- The volume of a cone with the same radius and height as a cylinder with volume 150π cm³ is 50π cm³ -/
theorem cone_cylinder_volume_relation (r h : ℝ) (h1 : r > 0) (h2 : h > 0) : 
  π * r^2 * h = 150 * π → (1/3) * π * r^2 * h = 50 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_cylinder_volume_relation_l2267_226712


namespace NUMINAMATH_CALUDE_unique_solution_condition_l2267_226752

theorem unique_solution_condition (c d : ℝ) :
  (∃! x : ℝ, ∃ y : ℝ, 4 * x - 7 + c = d * x + 2 * y + 4) ↔ d ≠ 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l2267_226752


namespace NUMINAMATH_CALUDE_gcd_problem_l2267_226703

theorem gcd_problem (n : ℕ) : 
  80 ≤ n ∧ n ≤ 100 → Nat.gcd 36 n = 12 → n = 84 ∨ n = 96 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l2267_226703


namespace NUMINAMATH_CALUDE_tan_equality_implies_45_l2267_226774

theorem tan_equality_implies_45 (n : ℤ) :
  -90 < n ∧ n < 90 ∧ Real.tan (n * π / 180) = Real.tan (225 * π / 180) → n = 45 := by
  sorry

end NUMINAMATH_CALUDE_tan_equality_implies_45_l2267_226774


namespace NUMINAMATH_CALUDE_correct_calculation_l2267_226721

theorem correct_calculation (x : ℤ) (h : x + 65 = 125) : x + 95 = 155 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2267_226721


namespace NUMINAMATH_CALUDE_perpendicular_implies_parallel_perpendicular_to_plane_implies_parallel_l2267_226769

-- Define the types for lines and planes in space
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- Theorem 1: If m ⟂ α and m ⟂ β, then α ∥ β
theorem perpendicular_implies_parallel 
  (m : Line) (α β : Plane) 
  (h1 : perpendicular m α) (h2 : perpendicular m β) : 
  parallel α β :=
sorry

-- Theorem 2: If m ⟂ β and n ⟂ β, then m ∥ n
theorem perpendicular_to_plane_implies_parallel 
  (m n : Line) (β : Plane) 
  (h1 : perpendicular m β) (h2 : perpendicular n β) : 
  parallel_lines m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_implies_parallel_perpendicular_to_plane_implies_parallel_l2267_226769


namespace NUMINAMATH_CALUDE_segment_length_bound_polygon_perimeter_bound_l2267_226759

-- Define a segment in 2D plane
structure Segment where
  length : ℝ
  projection1 : ℝ
  projection2 : ℝ

-- Define a polygon in 2D plane
structure Polygon where
  perimeter : ℝ
  totalProjection1 : ℝ
  totalProjection2 : ℝ

-- Theorem for segment
theorem segment_length_bound (s : Segment) : 
  s.length ≥ (s.projection1 + s.projection2) / Real.sqrt 2 := by sorry

-- Theorem for polygon
theorem polygon_perimeter_bound (p : Polygon) :
  p.perimeter ≥ Real.sqrt 2 * (p.totalProjection1 + p.totalProjection2) := by sorry

end NUMINAMATH_CALUDE_segment_length_bound_polygon_perimeter_bound_l2267_226759


namespace NUMINAMATH_CALUDE_particular_number_exists_l2267_226739

theorem particular_number_exists : ∃! x : ℝ, 2 * ((x / 23) - 67) = 102 := by
  sorry

end NUMINAMATH_CALUDE_particular_number_exists_l2267_226739


namespace NUMINAMATH_CALUDE_mango_tree_columns_count_l2267_226731

/-- The number of columns of mango trees in a garden with given dimensions -/
def mango_tree_columns (garden_length : ℕ) (tree_distance : ℕ) (boundary_distance : ℕ) : ℕ :=
  let available_length := garden_length - 2 * boundary_distance
  let spaces := available_length / tree_distance
  spaces + 1

/-- Theorem stating that the number of mango tree columns is 12 given the specified conditions -/
theorem mango_tree_columns_count :
  mango_tree_columns 32 2 5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_mango_tree_columns_count_l2267_226731


namespace NUMINAMATH_CALUDE_prime_arithmetic_progression_l2267_226750

theorem prime_arithmetic_progression (p₁ p₂ p₃ : ℕ) (d : ℕ) : 
  Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ 
  p₁ > 3 ∧ p₂ > 3 ∧ p₃ > 3 ∧
  p₂ = p₁ + d ∧ p₃ = p₂ + d → 
  d % 6 = 0 := by
  sorry

#check prime_arithmetic_progression

end NUMINAMATH_CALUDE_prime_arithmetic_progression_l2267_226750


namespace NUMINAMATH_CALUDE_loan_payment_difference_l2267_226736

/-- Calculates the monthly payment for a loan -/
def monthly_payment (loan_amount : ℚ) (months : ℕ) : ℚ :=
  loan_amount / months

/-- Represents the loan details -/
structure LoanDetails where
  amount : ℚ
  short_term_months : ℕ
  long_term_months : ℕ

theorem loan_payment_difference (loan : LoanDetails) 
  (h1 : loan.amount = 6000)
  (h2 : loan.short_term_months = 24)
  (h3 : loan.long_term_months = 60) :
  monthly_payment loan.amount loan.short_term_months - 
  monthly_payment loan.amount loan.long_term_months = 150 := by
  sorry


end NUMINAMATH_CALUDE_loan_payment_difference_l2267_226736


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2267_226734

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 1}
def N : Set ℝ := {x | ∃ y : ℝ, y = Real.sqrt (3 - x^2)}

-- Define the closed interval [-1, √3]
def interval : Set ℝ := {x | -1 ≤ x ∧ x ≤ Real.sqrt 3}

-- Theorem statement
theorem intersection_of_M_and_N : M ∩ N = interval := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2267_226734


namespace NUMINAMATH_CALUDE_max_knights_and_courtiers_l2267_226705

/-- Represents the number of people at the king's table -/
def kings_table : ℕ := 7

/-- Represents the minimum number of courtiers -/
def min_courtiers : ℕ := 12

/-- Represents the maximum number of courtiers -/
def max_courtiers : ℕ := 18

/-- Represents the minimum number of knights -/
def min_knights : ℕ := 10

/-- Represents the maximum number of knights -/
def max_knights : ℕ := 20

/-- Represents the rule that the lunch of a knight plus the lunch of a courtier equals the lunch of the king -/
def lunch_rule (courtiers knights : ℕ) : Prop :=
  (1 : ℚ) / courtiers + (1 : ℚ) / knights = (1 : ℚ) / kings_table

/-- The main theorem stating the maximum number of knights and courtiers -/
theorem max_knights_and_courtiers :
  ∃ (k c : ℕ), 
    min_courtiers ≤ c ∧ c ≤ max_courtiers ∧
    min_knights ≤ k ∧ k ≤ max_knights ∧
    lunch_rule c k ∧
    (∀ (k' c' : ℕ), 
      min_courtiers ≤ c' ∧ c' ≤ max_courtiers ∧
      min_knights ≤ k' ∧ k' ≤ max_knights ∧
      lunch_rule c' k' →
      k' ≤ k) ∧
    k = 14 ∧ c = 14 :=
  sorry

end NUMINAMATH_CALUDE_max_knights_and_courtiers_l2267_226705


namespace NUMINAMATH_CALUDE_total_stripes_is_22_l2267_226749

/-- The number of stripes on each of Olga's tennis shoes -/
def olga_stripes : ℕ := 3

/-- The number of stripes on each of Rick's tennis shoes -/
def rick_stripes : ℕ := olga_stripes - 1

/-- The number of stripes on each of Hortense's tennis shoes -/
def hortense_stripes : ℕ := olga_stripes * 2

/-- The number of shoes each person has -/
def shoes_per_person : ℕ := 2

/-- The total number of stripes on all their shoes combined -/
def total_stripes : ℕ := shoes_per_person * (olga_stripes + rick_stripes + hortense_stripes)

theorem total_stripes_is_22 : total_stripes = 22 := by
  sorry

end NUMINAMATH_CALUDE_total_stripes_is_22_l2267_226749


namespace NUMINAMATH_CALUDE_z_equals_2_minus_12i_z_is_pure_imaginary_l2267_226758

/-- Complex number z as a function of real number m -/
def z (m : ℝ) : ℂ := Complex.mk (m^2 + 5*m + 6) (m^2 - 2*m - 15)

/-- Theorem for the first condition -/
theorem z_equals_2_minus_12i (m : ℝ) : z m = Complex.mk 2 (-12) ↔ m = -1 := by sorry

/-- Theorem for the second condition -/
theorem z_is_pure_imaginary (m : ℝ) : (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = -2 := by sorry

end NUMINAMATH_CALUDE_z_equals_2_minus_12i_z_is_pure_imaginary_l2267_226758


namespace NUMINAMATH_CALUDE_expression_equals_six_l2267_226754

theorem expression_equals_six : 
  Real.sqrt 16 - 2 * Real.tan (45 * π / 180) + |(-3)| + (π - 2022) ^ (0 : ℕ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_six_l2267_226754


namespace NUMINAMATH_CALUDE_abs_inequality_equivalence_l2267_226745

theorem abs_inequality_equivalence (x : ℝ) : 
  |2*x - 1| - |x - 2| < 0 ↔ -1 < x ∧ x < 1 := by
sorry

end NUMINAMATH_CALUDE_abs_inequality_equivalence_l2267_226745


namespace NUMINAMATH_CALUDE_max_x_on_circle_max_x_achieved_l2267_226799

theorem max_x_on_circle (x y : ℝ) (h : x^2 + y^2 = 18*x + 20*y) :
  x ≤ 9 + Real.sqrt 181 :=
by sorry

theorem max_x_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ (x y : ℝ), x^2 + y^2 = 18*x + 20*y ∧ x > 9 + Real.sqrt 181 - ε :=
by sorry

end NUMINAMATH_CALUDE_max_x_on_circle_max_x_achieved_l2267_226799


namespace NUMINAMATH_CALUDE_ellipse_k_range_l2267_226715

/-- An ellipse with equation x^2 + ky^2 = 2 and foci on the y-axis -/
structure Ellipse where
  k : ℝ
  eq : ∀ x y : ℝ, x^2 + k * y^2 = 2
  foci_on_y : True  -- This is a placeholder for the foci condition

/-- The range of k for a valid ellipse with foci on the y-axis -/
def valid_k_range (e : Ellipse) : Prop :=
  0 < e.k ∧ e.k < 1

/-- Theorem stating that for any ellipse with the given properties, k must be in (0, 1) -/
theorem ellipse_k_range (e : Ellipse) : valid_k_range e := by
  sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l2267_226715


namespace NUMINAMATH_CALUDE_percentage_square_divide_l2267_226751

theorem percentage_square_divide (x : ℝ) :
  ((208 / 100 * 1265) ^ 2) / 12 = 576857.87 := by
  sorry

end NUMINAMATH_CALUDE_percentage_square_divide_l2267_226751


namespace NUMINAMATH_CALUDE_tax_savings_proof_l2267_226709

/-- Represents the tax brackets and rates -/
structure TaxBracket :=
  (lower : ℕ) (upper : ℕ) (rate : ℚ)

/-- Calculates the tax for a given income and tax brackets -/
def calculateTax (income : ℕ) (brackets : List TaxBracket) : ℚ :=
  sorry

/-- Represents the tax system -/
structure TaxSystem :=
  (brackets : List TaxBracket)
  (standardDeduction : ℕ)
  (childCredit : ℕ)

/-- Calculates the total tax liability for a given income and tax system -/
def calculateTaxLiability (income : ℕ) (children : ℕ) (system : TaxSystem) : ℚ :=
  sorry

/-- The current tax system -/
def currentSystem : TaxSystem :=
  { brackets := [
      ⟨0, 15000, 15/100⟩,
      ⟨15001, 45000, 42/100⟩,
      ⟨45001, 1000000, 50/100⟩  -- Using a large number for the upper bound of the highest bracket
    ],
    standardDeduction := 3000,
    childCredit := 1000
  }

/-- The proposed tax system -/
def proposedSystem : TaxSystem :=
  { brackets := [
      ⟨0, 15000, 12/100⟩,
      ⟨15001, 45000, 28/100⟩,
      ⟨45001, 1000000, 50/100⟩  -- Using a large number for the upper bound of the highest bracket
    ],
    standardDeduction := 3000,
    childCredit := 1000
  }

theorem tax_savings_proof (income : ℕ) (h : income = 34500) :
  calculateTaxLiability income 2 currentSystem - calculateTaxLiability income 2 proposedSystem = 2760 := by
  sorry

end NUMINAMATH_CALUDE_tax_savings_proof_l2267_226709


namespace NUMINAMATH_CALUDE_min_fish_in_aquarium_l2267_226788

/-- Represents the number of fish of each known color in the aquarium -/
structure AquariumFish where
  yellow : ℕ
  blue : ℕ
  green : ℕ

/-- The conditions of the aquarium as described in the problem -/
def aquarium_conditions (fish : AquariumFish) : Prop :=
  fish.yellow = 12 ∧
  fish.blue = fish.yellow / 2 ∧
  fish.green = fish.yellow * 2

/-- The theorem stating the minimum number of fish in the aquarium -/
theorem min_fish_in_aquarium (fish : AquariumFish) 
  (h : aquarium_conditions fish) : 
  fish.yellow + fish.blue + fish.green = 42 := by
  sorry

end NUMINAMATH_CALUDE_min_fish_in_aquarium_l2267_226788


namespace NUMINAMATH_CALUDE_smallest_multiple_l2267_226700

theorem smallest_multiple : ∃ (a : ℕ), 
  (∀ (n : ℕ), n > 0 ∧ 6 ∣ n ∧ 15 ∣ n ∧ n > 40 → a ≤ n) ∧
  6 ∣ a ∧ 15 ∣ a ∧ a > 40 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_l2267_226700


namespace NUMINAMATH_CALUDE_percentage_men_not_speaking_french_or_spanish_l2267_226717

theorem percentage_men_not_speaking_french_or_spanish :
  let total_men_percentage : ℚ := 100
  let french_speaking_men_percentage : ℚ := 55
  let spanish_speaking_men_percentage : ℚ := 35
  let other_languages_men_percentage : ℚ := 10
  (total_men_percentage = french_speaking_men_percentage + spanish_speaking_men_percentage + other_languages_men_percentage) →
  (other_languages_men_percentage = 10) :=
by sorry

end NUMINAMATH_CALUDE_percentage_men_not_speaking_french_or_spanish_l2267_226717


namespace NUMINAMATH_CALUDE_max_prime_value_l2267_226755

theorem max_prime_value (a b : ℕ) (p : ℕ) (h_prime : Nat.Prime p) 
  (h_eq : p = (b / 4) * Real.sqrt ((2 * a - b) / (2 * a + b))) : 
  p ≤ 5 ∧ ∃ (a' b' : ℕ), (5 : ℕ) = (b' / 4) * Real.sqrt ((2 * a' - b') / (2 * a' + b')) := by
  sorry

end NUMINAMATH_CALUDE_max_prime_value_l2267_226755


namespace NUMINAMATH_CALUDE_min_value_cyclic_sum_l2267_226725

theorem min_value_cyclic_sum (a b c k : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hk : 0 < k) :
  (k * a / b) + (k * b / c) + (k * c / a) ≥ 3 * k ∧
  ((k * a / b) + (k * b / c) + (k * c / a) = 3 * k ↔ a = b ∧ b = c) :=
sorry

end NUMINAMATH_CALUDE_min_value_cyclic_sum_l2267_226725


namespace NUMINAMATH_CALUDE_assignment_conditions_l2267_226772

/-- The number of ways to assign four students to three classes -/
def assignStudents : ℕ :=
  Nat.choose 4 2 * (3 * 2 * 1) - (3 * 2 * 1)

/-- Conditions of the problem -/
theorem assignment_conditions :
  (∀ (assignment : Fin 4 → Fin 3), 
    (∀ c : Fin 3, ∃ s : Fin 4, assignment s = c) ∧ 
    (assignment 0 ≠ assignment 1)) →
  (assignStudents = 30) :=
sorry

end NUMINAMATH_CALUDE_assignment_conditions_l2267_226772


namespace NUMINAMATH_CALUDE_final_retail_price_l2267_226780

/-- Calculates the final retail price of a machine given wholesale price, markup, discount, and desired profit percentage. -/
theorem final_retail_price
  (wholesale_price : ℝ)
  (markup_percentage : ℝ)
  (discount_percentage : ℝ)
  (desired_profit_percentage : ℝ)
  (h1 : wholesale_price = 90)
  (h2 : markup_percentage = 1)
  (h3 : discount_percentage = 0.2)
  (h4 : desired_profit_percentage = 0.6)
  : wholesale_price * (1 + markup_percentage) * (1 - discount_percentage) = 144 :=
by sorry

end NUMINAMATH_CALUDE_final_retail_price_l2267_226780


namespace NUMINAMATH_CALUDE_book_length_l2267_226718

theorem book_length (area : ℝ) (width : ℝ) (h1 : area = 50) (h2 : width = 10) :
  area / width = 5 := by
  sorry

end NUMINAMATH_CALUDE_book_length_l2267_226718


namespace NUMINAMATH_CALUDE_log_equality_implies_base_l2267_226711

theorem log_equality_implies_base (y : ℝ) (h : y > 0) :
  (Real.log 8 / Real.log y = Real.log 5 / Real.log 125) → y = 512 := by
  sorry

end NUMINAMATH_CALUDE_log_equality_implies_base_l2267_226711


namespace NUMINAMATH_CALUDE_marble_distribution_l2267_226730

theorem marble_distribution (n : ℕ) (hn : n = 480) :
  (Finset.filter (fun m => m > 1 ∧ m < n) (Finset.range (n + 1))).card = 22 :=
by sorry

end NUMINAMATH_CALUDE_marble_distribution_l2267_226730


namespace NUMINAMATH_CALUDE_symmetry_point_l2267_226706

/-- Given two points A and B in a 2D plane, they are symmetric with respect to the origin
    if the sum of their coordinates is (0, 0) -/
def symmetric_wrt_origin (A B : ℝ × ℝ) : Prop :=
  A.1 + B.1 = 0 ∧ A.2 + B.2 = 0

theorem symmetry_point :
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (2, -3)
  symmetric_wrt_origin A B → B = (2, -3) := by
sorry

end NUMINAMATH_CALUDE_symmetry_point_l2267_226706


namespace NUMINAMATH_CALUDE_video_game_sales_earnings_l2267_226797

def total_games : ℕ := 15
def non_working_games : ℕ := 9
def price_per_game : ℕ := 5

theorem video_game_sales_earnings : 
  (total_games - non_working_games) * price_per_game = 30 := by
  sorry

end NUMINAMATH_CALUDE_video_game_sales_earnings_l2267_226797


namespace NUMINAMATH_CALUDE_peters_parrots_l2267_226765

/-- Calculates the number of parrots Peter has based on the given conditions -/
theorem peters_parrots :
  let parakeet_consumption : ℕ := 2 -- grams per day
  let parrot_consumption : ℕ := 14 -- grams per day
  let finch_consumption : ℕ := parakeet_consumption / 2 -- grams per day
  let num_parakeets : ℕ := 3
  let num_finches : ℕ := 4
  let total_birdseed : ℕ := 266 -- grams for a week
  let days_in_week : ℕ := 7

  let parakeet_weekly_consumption : ℕ := num_parakeets * parakeet_consumption * days_in_week
  let finch_weekly_consumption : ℕ := num_finches * finch_consumption * days_in_week
  let remaining_birdseed : ℕ := total_birdseed - parakeet_weekly_consumption - finch_weekly_consumption
  let parrot_weekly_consumption : ℕ := parrot_consumption * days_in_week

  remaining_birdseed / parrot_weekly_consumption = 2 :=
by sorry

end NUMINAMATH_CALUDE_peters_parrots_l2267_226765


namespace NUMINAMATH_CALUDE_parabola_properties_l2267_226747

-- Define the parabola
def parabola (a b c x : ℝ) := a * x^2 + b * x + c

-- State the theorem
theorem parabola_properties (a b c : ℝ) 
  (h1 : parabola a b c 2 = 0) :
  (b = -2*a → parabola a b c 0 = 0) ∧ 
  (c ≠ 4*a → (b^2 - 4*a*c > 0)) ∧
  (∀ x1 x2, x1 > x2 ∧ x2 > -1 ∧ parabola a b c x1 > parabola a b c x2 → 8*a + c ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l2267_226747


namespace NUMINAMATH_CALUDE_acute_triangle_from_sides_l2267_226792

theorem acute_triangle_from_sides (a b c : ℝ) (ha : a = 5) (hb : b = 6) (hc : c = 7) :
  a + b > c ∧ b + c > a ∧ c + a > b ∧ a^2 + b^2 > c^2 := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_from_sides_l2267_226792


namespace NUMINAMATH_CALUDE_arrangement_sum_l2267_226760

theorem arrangement_sum (n : ℕ+) 
  (h1 : n + 3 ≤ 2 * n) 
  (h2 : n + 1 ≤ 4) : 
  Nat.descFactorial (2 * n) (n + 3) + Nat.descFactorial 4 (n + 1) = 744 :=
sorry

end NUMINAMATH_CALUDE_arrangement_sum_l2267_226760


namespace NUMINAMATH_CALUDE_regression_line_not_necessarily_through_points_l2267_226741

/-- Sample data point -/
structure DataPoint where
  x : ℝ
  y : ℝ

/-- Linear regression model -/
structure LinearRegression where
  a : ℝ  -- intercept
  b : ℝ  -- slope

/-- Predicts y value for a given x using the linear regression model -/
def predict (model : LinearRegression) (x : ℝ) : ℝ :=
  model.a + model.b * x

/-- Checks if a point lies on the regression line -/
def pointOnLine (model : LinearRegression) (point : DataPoint) : Prop :=
  predict model point.x = point.y

theorem regression_line_not_necessarily_through_points 
  (model : LinearRegression) (data : List DataPoint) : 
  ¬ (∀ point ∈ data, pointOnLine model point) := by
  sorry

#check regression_line_not_necessarily_through_points

end NUMINAMATH_CALUDE_regression_line_not_necessarily_through_points_l2267_226741


namespace NUMINAMATH_CALUDE_reading_time_difference_is_360_l2267_226733

/-- Calculates the difference in reading time between two people in minutes -/
def reading_time_difference (xanthia_speed molly_speed book_pages : ℕ) : ℕ :=
  let xanthia_time := book_pages / xanthia_speed
  let molly_time := book_pages / molly_speed
  (molly_time - xanthia_time) * 60

/-- The difference in reading time between Molly and Xanthia is 360 minutes -/
theorem reading_time_difference_is_360 :
  reading_time_difference 120 40 360 = 360 :=
by sorry

end NUMINAMATH_CALUDE_reading_time_difference_is_360_l2267_226733


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l2267_226794

theorem decimal_to_fraction (x : ℚ) (h : x = 336/100) : 
  ∃ (a b : ℕ), x = a / b ∧ a = 84 ∧ b = 25 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l2267_226794


namespace NUMINAMATH_CALUDE_probability_six_odd_in_eight_rolls_l2267_226779

theorem probability_six_odd_in_eight_rolls (n : ℕ) (p : ℚ) : 
  n = 8 →                   -- number of rolls
  p = 1/2 →                 -- probability of rolling an odd number
  (n.choose 6 : ℚ) * p^6 * (1 - p)^(n - 6) = 7/64 := by
sorry

end NUMINAMATH_CALUDE_probability_six_odd_in_eight_rolls_l2267_226779


namespace NUMINAMATH_CALUDE_total_net_increase_l2267_226776

/-- Represents a time period with birth and death rates -/
structure TimePeriod where
  birthRate : Nat
  deathRate : Nat

/-- Calculates the net population increase for a given time period -/
def netIncrease (tp : TimePeriod) : Nat :=
  (tp.birthRate - tp.deathRate) * 10800

/-- The four time periods in a day -/
def dayPeriods : List TimePeriod := [
  { birthRate := 4, deathRate := 3 },
  { birthRate := 8, deathRate := 3 },
  { birthRate := 10, deathRate := 4 },
  { birthRate := 6, deathRate := 2 }
]

/-- Theorem: The total net population increase in one day is 172,800 -/
theorem total_net_increase : 
  (dayPeriods.map netIncrease).sum = 172800 := by
  sorry

end NUMINAMATH_CALUDE_total_net_increase_l2267_226776


namespace NUMINAMATH_CALUDE_inverse_sum_reciprocal_l2267_226775

theorem inverse_sum_reciprocal (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x⁻¹ + y⁻¹ + z⁻¹)⁻¹ = (x * y * z) / (x * z + y * z + x * y) := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_reciprocal_l2267_226775


namespace NUMINAMATH_CALUDE_average_monthly_balance_l2267_226729

def monthly_balances : List ℝ := [120, 240, 180, 180, 160, 200]

theorem average_monthly_balance :
  (monthly_balances.sum / monthly_balances.length : ℝ) = 180 := by
  sorry

end NUMINAMATH_CALUDE_average_monthly_balance_l2267_226729


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2267_226744

theorem inequality_solution_set (x : ℝ) :
  (((x + 5) / 2) - 2 < (3 * x + 2) / 2) ↔ (x > -1/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2267_226744


namespace NUMINAMATH_CALUDE_turtle_difference_l2267_226762

/-- Given the following conditions about turtle ownership:
  1. Trey has 9 times as many turtles as Kris
  2. Kris has 1/3 as many turtles as Kristen
  3. Layla has twice as many turtles as Trey
  4. Tim has half as many turtles as Kristen
  5. Kristen has 18 turtles

  Prove that Trey has 45 more turtles than Tim. -/
theorem turtle_difference (kristen tim trey kris layla : ℕ) : 
  kristen = 18 →
  kris = kristen / 3 →
  trey = 9 * kris →
  layla = 2 * trey →
  tim = kristen / 2 →
  trey - tim = 45 := by
sorry

end NUMINAMATH_CALUDE_turtle_difference_l2267_226762


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2267_226737

open Set

def U : Set Nat := {0, 1, 2, 3, 4}
def M : Set Nat := {0, 1, 2}
def N : Set Nat := {2, 3}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2267_226737


namespace NUMINAMATH_CALUDE_no_solution_sqrt_equation_l2267_226742

theorem no_solution_sqrt_equation : ¬ ∃ x : ℝ, Real.sqrt (3 * x - 2) + Real.sqrt (2 * x - 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_sqrt_equation_l2267_226742


namespace NUMINAMATH_CALUDE_perpendicular_lines_slope_product_l2267_226716

/-- Given two lines in the plane, if they are perpendicular, then the product of their slopes is -1 -/
theorem perpendicular_lines_slope_product (a : ℝ) : 
  (∀ x y : ℝ, 2*x + y + 1 = 0 → x + a*y + 3 = 0 → (2 : ℝ) * (1/a) = -1) → 
  a = -2 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_slope_product_l2267_226716


namespace NUMINAMATH_CALUDE_triangle_larger_segment_is_82_5_l2267_226701

/-- A triangle with sides a, b, c, where c is the longest side --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  h_c_longest : c ≥ a ∧ c ≥ b

/-- The angle opposite to the longest side of the triangle --/
def Triangle.angle_opposite_longest (t : Triangle) : ℝ := sorry

/-- The altitude to the longest side of the triangle --/
def Triangle.altitude_to_longest (t : Triangle) : ℝ := sorry

/-- The larger segment cut off by the altitude on the longest side --/
def Triangle.larger_segment (t : Triangle) : ℝ := sorry

theorem triangle_larger_segment_is_82_5 (t : Triangle) 
  (h_sides : t.a = 40 ∧ t.b = 90 ∧ t.c = 100) 
  (h_angle : t.angle_opposite_longest = Real.pi / 3) : 
  t.larger_segment = 82.5 := by sorry

end NUMINAMATH_CALUDE_triangle_larger_segment_is_82_5_l2267_226701


namespace NUMINAMATH_CALUDE_brad_balloons_l2267_226782

/-- Given that Brad has 8 red balloons and 9 green balloons, prove that he has 17 balloons in total. -/
theorem brad_balloons (red_balloons green_balloons : ℕ) 
  (h1 : red_balloons = 8) 
  (h2 : green_balloons = 9) : 
  red_balloons + green_balloons = 17 := by
  sorry

end NUMINAMATH_CALUDE_brad_balloons_l2267_226782


namespace NUMINAMATH_CALUDE_probability_theorem_l2267_226784

def total_cups : ℕ := 8
def white_cups : ℕ := 3
def red_cups : ℕ := 3
def black_cups : ℕ := 2
def selected_cups : ℕ := 5

def probability_specific_order : ℚ := (white_cups * (white_cups - 1) * red_cups * (red_cups - 1) * black_cups) / 
  (total_cups * (total_cups - 1) * (total_cups - 2) * (total_cups - 3) * (total_cups - 4))

def number_of_arrangements : ℕ := Nat.factorial selected_cups / 
  (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1)

theorem probability_theorem : 
  (↑number_of_arrangements * probability_specific_order : ℚ) = 9 / 14 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l2267_226784


namespace NUMINAMATH_CALUDE_max_value_theorem_l2267_226740

theorem max_value_theorem (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) :
  ∀ x y z, 0 ≤ x ∧ x ≤ 2 → 0 ≤ y ∧ y ≤ 2 → 0 ≤ z ∧ z ≤ 2 →
    Real.sqrt (a^2 * b^2 * c^2) + Real.sqrt ((2 - a) * (2 - b) * (2 - c)) ≤
    Real.sqrt (x^2 * y^2 * z^2) + Real.sqrt ((2 - x) * (2 - y) * (2 - z)) →
    Real.sqrt (a^2 * b^2 * c^2) + Real.sqrt ((2 - a) * (2 - b) * (2 - c)) ≤ 8 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2267_226740


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2267_226796

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2267_226796


namespace NUMINAMATH_CALUDE_afternoon_sales_problem_l2267_226735

/-- Calculates the number of cookies sold in the afternoon given the initial count,
    morning sales, lunch sales, and remaining cookies. -/
def afternoon_sales (initial : ℕ) (morning_dozens : ℕ) (lunch : ℕ) (remaining : ℕ) : ℕ :=
  initial - (morning_dozens * 12 + lunch) - remaining

theorem afternoon_sales_problem :
  afternoon_sales 120 3 57 11 = 16 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_sales_problem_l2267_226735


namespace NUMINAMATH_CALUDE_triple_composition_even_l2267_226726

-- Define an even function
def EvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (-x)

-- State the theorem
theorem triple_composition_even (g : ℝ → ℝ) (h : EvenFunction g) :
  EvenFunction (fun x ↦ g (g (g x))) :=
by
  sorry

end NUMINAMATH_CALUDE_triple_composition_even_l2267_226726


namespace NUMINAMATH_CALUDE_current_trees_proof_current_trees_is_25_l2267_226771

/-- The number of popular trees currently in the park -/
def current_trees : ℕ := sorry

/-- The number of popular trees to be planted today -/
def trees_to_plant : ℕ := 73

/-- The total number of popular trees after planting -/
def total_trees : ℕ := 98

/-- Theorem stating that the current number of trees plus the trees to be planted equals the total trees after planting -/
theorem current_trees_proof : current_trees + trees_to_plant = total_trees := by sorry

/-- Theorem proving that the number of current trees is 25 -/
theorem current_trees_is_25 : current_trees = 25 := by sorry

end NUMINAMATH_CALUDE_current_trees_proof_current_trees_is_25_l2267_226771


namespace NUMINAMATH_CALUDE_inscribed_circle_theorem_l2267_226738

theorem inscribed_circle_theorem (r : ℝ) (a b c : ℝ) :
  r > 0 ∧ a > 0 ∧ b > 0 ∧ c > 0 →
  r = 4 →
  a + b = 14 →
  (∃ (s : ℝ), s = (a + b + c) / 2 ∧ s * r = Real.sqrt (s * (s - a) * (s - b) * (s - c))) →
  (c = 13 ∧ b = 15) ∨ (c = 15 ∧ b = 13) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_theorem_l2267_226738


namespace NUMINAMATH_CALUDE_min_distance_to_plane_l2267_226714

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  normal : Point3D
  d : ℝ

/-- The distance between a point and a plane -/
def distPointToPlane (p : Point3D) (plane : Plane) : ℝ :=
  sorry

/-- The distance between two points -/
def distBetweenPoints (p1 p2 : Point3D) : ℝ :=
  sorry

theorem min_distance_to_plane (α β γ : Plane) (A P : Point3D) :
  -- Planes are mutually perpendicular
  (α.normal.x * β.normal.x + α.normal.y * β.normal.y + α.normal.z * β.normal.z = 0) →
  (β.normal.x * γ.normal.x + β.normal.y * γ.normal.y + β.normal.z * γ.normal.z = 0) →
  (γ.normal.x * α.normal.x + γ.normal.y * α.normal.y + γ.normal.z * α.normal.z = 0) →
  -- A is on plane α
  (distPointToPlane A α = 0) →
  -- Distance from A to plane β is 3
  (distPointToPlane A β = 3) →
  -- Distance from A to plane γ is 3
  (distPointToPlane A γ = 3) →
  -- P is on plane α
  (distPointToPlane P α = 0) →
  -- Distance from P to plane β is twice the distance from P to point A
  (distPointToPlane P β = 2 * distBetweenPoints P A) →
  -- The minimum distance from points on the trajectory of P to plane γ is 3 - √3
  (∃ (P' : Point3D), distPointToPlane P' α = 0 ∧
    distPointToPlane P' β = 2 * distBetweenPoints P' A ∧
    distPointToPlane P' γ = 3 - Real.sqrt 3 ∧
    ∀ (P'' : Point3D), distPointToPlane P'' α = 0 →
      distPointToPlane P'' β = 2 * distBetweenPoints P'' A →
      distPointToPlane P'' γ ≥ 3 - Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_CALUDE_min_distance_to_plane_l2267_226714


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2267_226767

theorem cubic_root_sum (a b : ℝ) : 
  (Complex.I * Real.sqrt 2 + 2 : ℂ) ^ 3 + a * (Complex.I * Real.sqrt 2 + 2) + b = 0 → 
  a + b = 14 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2267_226767


namespace NUMINAMATH_CALUDE_exactly_nine_heads_probability_l2267_226724

/-- The probability of getting heads when flipping the biased coin -/
def p : ℚ := 3/4

/-- The number of coin flips -/
def n : ℕ := 12

/-- The number of heads we want to get -/
def k : ℕ := 9

/-- The probability of getting exactly k heads in n flips of a coin with probability p of heads -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem exactly_nine_heads_probability :
  binomial_probability n k p = 4330260/16777216 := by
  sorry

end NUMINAMATH_CALUDE_exactly_nine_heads_probability_l2267_226724


namespace NUMINAMATH_CALUDE_republican_votes_for_candidate_a_l2267_226722

theorem republican_votes_for_candidate_a (total_voters : ℝ) 
  (h1 : total_voters > 0) 
  (democrat_percent : ℝ) 
  (h2 : democrat_percent = 0.60)
  (republican_percent : ℝ) 
  (h3 : republican_percent = 1 - democrat_percent)
  (democrat_votes_for_a_percent : ℝ) 
  (h4 : democrat_votes_for_a_percent = 0.65)
  (total_votes_for_a_percent : ℝ) 
  (h5 : total_votes_for_a_percent = 0.47) : 
  (total_votes_for_a_percent * total_voters - democrat_votes_for_a_percent * democrat_percent * total_voters) / 
  (republican_percent * total_voters) = 0.20 := by
sorry

end NUMINAMATH_CALUDE_republican_votes_for_candidate_a_l2267_226722


namespace NUMINAMATH_CALUDE_pen_price_l2267_226766

theorem pen_price (total_pens : ℕ) (total_cost : ℚ) (regular_price : ℚ) : 
  total_pens = 20 ∧ total_cost = 30 ∧ 
  (regular_price * (total_pens / 2) + (regular_price / 2) * (total_pens / 2) = total_cost) →
  regular_price = 2 := by
sorry

end NUMINAMATH_CALUDE_pen_price_l2267_226766


namespace NUMINAMATH_CALUDE_sequence_2021st_term_l2267_226704

/-- The sequence function that gives the n-th term of the sequence -/
def sequenceFunction (n : ℕ) : ℕ := sorry

/-- The sum of the first n natural numbers -/
def triangularNumber (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The property that the n-th positive integer appears n times in the sequence -/
axiom sequence_property (n : ℕ) : 
  ∀ k, triangularNumber (n - 1) < k ∧ k ≤ triangularNumber n → sequenceFunction k = n

/-- The theorem stating that the 2021st term of the sequence is 64 -/
theorem sequence_2021st_term : sequenceFunction 2021 = 64 := by sorry

end NUMINAMATH_CALUDE_sequence_2021st_term_l2267_226704


namespace NUMINAMATH_CALUDE_sin_alpha_value_l2267_226795

theorem sin_alpha_value (α β : Real) 
  (h1 : (0 : Real) < α ∧ α < Real.pi / 2)
  (h2 : -Real.pi / 2 < β ∧ β < 0)
  (h3 : Real.sin β = -5 / 13)
  (h4 : Real.sqrt ((Real.cos α - Real.cos β)^2 + (Real.sin α - Real.sin β)^2) = 2 * Real.sqrt 5 / 5) :
  Real.sin α = 33 / 65 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l2267_226795


namespace NUMINAMATH_CALUDE_triangle_side_length_l2267_226743

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C,
    if a = 3, C = 120°, and the area S = 15√3/4, then c = 7. -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) (S : ℝ) : 
  a = 3 →
  C = 2 * π / 3 →  -- 120° in radians
  S = 15 * Real.sqrt 3 / 4 →
  S = 1/2 * a * b * Real.sin C →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  c = 7 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2267_226743


namespace NUMINAMATH_CALUDE_cost_reduction_percentage_l2267_226768

/-- Proves the percentage reduction in cost price given specific conditions --/
theorem cost_reduction_percentage
  (original_cost : ℝ)
  (original_profit_rate : ℝ)
  (price_reduction : ℝ)
  (new_profit_rate : ℝ)
  (h1 : original_cost = 40)
  (h2 : original_profit_rate = 0.25)
  (h3 : price_reduction = 8.40)
  (h4 : new_profit_rate = 0.30)
  : ∃ (reduction_rate : ℝ),
    reduction_rate = 0.20 ∧
    (1 + new_profit_rate) * (original_cost * (1 - reduction_rate)) =
    (1 + original_profit_rate) * original_cost - price_reduction :=
by sorry

end NUMINAMATH_CALUDE_cost_reduction_percentage_l2267_226768


namespace NUMINAMATH_CALUDE_reflection_distance_l2267_226777

/-- Given a point A with coordinates (1, -3), prove that the distance between A
    and its reflection A' over the y-axis is 2. -/
theorem reflection_distance : 
  let A : ℝ × ℝ := (1, -3)
  let A' : ℝ × ℝ := (-1, -3)  -- Reflection of A over y-axis
  ‖A - A'‖ = 2 := by
  sorry

end NUMINAMATH_CALUDE_reflection_distance_l2267_226777


namespace NUMINAMATH_CALUDE_log_equation_relationships_l2267_226761

/-- Given real numbers a and b satisfying log_(1/2)(a) = log_(1/3)(b), 
    exactly 2 out of 5 given relationships cannot hold true. -/
theorem log_equation_relationships (a b : ℝ) 
  (h : Real.log a / Real.log (1/2) = Real.log b / Real.log (1/3)) : 
  ∃! (s : Finset (Fin 5)), s.card = 2 ∧ 
  (∀ i ∈ s, match i with
    | 0 => ¬(a > b ∧ b > 1)
    | 1 => ¬(0 < b ∧ b < a ∧ a < 1)
    | 2 => ¬(b > a ∧ a > 1)
    | 3 => ¬(0 < a ∧ a < b ∧ b < 1)
    | 4 => ¬(a = b)
  ) ∧
  (∀ i ∉ s, match i with
    | 0 => (a > b ∧ b > 1)
    | 1 => (0 < b ∧ b < a ∧ a < 1)
    | 2 => (b > a ∧ a > 1)
    | 3 => (0 < a ∧ a < b ∧ b < 1)
    | 4 => (a = b)
  ) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_relationships_l2267_226761


namespace NUMINAMATH_CALUDE_triangle_side_product_l2267_226732

noncomputable def Triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_side_product (a b c : ℝ) :
  Triangle a b c →
  (a + b)^2 - c^2 = 4 →
  Real.cos (60 * π / 180) = 1/2 →
  a * b = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_product_l2267_226732


namespace NUMINAMATH_CALUDE_trapezoid_bc_length_l2267_226707

-- Define the trapezoid and its properties
structure Trapezoid where
  area : ℝ
  altitude : ℝ
  ab_length : ℝ
  cd_length : ℝ
  ad_cd_angle : ℝ

-- Define the theorem
theorem trapezoid_bc_length (t : Trapezoid) 
  (h1 : t.area = 200)
  (h2 : t.altitude = 10)
  (h3 : t.ab_length = 15)
  (h4 : t.cd_length = 25)
  (h5 : t.ad_cd_angle = π/4)
  : ∃ (bc_length : ℝ), bc_length = (200 - (25 * Real.sqrt 5 + 25 * Real.sqrt 21)) / 10 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_bc_length_l2267_226707


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_1000_l2267_226789

def is_mersenne_prime (n : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ n = 2^p - 1 ∧ Nat.Prime n

theorem largest_mersenne_prime_under_1000 :
  ∀ n : ℕ, is_mersenne_prime n → n < 1000 → n ≤ 127 :=
by sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_1000_l2267_226789


namespace NUMINAMATH_CALUDE_angle_property_equivalence_l2267_226798

theorem angle_property_equivalence (θ : Real) (h : 0 ≤ θ ∧ θ ≤ 2 * Real.pi) :
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → x^2 * Real.cos θ - x * (1 - x) + 2 * (1 - x)^2 * Real.sin θ > 0) ↔
  (π / 24 < θ ∧ θ < 11 * π / 24) ∨ (25 * π / 24 < θ ∧ θ < 47 * π / 24) :=
by sorry

end NUMINAMATH_CALUDE_angle_property_equivalence_l2267_226798


namespace NUMINAMATH_CALUDE_circle_line_problem_l2267_226728

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 15 = 0

-- Define the line l
def l (x y k : ℝ) : Prop := y = k*x - 2

-- Define tangency condition
def is_tangent (k : ℝ) : Prop := ∃ x y : ℝ, C x y ∧ l x y k

-- Define the condition for a point on l to be within distance 2 from the center of C
def point_within_distance (k : ℝ) : Prop := 
  ∃ x y : ℝ, l x y k ∧ (x - 4)^2 + y^2 ≤ 4

theorem circle_line_problem (k : ℝ) :
  (is_tangent k → k = (8 + Real.sqrt 19) / 15 ∨ k = (8 - Real.sqrt 19) / 15) ∧
  (point_within_distance k → 0 ≤ k ∧ k ≤ 4/3) :=
sorry

end NUMINAMATH_CALUDE_circle_line_problem_l2267_226728


namespace NUMINAMATH_CALUDE_circle_equation_l2267_226727

theorem circle_equation (x y : ℝ) : 
  (∃ (R : ℝ), (x - 3)^2 + (y - 1)^2 = R^2) ∧ 
  (0 - 3)^2 + (0 - 1)^2 = 10 →
  (x - 3)^2 + (y - 1)^2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l2267_226727


namespace NUMINAMATH_CALUDE_wood_cost_is_1_50_l2267_226763

/-- The cost of producing birdhouses and selling them to Danny -/
structure BirdhouseProduction where
  wood_per_birdhouse : ℕ
  profit_per_birdhouse : ℚ
  price_for_two : ℚ

/-- Calculate the cost of each piece of wood -/
def wood_cost (p : BirdhouseProduction) : ℚ :=
  (p.price_for_two - 2 * p.profit_per_birdhouse) / (2 * p.wood_per_birdhouse)

/-- Theorem: Given the conditions, the cost of each piece of wood is $1.50 -/
theorem wood_cost_is_1_50 (p : BirdhouseProduction) 
  (h1 : p.wood_per_birdhouse = 7)
  (h2 : p.profit_per_birdhouse = 11/2)
  (h3 : p.price_for_two = 32) : 
  wood_cost p = 3/2 := by
  sorry

#eval wood_cost ⟨7, 11/2, 32⟩

end NUMINAMATH_CALUDE_wood_cost_is_1_50_l2267_226763


namespace NUMINAMATH_CALUDE_expression_evaluation_l2267_226710

theorem expression_evaluation (x y : ℝ) (hx : x = 2) (hy : y = -1) :
  2*x*y - 1/2*(4*x*y - 8*x^2*y^2) + 2*(3*x*y - 5*x^2*y^2) = -36 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2267_226710


namespace NUMINAMATH_CALUDE_train_crossing_time_l2267_226708

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  train_speed_kmh = 48 →
  crossing_time = 9 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) :=
by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l2267_226708


namespace NUMINAMATH_CALUDE_student_ratio_proof_l2267_226764

/-- Proves that the ratio of elementary school students to other students is 8/9 -/
theorem student_ratio_proof 
  (m n : ℕ) -- number of elementary and other students
  (a b : ℝ) -- average heights of elementary and other students
  (α β : ℝ) -- given constants
  (h1 : a = α * b) -- condition 1
  (h2 : α = 3/4) -- given value of α
  (h3 : a = β * ((a * m + b * n) / (m + n))) -- condition 2
  (h4 : β = 19/20) -- given value of β
  : m / n = 8/9 := by
  sorry


end NUMINAMATH_CALUDE_student_ratio_proof_l2267_226764


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_squared_l2267_226770

theorem equilateral_triangle_side_length_squared 
  (α β γ : ℂ) (s t : ℂ) :
  (∀ z, z^3 + s*z + t = 0 ↔ z = α ∨ z = β ∨ z = γ) →
  Complex.abs α ^ 2 + Complex.abs β ^ 2 + Complex.abs γ ^ 2 = 360 →
  ∃ l : ℝ, l > 0 ∧ 
    Complex.abs (α - β) = l ∧
    Complex.abs (β - γ) = l ∧
    Complex.abs (γ - α) = l →
  Complex.abs (α - β) ^ 2 = 360 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_squared_l2267_226770


namespace NUMINAMATH_CALUDE_racket_sales_total_l2267_226787

/-- The total amount for which rackets were sold, given the average price per pair and the number of pairs sold. -/
theorem racket_sales_total (avg_price : ℝ) (num_pairs : ℕ) (h1 : avg_price = 9.8) (h2 : num_pairs = 70) :
  avg_price * (num_pairs : ℝ) = 686 := by
  sorry

end NUMINAMATH_CALUDE_racket_sales_total_l2267_226787


namespace NUMINAMATH_CALUDE_vessel_base_length_l2267_226702

/-- Given a cube immersed in a rectangular vessel, calculate the length of the vessel's base. -/
theorem vessel_base_length (cube_edge : ℝ) (vessel_width : ℝ) (water_rise : ℝ) : 
  cube_edge = 15 →
  vessel_width = 14 →
  water_rise = 12.053571428571429 →
  (cube_edge ^ 3) / (vessel_width * water_rise) = 20 := by
  sorry

end NUMINAMATH_CALUDE_vessel_base_length_l2267_226702


namespace NUMINAMATH_CALUDE_shortest_chord_through_A_equals_4_l2267_226753

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

-- Define point A
def point_A : ℝ × ℝ := (1, 1)

-- Define the function to calculate the shortest chord length
noncomputable def shortest_chord_length (c : (ℝ → ℝ → Prop)) (p : ℝ × ℝ) : ℝ :=
  sorry -- Implementation details are omitted

-- Theorem statement
theorem shortest_chord_through_A_equals_4 :
  shortest_chord_length circle_M point_A = 4 := by sorry

end NUMINAMATH_CALUDE_shortest_chord_through_A_equals_4_l2267_226753


namespace NUMINAMATH_CALUDE_calculator_key_presses_l2267_226720

def f (x : ℕ) : ℕ := x^2 - 3

def iterate_f (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n+1 => f (iterate_f n x)

theorem calculator_key_presses :
  iterate_f 2 4 ≤ 2000 ∧ iterate_f 3 4 > 2000 := by
  sorry

end NUMINAMATH_CALUDE_calculator_key_presses_l2267_226720


namespace NUMINAMATH_CALUDE_square_perimeter_l2267_226723

theorem square_perimeter (s : ℝ) (h1 : s > 0) : 
  (∃ (w h : ℝ), w > 0 ∧ h > 0 ∧ w * h = s^2 / 4 ∧ 2*(w + h) = 40) → 
  4 * s = 64 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_l2267_226723


namespace NUMINAMATH_CALUDE_kevin_born_1984_l2267_226785

/-- The year of the first AMC 8 competition -/
def first_amc8_year : ℕ := 1988

/-- The year Kevin took the AMC 8 -/
def kevins_amc8_year : ℕ := first_amc8_year + 9

/-- Kevin's age when he took the AMC 8 -/
def kevins_age : ℕ := 13

/-- Kevin's birth year -/
def kevins_birth_year : ℕ := kevins_amc8_year - kevins_age

theorem kevin_born_1984 : kevins_birth_year = 1984 := by
  sorry

end NUMINAMATH_CALUDE_kevin_born_1984_l2267_226785


namespace NUMINAMATH_CALUDE_trip_cost_per_person_l2267_226748

/-- Given a group of 11 people and a total cost of $12,100 for a trip,
    the cost per person is $1,100. -/
theorem trip_cost_per_person :
  let total_people : ℕ := 11
  let total_cost : ℕ := 12100
  total_cost / total_people = 1100 := by
  sorry

end NUMINAMATH_CALUDE_trip_cost_per_person_l2267_226748
