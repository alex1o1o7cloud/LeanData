import Mathlib

namespace NUMINAMATH_CALUDE_betty_height_in_feet_l3045_304567

/-- Given the heights of Carter, his dog, and Betty, prove Betty's height in feet. -/
theorem betty_height_in_feet :
  ∀ (carter_height dog_height betty_height : ℕ),
    carter_height = 2 * dog_height →
    dog_height = 24 →
    betty_height = carter_height - 12 →
    betty_height / 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_betty_height_in_feet_l3045_304567


namespace NUMINAMATH_CALUDE_smallest_number_l3045_304513

theorem smallest_number (S : Set ℤ) (h : S = {0, -1, 1, -5}) : 
  ∃ x ∈ S, ∀ y ∈ S, x ≤ y ∧ x = -5 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_l3045_304513


namespace NUMINAMATH_CALUDE_min_value_expression_l3045_304588

theorem min_value_expression (a b : ℝ) (h1 : b = 1 + a) (h2 : 0 < b) (h3 : b < 1) :
  ∀ x y : ℝ, x = 1 + y → 0 < x → x < 1 → 
    (2023 / b - (a + 1) / (2023 * a)) ≤ (2023 / x - (y + 1) / (2023 * y)) →
    2023 / b - (a + 1) / (2023 * a) ≥ 2025 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3045_304588


namespace NUMINAMATH_CALUDE_product_sum_equality_l3045_304504

theorem product_sum_equality : ∃ (p q r s : ℝ),
  (∀ x, (4 * x^2 - 6 * x + 5) * (8 - 3 * x) = p * x^3 + q * x^2 + r * x + s) →
  8 * p + 4 * q + 2 * r + s = 18 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_equality_l3045_304504


namespace NUMINAMATH_CALUDE_solve_for_a_l3045_304553

theorem solve_for_a (a : ℝ) : (∀ x : ℝ, a * x + 3 * x = 2) → (1 : ℝ) * a + 3 * (1 : ℝ) = 2 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l3045_304553


namespace NUMINAMATH_CALUDE_holmium166_neutron_proton_difference_l3045_304582

/-- Properties of Holmium-166 isotope -/
structure Holmium166 where
  mass_number : ℕ
  proton_number : ℕ
  mass_number_eq : mass_number = 166
  proton_number_eq : proton_number = 67

/-- Theorem: The difference between neutrons and protons in Holmium-166 is 32 -/
theorem holmium166_neutron_proton_difference (ho : Holmium166) :
  ho.mass_number - ho.proton_number - ho.proton_number = 32 := by
  sorry

#check holmium166_neutron_proton_difference

end NUMINAMATH_CALUDE_holmium166_neutron_proton_difference_l3045_304582


namespace NUMINAMATH_CALUDE_smallest_m_for_integral_multiple_roots_l3045_304594

def has_integral_multiple_roots (m : ℕ) : Prop :=
  ∃ x y : ℤ, x ≠ y ∧ (10 * x^2 - m * x + 360 = 0) ∧ 
             (10 * y^2 - m * y + 360 = 0) ∧
             (x ∣ y ∨ y ∣ x)

theorem smallest_m_for_integral_multiple_roots :
  (has_integral_multiple_roots 120) ∧
  (∀ m : ℕ, m > 0 ∧ m < 120 → ¬(has_integral_multiple_roots m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_for_integral_multiple_roots_l3045_304594


namespace NUMINAMATH_CALUDE_residue_11_1201_mod_19_l3045_304528

theorem residue_11_1201_mod_19 :
  (11 : ℤ) ^ 1201 ≡ 1 [ZMOD 19] := by sorry

end NUMINAMATH_CALUDE_residue_11_1201_mod_19_l3045_304528


namespace NUMINAMATH_CALUDE_organic_egg_tray_price_l3045_304575

/-- The price of a tray of organic eggs -/
def tray_price (individual_price : ℚ) (tray_size : ℕ) (savings_per_egg : ℚ) : ℚ :=
  (individual_price - savings_per_egg) * tray_size / 100

/-- Proof that the price of a tray of 30 organic eggs is $12 -/
theorem organic_egg_tray_price :
  let individual_price : ℚ := 50
  let tray_size : ℕ := 30
  let savings_per_egg : ℚ := 10
  tray_price individual_price tray_size savings_per_egg = 12 := by sorry

end NUMINAMATH_CALUDE_organic_egg_tray_price_l3045_304575


namespace NUMINAMATH_CALUDE_competition_participants_count_l3045_304509

/-- Represents the math competition scenario -/
structure Competition where
  fullScore : ℕ
  initialGoldThreshold : ℕ
  initialSilverLowerThreshold : ℕ
  initialSilverUpperThreshold : ℕ
  changedGoldThreshold : ℕ
  changedSilverLowerThreshold : ℕ
  changedSilverUpperThreshold : ℕ
  initialGoldCount : ℕ
  initialSilverCount : ℕ
  nonMedalCount : ℕ
  changedGoldCount : ℕ
  changedSilverCount : ℕ
  changedGoldAverage : ℕ
  changedSilverAverage : ℕ

/-- The theorem to be proved -/
theorem competition_participants_count (c : Competition) 
  (h1 : c.fullScore = 120)
  (h2 : c.initialGoldThreshold = 100)
  (h3 : c.initialSilverLowerThreshold = 80)
  (h4 : c.initialSilverUpperThreshold = 99)
  (h5 : c.changedGoldThreshold = 90)
  (h6 : c.changedSilverLowerThreshold = 70)
  (h7 : c.changedSilverUpperThreshold = 89)
  (h8 : c.initialSilverCount = c.initialGoldCount + 8)
  (h9 : c.nonMedalCount = c.initialGoldCount + c.initialSilverCount + 9)
  (h10 : c.changedGoldCount = c.initialGoldCount + 5)
  (h11 : c.changedSilverCount = c.initialSilverCount + 5)
  (h12 : c.changedGoldCount * c.changedGoldAverage = c.changedSilverCount * c.changedSilverAverage)
  (h13 : c.changedGoldAverage = 95)
  (h14 : c.changedSilverAverage = 75) :
  c.initialGoldCount + c.initialSilverCount + c.nonMedalCount = 125 :=
sorry


end NUMINAMATH_CALUDE_competition_participants_count_l3045_304509


namespace NUMINAMATH_CALUDE_purchase_cost_l3045_304505

theorem purchase_cost (x y z : ℚ) 
  (eq1 : 4 * x + 9/2 * y + 12 * z = 6)
  (eq2 : 12 * x + 6 * y + 6 * z = 8) :
  4 * x + 3 * y + 6 * z = 4 := by
sorry

end NUMINAMATH_CALUDE_purchase_cost_l3045_304505


namespace NUMINAMATH_CALUDE_three_true_propositions_l3045_304560

-- Define reciprocals
def reciprocals (x y : ℝ) : Prop := x * y = 1

-- Define triangle congruence and area
def triangle_congruent (t1 t2 : Set ℝ × Set ℝ) : Prop := sorry
def triangle_area (t : Set ℝ × Set ℝ) : ℝ := sorry

-- Define the quadratic equation
def has_real_roots (m : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + m = 0

theorem three_true_propositions :
  (∀ x y : ℝ, reciprocals x y → x * y = 1) ∧
  (∃ t1 t2 : Set ℝ × Set ℝ, triangle_area t1 = triangle_area t2 ∧ ¬ triangle_congruent t1 t2) ∧
  (∀ m : ℝ, ¬ has_real_roots m → m > 1) :=
by sorry

end NUMINAMATH_CALUDE_three_true_propositions_l3045_304560


namespace NUMINAMATH_CALUDE_cyrus_additional_bites_l3045_304511

/-- The number of mosquito bites Cyrus initially counted on his arms and legs -/
def initial_bites : ℕ := 14

/-- The number of people in Cyrus's family, excluding Cyrus -/
def family_members : ℕ := 6

/-- The number of additional mosquito bites on Cyrus's body -/
def additional_bites : ℕ := 14

/-- The total number of mosquito bites Cyrus got -/
def total_cyrus_bites : ℕ := initial_bites + additional_bites

/-- The total number of mosquito bites Cyrus's family got -/
def family_bites : ℕ := total_cyrus_bites / 2

/-- The number of mosquito bites each family member got -/
def bites_per_family_member : ℚ := family_bites / family_members

theorem cyrus_additional_bites :
  bites_per_family_member = additional_bites / family_members :=
by sorry

end NUMINAMATH_CALUDE_cyrus_additional_bites_l3045_304511


namespace NUMINAMATH_CALUDE_sally_savings_l3045_304530

/-- Represents the trip expenses and savings for Sally's Sea World trip --/
structure SeaWorldTrip where
  parking_cost : ℕ
  entrance_cost : ℕ
  meal_pass_cost : ℕ
  distance_to_sea_world : ℕ
  car_efficiency : ℕ
  gas_cost_per_gallon : ℕ
  additional_savings_needed : ℕ

/-- Calculates the total cost of the trip --/
def total_cost (trip : SeaWorldTrip) : ℕ :=
  trip.parking_cost + trip.entrance_cost + trip.meal_pass_cost +
  (2 * trip.distance_to_sea_world * trip.gas_cost_per_gallon + trip.car_efficiency - 1) / trip.car_efficiency

/-- Theorem stating that Sally has already saved $28 --/
theorem sally_savings (trip : SeaWorldTrip)
  (h1 : trip.parking_cost = 10)
  (h2 : trip.entrance_cost = 55)
  (h3 : trip.meal_pass_cost = 25)
  (h4 : trip.distance_to_sea_world = 165)
  (h5 : trip.car_efficiency = 30)
  (h6 : trip.gas_cost_per_gallon = 3)
  (h7 : trip.additional_savings_needed = 95) :
  total_cost trip - trip.additional_savings_needed = 28 := by
  sorry


end NUMINAMATH_CALUDE_sally_savings_l3045_304530


namespace NUMINAMATH_CALUDE_quadrilateral_count_l3045_304555

/-- The number of distinct points on the circle -/
def n : ℕ := 12

/-- The number of points needed to form a quadrilateral -/
def k : ℕ := 4

/-- Theorem: The number of ways to choose 4 points from 12 distinct points 
    on a circle to form convex quadrilaterals is equal to 495 -/
theorem quadrilateral_count : Nat.choose n k = 495 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_count_l3045_304555


namespace NUMINAMATH_CALUDE_all_natural_numbers_reachable_l3045_304576

-- Define the operations
def f (n : ℕ) : ℕ := 10 * n

def g (n : ℕ) : ℕ := 10 * n + 4

def h (n : ℕ) : ℕ := n / 2

-- Define the set of reachable numbers
inductive Reachable : ℕ → Prop where
  | start : Reachable 4
  | apply_f {n : ℕ} : Reachable n → Reachable (f n)
  | apply_g {n : ℕ} : Reachable n → Reachable (g n)
  | apply_h {n : ℕ} : Even n → Reachable n → Reachable (h n)

-- Theorem statement
theorem all_natural_numbers_reachable : ∀ m : ℕ, Reachable m := by
  sorry

end NUMINAMATH_CALUDE_all_natural_numbers_reachable_l3045_304576


namespace NUMINAMATH_CALUDE_cube_five_minus_thirteen_equals_square_six_plus_seventysix_l3045_304525

theorem cube_five_minus_thirteen_equals_square_six_plus_seventysix :
  5^3 - 13 = 6^2 + 76 := by
  sorry

end NUMINAMATH_CALUDE_cube_five_minus_thirteen_equals_square_six_plus_seventysix_l3045_304525


namespace NUMINAMATH_CALUDE_acid_solution_mixture_l3045_304574

/-- Given:
  n : ℝ, amount of initial solution in ounces
  y : ℝ, amount of added solution in ounces
  n > 30
  initial solution concentration is n%
  added solution concentration is 20%
  final solution concentration is (n-15)%
Prove: y = 15n / (n+35) -/
theorem acid_solution_mixture (n : ℝ) (y : ℝ) (h1 : n > 30) :
  (n * (n / 100) + y * (20 / 100)) / (n + y) = (n - 15) / 100 →
  y = 15 * n / (n + 35) := by
sorry

end NUMINAMATH_CALUDE_acid_solution_mixture_l3045_304574


namespace NUMINAMATH_CALUDE_de_morgan_laws_l3045_304535

theorem de_morgan_laws (A B : Prop) : 
  (¬(A ∧ B) ↔ ¬A ∨ ¬B) ∧ (¬(A ∨ B) ↔ ¬A ∧ ¬B) := by
  sorry

end NUMINAMATH_CALUDE_de_morgan_laws_l3045_304535


namespace NUMINAMATH_CALUDE_triangular_frame_is_stable_bicycle_frame_triangle_stability_l3045_304573

/-- A bicycle frame is a structure used in bicycles. -/
structure BicycleFrame where
  shape : Type

/-- A triangle is a geometric shape with three sides. -/
inductive Triangle : Type

/-- Stability is a property that can be possessed by structures. -/
class Stable (α : Type) where
  is_stable : α → Prop

/-- A bicycle frame made in the shape of a triangle -/
def triangular_frame : BicycleFrame := { shape := Triangle }

/-- The theorem stating that a triangular bicycle frame is stable -/
theorem triangular_frame_is_stable :
  Stable Triangle → Stable (triangular_frame.shape) :=
by
  sorry

/-- The main theorem proving that a bicycle frame made in the shape of a triangle is stable -/
theorem bicycle_frame_triangle_stability :
  Stable (triangular_frame.shape) :=
by
  sorry

end NUMINAMATH_CALUDE_triangular_frame_is_stable_bicycle_frame_triangle_stability_l3045_304573


namespace NUMINAMATH_CALUDE_complete_sets_l3045_304550

def is_complete (A : Set ℕ) : Prop :=
  ∀ a b : ℕ, (a + b) ∈ A → (a * b) ∈ A

theorem complete_sets :
  ∀ A : Set ℕ, A.Nonempty →
    (is_complete A ↔ 
      A = {1} ∨ 
      A = {1, 2} ∨ 
      A = {1, 2, 3, 4} ∨ 
      A = Set.univ) :=
sorry

end NUMINAMATH_CALUDE_complete_sets_l3045_304550


namespace NUMINAMATH_CALUDE_haley_marbles_l3045_304536

/-- The number of marbles Haley had, given the number of boys and marbles per boy -/
def total_marbles (num_boys : ℕ) (marbles_per_boy : ℕ) : ℕ :=
  num_boys * marbles_per_boy

/-- Theorem stating that Haley had 99 marbles -/
theorem haley_marbles : total_marbles 11 9 = 99 := by
  sorry

end NUMINAMATH_CALUDE_haley_marbles_l3045_304536


namespace NUMINAMATH_CALUDE_subcommittee_formation_ways_l3045_304542

def number_of_ways_to_form_subcommittee (total_republicans : ℕ) (total_democrats : ℕ) 
  (subcommittee_republicans : ℕ) (subcommittee_democrats : ℕ) : ℕ :=
  Nat.choose total_republicans subcommittee_republicans * 
  Nat.choose total_democrats subcommittee_democrats

theorem subcommittee_formation_ways : 
  number_of_ways_to_form_subcommittee 10 8 4 3 = 11760 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_formation_ways_l3045_304542


namespace NUMINAMATH_CALUDE_bus_arrival_probability_l3045_304552

-- Define the probability of the bus arriving on time
def p : ℚ := 3/5

-- Define the probability of the bus not arriving on time
def q : ℚ := 1 - p

-- Define the function to calculate the probability of exactly k successes in n trials
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

-- State the theorem
theorem bus_arrival_probability :
  binomial_probability 3 2 p + binomial_probability 3 3 p = 81/125 := by
  sorry

end NUMINAMATH_CALUDE_bus_arrival_probability_l3045_304552


namespace NUMINAMATH_CALUDE_tan_plus_pi_fourth_implies_cos_double_l3045_304599

theorem tan_plus_pi_fourth_implies_cos_double (θ : Real) : 
  Real.tan (θ + Real.pi / 4) = 3 → Real.cos (2 * θ) = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_tan_plus_pi_fourth_implies_cos_double_l3045_304599


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3045_304566

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) (h : is_geometric_sequence a) :
  a 3 * a 7 = 8 → a 5 = 2 * Real.sqrt 2 ∨ a 5 = -2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3045_304566


namespace NUMINAMATH_CALUDE_fundraiser_customers_l3045_304520

/-- The number of customers who participated in the fundraiser -/
def num_customers : ℕ := 40

/-- The restaurant's donation ratio -/
def restaurant_ratio : ℚ := 2 / 10

/-- The average donation per customer -/
def avg_donation : ℚ := 3

/-- The total donation by the restaurant -/
def total_restaurant_donation : ℚ := 24

/-- Theorem stating that the number of customers is correct given the conditions -/
theorem fundraiser_customers :
  restaurant_ratio * (↑num_customers * avg_donation) = total_restaurant_donation :=
by sorry

end NUMINAMATH_CALUDE_fundraiser_customers_l3045_304520


namespace NUMINAMATH_CALUDE_perimeter_ratio_after_folding_and_cutting_l3045_304571

theorem perimeter_ratio_after_folding_and_cutting (s : ℝ) (h : s > 0) :
  let original_square_perimeter := 4 * s
  let small_rectangle_perimeter := 2 * (s / 2 + s / 4)
  small_rectangle_perimeter / original_square_perimeter = 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_perimeter_ratio_after_folding_and_cutting_l3045_304571


namespace NUMINAMATH_CALUDE_total_fish_caught_l3045_304534

def blaine_fish : ℕ := 5

def keith_fish (blaine : ℕ) : ℕ := 2 * blaine

theorem total_fish_caught : blaine_fish + keith_fish blaine_fish = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_caught_l3045_304534


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3045_304546

theorem min_value_quadratic : 
  ∀ x : ℝ, 3 * x^2 - 12 * x + 908 ≥ 896 ∧ 
  ∃ x₀ : ℝ, 3 * x₀^2 - 12 * x₀ + 908 = 896 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3045_304546


namespace NUMINAMATH_CALUDE_power_of_two_consecutive_zeros_l3045_304590

/-- For any positive integer k, there exists a positive integer n such that
    the decimal representation of 2^n contains exactly k consecutive zeros. -/
theorem power_of_two_consecutive_zeros (k : ℕ) (hk : k ≥ 1) :
  ∃ n : ℕ, ∃ a b : ℕ, ∃ m : ℕ,
    (a ≠ 0) ∧ (b ≠ 0) ∧ (m > k) ∧
    (2^n : ℕ) = a * 10^m + b * 10^(m-k) :=
sorry

end NUMINAMATH_CALUDE_power_of_two_consecutive_zeros_l3045_304590


namespace NUMINAMATH_CALUDE_fraction_equality_l3045_304579

theorem fraction_equality : (3 : ℚ) / (1 + 3 / 5) = 15 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3045_304579


namespace NUMINAMATH_CALUDE_moss_pollen_scientific_notation_l3045_304548

/-- The diameter of a moss flower's pollen in meters -/
def moss_pollen_diameter : ℝ := 0.0000084

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Theorem stating that the moss pollen diameter is equal to its scientific notation representation -/
theorem moss_pollen_scientific_notation :
  ∃ (sn : ScientificNotation), moss_pollen_diameter = sn.coefficient * (10 : ℝ) ^ sn.exponent ∧
  sn.coefficient = 8.4 ∧ sn.exponent = -6 := by
  sorry

end NUMINAMATH_CALUDE_moss_pollen_scientific_notation_l3045_304548


namespace NUMINAMATH_CALUDE_series_solution_l3045_304591

/-- The sum of the infinite geometric series with first term a and common ratio r -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- The series in question -/
noncomputable def series (k : ℝ) : ℝ :=
  5 + geometric_sum ((5 + k) / 3) (1 / 3)

theorem series_solution :
  ∃ k : ℝ, series k = 15 ∧ k = 7.5 := by sorry

end NUMINAMATH_CALUDE_series_solution_l3045_304591


namespace NUMINAMATH_CALUDE_abc_product_l3045_304501

theorem abc_product (a b c : ℝ) (h1 : b + c = 3) (h2 : c + a = 6) (h3 : a + b = 7) : a * b * c = 10 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l3045_304501


namespace NUMINAMATH_CALUDE_unique_real_root_l3045_304569

theorem unique_real_root : ∃! x : ℝ, x^3 + 9 = 0 ∧ x + 3 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_real_root_l3045_304569


namespace NUMINAMATH_CALUDE_minimum_nickels_needed_l3045_304585

def sneaker_cost : ℚ := 42.5
def tax_rate : ℚ := 0.08
def five_dollar_bills : ℕ := 4
def one_dollar_bills : ℕ := 6
def quarters : ℕ := 10

def total_cost : ℚ := sneaker_cost * (1 + tax_rate)

def money_without_nickels : ℚ := 
  (five_dollar_bills * 5) + one_dollar_bills + (quarters * 0.25)

theorem minimum_nickels_needed :
  ∃ n : ℕ, 
    (money_without_nickels + n * 0.05 ≥ total_cost) ∧
    (∀ m : ℕ, m < n → money_without_nickels + m * 0.05 < total_cost) ∧
    n = 348 := by
  sorry

end NUMINAMATH_CALUDE_minimum_nickels_needed_l3045_304585


namespace NUMINAMATH_CALUDE_ice_cream_parlor_distance_l3045_304519

/-- The distance to the ice cream parlor satisfies the equation relating to Rita's canoe trip --/
theorem ice_cream_parlor_distance :
  ∃ D : ℝ, (D / (3 - 2)) + (D / (9 + 4)) = 8 - 0.25 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_parlor_distance_l3045_304519


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_l3045_304570

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

theorem dodecagon_diagonals :
  num_diagonals dodecagon_sides = 54 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_l3045_304570


namespace NUMINAMATH_CALUDE_quadratic_abs_value_analysis_l3045_304544

theorem quadratic_abs_value_analysis (x a : ℝ) :
  (x ≥ a → x^2 + 4*x - 2*|x - a| + 2 - a = x^2 + 2*x + a + 2) ∧
  (x < a → x^2 + 4*x - 2*|x - a| + 2 - a = x^2 + 6*x - 3*a + 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_abs_value_analysis_l3045_304544


namespace NUMINAMATH_CALUDE_prob_ace_jack_queen_is_8_over_16575_l3045_304522

/-- A standard deck of cards. -/
def StandardDeck : ℕ := 52

/-- The number of Aces in a standard deck. -/
def NumAces : ℕ := 4

/-- The number of Jacks in a standard deck. -/
def NumJacks : ℕ := 4

/-- The number of Queens in a standard deck. -/
def NumQueens : ℕ := 4

/-- The probability of drawing an Ace, then a Jack, then a Queen from a standard deck without replacement. -/
def probAceJackQueen : ℚ :=
  (NumAces : ℚ) / StandardDeck *
  (NumJacks : ℚ) / (StandardDeck - 1) *
  (NumQueens : ℚ) / (StandardDeck - 2)

theorem prob_ace_jack_queen_is_8_over_16575 :
  probAceJackQueen = 8 / 16575 := by
  sorry

end NUMINAMATH_CALUDE_prob_ace_jack_queen_is_8_over_16575_l3045_304522


namespace NUMINAMATH_CALUDE_total_spending_is_638_l3045_304515

/-- The total spending of Elizabeth, Emma, and Elsa -/
def total_spending (emma_spending : ℕ) : ℕ :=
  let elsa_spending := 2 * emma_spending
  let elizabeth_spending := 4 * elsa_spending
  emma_spending + elsa_spending + elizabeth_spending

/-- Theorem: The total spending is $638 given the conditions -/
theorem total_spending_is_638 : total_spending 58 = 638 := by
  sorry

end NUMINAMATH_CALUDE_total_spending_is_638_l3045_304515


namespace NUMINAMATH_CALUDE_disease_test_probability_l3045_304506

theorem disease_test_probability (disease_prevalence : ℝ) 
  (test_sensitivity : ℝ) (test_specificity : ℝ) : 
  disease_prevalence = 1/1000 →
  test_sensitivity = 1 →
  test_specificity = 0.95 →
  (disease_prevalence * test_sensitivity) / 
  (disease_prevalence * test_sensitivity + 
   (1 - disease_prevalence) * (1 - test_specificity)) = 100/5095 := by
sorry

end NUMINAMATH_CALUDE_disease_test_probability_l3045_304506


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3045_304598

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x - 3 ≥ 0) ↔ (∃ x : ℝ, x^2 - 2*x - 3 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3045_304598


namespace NUMINAMATH_CALUDE_no_real_solution_for_matrix_equation_l3045_304578

theorem no_real_solution_for_matrix_equation :
  (∀ a b : ℝ, Matrix.det !![a, 2*b; 2*a, b] = a*b - 4*a*b) →
  ¬∃ x : ℝ, Matrix.det !![3*x, 2; 6*x, x] = 6 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_for_matrix_equation_l3045_304578


namespace NUMINAMATH_CALUDE_seating_theorem_l3045_304596

/-- The number of ways to seat n people around a round table. -/
def circular_permutations (n : ℕ) : ℕ := (n - 1).factorial

/-- The number of ways to seat 6 people around a round table,
    with two specific people always sitting next to each other. -/
def seating_arrangements : ℕ :=
  2 * circular_permutations 5

theorem seating_theorem : seating_arrangements = 48 := by
  sorry

end NUMINAMATH_CALUDE_seating_theorem_l3045_304596


namespace NUMINAMATH_CALUDE_cooper_savings_l3045_304521

theorem cooper_savings (total_savings : ℕ) (days_in_year : ℕ) (daily_savings : ℕ) :
  total_savings = 12410 →
  days_in_year = 365 →
  daily_savings * days_in_year = total_savings →
  daily_savings = 34 := by
  sorry

end NUMINAMATH_CALUDE_cooper_savings_l3045_304521


namespace NUMINAMATH_CALUDE_parabola_parameter_values_l3045_304564

/-- Represents a point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- The point satisfies the parabola equation -/
def on_parabola (point : ParabolaPoint) (parabola : Parabola) : Prop :=
  point.y^2 = 2 * parabola.p * point.x

/-- The distance from the point to the directrix (x = -p/2) is 10 -/
def distance_to_directrix (point : ParabolaPoint) (parabola : Parabola) : Prop :=
  point.x + parabola.p / 2 = 10

/-- The distance from the point to the axis of symmetry (y-axis) is 6 -/
def distance_to_axis (point : ParabolaPoint) : Prop :=
  point.y = 6 ∨ point.y = -6

theorem parabola_parameter_values
  (parabola : Parabola)
  (point : ParabolaPoint)
  (h_on_parabola : on_parabola point parabola)
  (h_directrix : distance_to_directrix point parabola)
  (h_axis : distance_to_axis point) :
  parabola.p = 2 ∨ parabola.p = 18 := by
  sorry

end NUMINAMATH_CALUDE_parabola_parameter_values_l3045_304564


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_l3045_304587

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x + 3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

-- Theorem stating that the tangent line equation is correct
theorem tangent_line_at_x_1 : 
  ∀ x y : ℝ, (y = f 1 + f' 1 * (x - 1)) ↔ (2*x - y + 1 = 0) :=
by sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_tangent_line_at_x_1_l3045_304587


namespace NUMINAMATH_CALUDE_highest_score_l3045_304507

theorem highest_score (a b c d : ℝ) 
  (sum_eq : a + b = c + d)
  (sum_ineq : b + d > a + c)
  (a_gt_bc : a > b + c) :
  d > a ∧ d > b ∧ d > c := by
  sorry

end NUMINAMATH_CALUDE_highest_score_l3045_304507


namespace NUMINAMATH_CALUDE_smallest_integer_m_l3045_304558

theorem smallest_integer_m (x y m : ℝ) : 
  (3 * x + y = m + 8) →
  (2 * x + 2 * y = 2 * m + 5) →
  (x - y < 1) →
  (∀ k : ℤ, k < m → k ≤ 2) →
  (m = 3) := by sorry

end NUMINAMATH_CALUDE_smallest_integer_m_l3045_304558


namespace NUMINAMATH_CALUDE_sum_256_64_base_8_l3045_304595

def to_base_8 (n : ℕ) : ℕ := sorry

theorem sum_256_64_base_8 : 
  to_base_8 (256 + 64) = 500 := by sorry

end NUMINAMATH_CALUDE_sum_256_64_base_8_l3045_304595


namespace NUMINAMATH_CALUDE_closest_to_300_l3045_304532

def expression : ℝ := 3.25 * 9.252 * (6.22 + 3.78) - 10

def options : List ℝ := [250, 300, 350, 400, 450]

theorem closest_to_300 : 
  ∀ x ∈ options, x ≠ 300 → |expression - 300| < |expression - x| :=
sorry

end NUMINAMATH_CALUDE_closest_to_300_l3045_304532


namespace NUMINAMATH_CALUDE_smallest_side_of_triangle_l3045_304516

theorem smallest_side_of_triangle (x : ℝ) : 
  10 + (3*x + 6) + (x + 5) = 60 →
  10 ≤ 3*x + 6 ∧ 10 ≤ x + 5 →
  10 = min 10 (min (3*x + 6) (x + 5)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_side_of_triangle_l3045_304516


namespace NUMINAMATH_CALUDE_rectangular_field_area_l3045_304526

/-- Calculates the area of a rectangular field given its perimeter and width-to-length ratio. -/
theorem rectangular_field_area (perimeter : ℝ) (width_ratio : ℝ) : 
  perimeter = 72 ∧ width_ratio = 1/3 → 
  (perimeter / (4 * (1 + width_ratio))) * (perimeter * width_ratio / (4 * (1 + width_ratio))) = 243 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l3045_304526


namespace NUMINAMATH_CALUDE_vlad_sister_height_difference_l3045_304508

/-- Converts feet and inches to total inches -/
def height_to_inches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet * 12 + inches

/-- Calculates the height difference in inches between two people -/
def height_difference (height1_feet height1_inches height2_feet height2_inches : ℕ) : ℕ :=
  (height_to_inches height1_feet height1_inches) - (height_to_inches height2_feet height2_inches)

theorem vlad_sister_height_difference :
  height_difference 6 3 2 10 = 41 := by
  sorry

end NUMINAMATH_CALUDE_vlad_sister_height_difference_l3045_304508


namespace NUMINAMATH_CALUDE_derivative_sin_squared_minus_cos_squared_l3045_304537

theorem derivative_sin_squared_minus_cos_squared (x : ℝ) :
  (deriv (fun x => Real.sin x ^ 2 - Real.cos x ^ 2)) x = 2 * Real.sin (2 * x) :=
by sorry

end NUMINAMATH_CALUDE_derivative_sin_squared_minus_cos_squared_l3045_304537


namespace NUMINAMATH_CALUDE_trapezoid_with_perpendicular_bisector_quadrilateral_is_isosceles_l3045_304503

-- Define the trapezoid and quadrilateral
variable (A B C D K L M N : Point)

-- Define the trapezoid ABCD
def is_trapezoid (A B C D : Point) : Prop := sorry

-- Define the angle bisectors of the trapezoid
def angle_bisectors_intersect (A B C D K L M N : Point) : Prop := sorry

-- Define the quadrilateral KLMN formed by the intersection of angle bisectors
def quadrilateral_from_bisectors (A B C D K L M N : Point) : Prop := sorry

-- Define perpendicular diagonals of KLMN
def perpendicular_diagonals (K L M N : Point) : Prop := sorry

-- Define an isosceles trapezoid
def is_isosceles_trapezoid (A B C D : Point) : Prop := sorry

-- Theorem statement
theorem trapezoid_with_perpendicular_bisector_quadrilateral_is_isosceles 
  (h1 : is_trapezoid A B C D)
  (h2 : angle_bisectors_intersect A B C D K L M N)
  (h3 : quadrilateral_from_bisectors A B C D K L M N)
  (h4 : perpendicular_diagonals K L M N) :
  is_isosceles_trapezoid A B C D := by sorry

end NUMINAMATH_CALUDE_trapezoid_with_perpendicular_bisector_quadrilateral_is_isosceles_l3045_304503


namespace NUMINAMATH_CALUDE_cube_root_negative_eight_properties_l3045_304583

-- Define the cube root function for real numbers
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Main theorem
theorem cube_root_negative_eight_properties :
  let y := cubeRoot (-8)
  ∃ (z : ℝ),
    -- Statement A: y represents the cube root of -8
    z^3 = -8 ∧
    -- Statement B: y results in -2
    y = -2 ∧
    -- Statement C: y is equal to -cubeRoot(8)
    y = -(cubeRoot 8) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_negative_eight_properties_l3045_304583


namespace NUMINAMATH_CALUDE_systematic_sampling_problem_l3045_304559

/-- Systematic sampling function -/
def systematicSample (totalStudents : ℕ) (sampleSize : ℕ) (firstSelected : ℕ) (groupIndex : ℕ) : ℕ :=
  firstSelected + (groupIndex - 1) * (totalStudents / sampleSize)

/-- Theorem for systematic sampling problem -/
theorem systematic_sampling_problem (totalStudents sampleSize firstSelected : ℕ) 
    (h1 : totalStudents = 1000)
    (h2 : sampleSize = 20)
    (h3 : firstSelected = 22) : 
  systematicSample totalStudents sampleSize firstSelected 18 = 872 := by
sorry

#eval systematicSample 1000 20 22 18

end NUMINAMATH_CALUDE_systematic_sampling_problem_l3045_304559


namespace NUMINAMATH_CALUDE_rectangle_area_difference_l3045_304589

theorem rectangle_area_difference : 
  let rect1_width : ℕ := 4
  let rect1_height : ℕ := 5
  let rect2_width : ℕ := 3
  let rect2_height : ℕ := 6
  let rect1_area := rect1_width * rect1_height
  let rect2_area := rect2_width * rect2_height
  rect1_area - rect2_area = 2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_difference_l3045_304589


namespace NUMINAMATH_CALUDE_inequality_theorem_l3045_304568

theorem inequality_theorem (a b : ℝ) (h : a > b) :
  ∀ x : ℝ, (a / (2^x + 1)) > (b / (2^x + 1)) := by
sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3045_304568


namespace NUMINAMATH_CALUDE_valid_fractions_characterization_l3045_304545

def is_valid_fraction (n d : ℕ) : Prop :=
  0 < d ∧ d < 10 ∧ (7:ℚ)/9 < (n:ℚ)/d ∧ (n:ℚ)/d < (8:ℚ)/9

def valid_fractions : Set (ℕ × ℕ) :=
  {(n, d) | is_valid_fraction n d}

theorem valid_fractions_characterization :
  valid_fractions = {(5, 6), (6, 7), (7, 8), (4, 5)} := by sorry

end NUMINAMATH_CALUDE_valid_fractions_characterization_l3045_304545


namespace NUMINAMATH_CALUDE_arithmetic_progression_bijection_l3045_304556

theorem arithmetic_progression_bijection (f : ℕ → ℕ) (hf : Function.Bijective f) :
  ∃ a b c : ℕ, (b - a = c - b) ∧ (f a < f b) ∧ (f b < f c) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_bijection_l3045_304556


namespace NUMINAMATH_CALUDE_todd_ate_cupcakes_l3045_304510

def initial_cupcakes : ℕ := 38
def packages : ℕ := 3
def cupcakes_per_package : ℕ := 8

theorem todd_ate_cupcakes : 
  initial_cupcakes - packages * cupcakes_per_package = 14 := by
  sorry

end NUMINAMATH_CALUDE_todd_ate_cupcakes_l3045_304510


namespace NUMINAMATH_CALUDE_cube_eight_eq_two_power_ten_unique_solution_l3045_304580

theorem cube_eight_eq_two_power_ten :
  8^3 + 8^3 + 8^3 = 2^10 := by
sorry

theorem unique_solution (x : ℕ) :
  8^3 + 8^3 + 8^3 = 2^x → x = 10 := by
sorry

end NUMINAMATH_CALUDE_cube_eight_eq_two_power_ten_unique_solution_l3045_304580


namespace NUMINAMATH_CALUDE_sphere_volume_circumscribing_cube_l3045_304539

/-- The volume of a sphere circumscribing a cube with edge length 2 is 4√3π -/
theorem sphere_volume_circumscribing_cube (cube_edge : ℝ) (sphere_volume : ℝ) : 
  cube_edge = 2 →
  sphere_volume = (4 / 3) * Real.pi * (Real.sqrt 3) ^ 3 →
  sphere_volume = 4 * Real.sqrt 3 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_circumscribing_cube_l3045_304539


namespace NUMINAMATH_CALUDE_benny_stored_bales_l3045_304533

/-- The number of bales Benny stored in the barn -/
def bales_stored (initial_bales current_bales : ℕ) : ℕ :=
  current_bales - initial_bales

/-- Theorem stating that Benny stored 35 bales in the barn -/
theorem benny_stored_bales : 
  let initial_bales : ℕ := 47
  let current_bales : ℕ := 82
  bales_stored initial_bales current_bales = 35 := by
sorry

end NUMINAMATH_CALUDE_benny_stored_bales_l3045_304533


namespace NUMINAMATH_CALUDE_three_at_five_equals_neg_six_l3045_304577

-- Define the @ operation
def at_op (a b : ℤ) : ℤ := 3 * a - 3 * b

-- Theorem statement
theorem three_at_five_equals_neg_six : at_op 3 5 = -6 := by
  sorry

end NUMINAMATH_CALUDE_three_at_five_equals_neg_six_l3045_304577


namespace NUMINAMATH_CALUDE_decimal_93_to_binary_l3045_304529

def decimalToBinary (n : Nat) : List Nat :=
  if n = 0 then [0]
  else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem decimal_93_to_binary :
  decimalToBinary 93 = [1, 0, 1, 1, 1, 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_decimal_93_to_binary_l3045_304529


namespace NUMINAMATH_CALUDE_ellen_painted_twenty_vines_l3045_304500

/-- Represents the time in minutes required to paint different types of flowers and vines. -/
structure PaintingTimes where
  lily : ℕ
  rose : ℕ
  orchid : ℕ
  vine : ℕ

/-- Represents the number of each type of flower and vine painted. -/
structure FlowerCounts where
  lilies : ℕ
  roses : ℕ
  orchids : ℕ
  vines : ℕ

/-- Calculates the total time spent painting given the painting times and flower counts. -/
def totalPaintingTime (times : PaintingTimes) (counts : FlowerCounts) : ℕ :=
  times.lily * counts.lilies + times.rose * counts.roses + 
  times.orchid * counts.orchids + times.vine * counts.vines

/-- Theorem stating that Ellen painted 20 vines given the problem conditions. -/
theorem ellen_painted_twenty_vines 
  (times : PaintingTimes)
  (counts : FlowerCounts)
  (h1 : times.lily = 5)
  (h2 : times.rose = 7)
  (h3 : times.orchid = 3)
  (h4 : times.vine = 2)
  (h5 : counts.lilies = 17)
  (h6 : counts.roses = 10)
  (h7 : counts.orchids = 6)
  (h8 : totalPaintingTime times counts = 213) :
  counts.vines = 20 := by
  sorry

end NUMINAMATH_CALUDE_ellen_painted_twenty_vines_l3045_304500


namespace NUMINAMATH_CALUDE_ice_cream_cones_l3045_304565

theorem ice_cream_cones (cost_per_cone : ℕ) (total_cost : ℕ) (h1 : cost_per_cone = 99) (h2 : total_cost = 198) :
  total_cost / cost_per_cone = 2 :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_cones_l3045_304565


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3045_304554

/-- Given an inequality system {x > 4, x > m} with solution set x > 4, 
    the range of values for m is m ≤ 4 -/
theorem inequality_system_solution (m : ℝ) : 
  (∀ x : ℝ, x > 4 ∧ x > m ↔ x > 4) → m ≤ 4 := by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3045_304554


namespace NUMINAMATH_CALUDE_rectangular_field_width_l3045_304557

theorem rectangular_field_width (length width : ℝ) : 
  length = 24 ∧ length = 2 * width - 3 → width = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_width_l3045_304557


namespace NUMINAMATH_CALUDE_f_sum_zero_a_geq_2_sufficient_not_necessary_l3045_304518

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := Real.log ((2 / (x + 1)) - 1) / Real.log 10

-- Define the domain A of function f
def A : Set ℝ := {x | -1 < x ∧ x < 1}

-- Define the function g (a is a parameter)
def g (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (1 - a^2 - 2*a*x - x^2)

-- Define the domain B of function g
def B (a : ℝ) : Set ℝ := {x | 1 - a^2 - 2*a*x - x^2 ≥ 0}

-- Statement 1: f(1/2013) + f(-1/2013) = 0
theorem f_sum_zero : f (1/2013) + f (-1/2013) = 0 := by sorry

-- Statement 2: a ≥ 2 is sufficient but not necessary for A ∩ B = ∅
theorem a_geq_2_sufficient_not_necessary :
  (∀ a : ℝ, a ≥ 2 → A ∩ B a = ∅) ∧
  ¬(∀ a : ℝ, A ∩ B a = ∅ → a ≥ 2) := by sorry

end

end NUMINAMATH_CALUDE_f_sum_zero_a_geq_2_sufficient_not_necessary_l3045_304518


namespace NUMINAMATH_CALUDE_complex_power_36_135_deg_l3045_304512

theorem complex_power_36_135_deg :
  (Complex.exp (Complex.I * Real.pi * (3 / 4)))^36 = Complex.ofReal (1 / 2) - Complex.I * Complex.ofReal (Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_36_135_deg_l3045_304512


namespace NUMINAMATH_CALUDE_orange_delivery_problem_l3045_304527

def bag_weights : List ℕ := [22, 25, 28, 31, 34, 36, 38, 40, 45]

def total_weight : ℕ := bag_weights.sum

theorem orange_delivery_problem (weights_A B : ℕ) (weight_C : ℕ) :
  weights_A = 2 * weights_B →
  weights_A + weights_B + weight_C = total_weight →
  weight_C ∈ bag_weights →
  weight_C % 3 = 2 →
  weight_C = 38 := by
  sorry

end NUMINAMATH_CALUDE_orange_delivery_problem_l3045_304527


namespace NUMINAMATH_CALUDE_quadratic_one_solution_sum_l3045_304540

theorem quadratic_one_solution_sum (a : ℝ) : 
  let f := fun x : ℝ => 3 * x^2 + a * x + 6 * x + 7
  (∃! x, f x = 0) → 
  ∃ a₁ a₂ : ℝ, a = a₁ ∨ a = a₂ ∧ a₁ + a₂ = -12 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_sum_l3045_304540


namespace NUMINAMATH_CALUDE_current_velocity_l3045_304502

theorem current_velocity (rowing_speed : ℝ) (distance : ℝ) (total_time : ℝ) :
  rowing_speed = 10 ∧ distance = 48 ∧ total_time = 10 →
  ∃ v : ℝ, v = 2 ∧ 
    distance / (rowing_speed - v) + distance / (rowing_speed + v) = total_time :=
by sorry

end NUMINAMATH_CALUDE_current_velocity_l3045_304502


namespace NUMINAMATH_CALUDE_imaginary_unit_power_l3045_304524

theorem imaginary_unit_power (i : ℂ) : i^2 = -1 → i^2015 = -i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_l3045_304524


namespace NUMINAMATH_CALUDE_greater_number_problem_l3045_304586

theorem greater_number_problem (a b : ℝ) (h1 : a + b = 40) (h2 : a - b = 12) (h3 : a > b) : a = 26 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_problem_l3045_304586


namespace NUMINAMATH_CALUDE_solutions_are_correct_l3045_304523

def solutions : Set ℂ := {
  (16/15)^(1/4) + Complex.I * (16/15)^(1/4),
  -(16/15)^(1/4) - Complex.I * (16/15)^(1/4),
  -(16/15)^(1/4) + Complex.I * (16/15)^(1/4),
  (16/15)^(1/4) - Complex.I * (16/15)^(1/4),
  Complex.I * 2^(2/3),
  -Complex.I * 2^(2/3)
}

theorem solutions_are_correct : {z : ℂ | z^6 = -16} = solutions := by
  sorry

end NUMINAMATH_CALUDE_solutions_are_correct_l3045_304523


namespace NUMINAMATH_CALUDE_train_crossing_time_specific_train_crossing_time_l3045_304543

/-- Proves that a train with given length, crossing its own length in a certain time,
    takes the calculated time to cross a platform of given length. -/
theorem train_crossing_time (train_length platform_length cross_own_length_time : ℝ) 
    (train_length_pos : 0 < train_length)
    (platform_length_pos : 0 < platform_length)
    (cross_own_length_time_pos : 0 < cross_own_length_time) :
  let train_speed := train_length / cross_own_length_time
  let total_distance := train_length + platform_length
  let crossing_time := total_distance / train_speed
  crossing_time = 45 :=
by
  sorry

/-- Specific instance of the train crossing problem -/
theorem specific_train_crossing_time :
  let train_length := 300
  let platform_length := 450
  let cross_own_length_time := 18
  let train_speed := train_length / cross_own_length_time
  let total_distance := train_length + platform_length
  let crossing_time := total_distance / train_speed
  crossing_time = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_specific_train_crossing_time_l3045_304543


namespace NUMINAMATH_CALUDE_dividend_calculation_l3045_304563

theorem dividend_calculation (divisor quotient remainder : ℕ) : 
  divisor = 10 * quotient →
  divisor = 5 * remainder →
  remainder = 46 →
  divisor * quotient + remainder = 5336 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3045_304563


namespace NUMINAMATH_CALUDE_f_non_monotonic_l3045_304517

/-- A piecewise function f defined on ℝ with a parameter a and a split point t -/
noncomputable def f (a t : ℝ) (x : ℝ) : ℝ :=
  if x ≤ t then (2*a - 1)*x + 3*a - 4 else x^3 - x

/-- The theorem stating the condition for non-monotonicity of f -/
theorem f_non_monotonic (a : ℝ) :
  (∀ t : ℝ, ¬ Monotone (f a t)) ↔ a ≤ (1/2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_f_non_monotonic_l3045_304517


namespace NUMINAMATH_CALUDE_speeding_ticket_percentage_l3045_304547

/-- The percentage of motorists who exceed the speed limit -/
def exceed_limit : ℝ := 16.666666666666664

/-- The percentage of speeding motorists who do not receive tickets -/
def no_ticket_rate : ℝ := 40

/-- The percentage of motorists who receive speeding tickets -/
def receive_ticket : ℝ := 10

theorem speeding_ticket_percentage :
  receive_ticket = exceed_limit * (1 - no_ticket_rate / 100) := by sorry

end NUMINAMATH_CALUDE_speeding_ticket_percentage_l3045_304547


namespace NUMINAMATH_CALUDE_C_is_rotated_X_l3045_304581

/-- A shape in a 2D plane -/
structure Shape :=
  (points : Set (ℝ × ℝ))

/-- Rotation of a shape by 90 degrees clockwise around its center -/
def rotate90 (s : Shape) : Shape := sorry

/-- Two shapes are superimposable if they have the same set of points -/
def superimposable (s1 s2 : Shape) : Prop :=
  s1.points = s2.points

/-- The original shape X -/
def X : Shape := sorry

/-- The alternative shapes -/
def A : Shape := sorry
def B : Shape := sorry
def C : Shape := sorry
def D : Shape := sorry
def E : Shape := sorry

/-- The theorem stating that C is the only shape superimposable with X after rotation -/
theorem C_is_rotated_X : 
  superimposable (rotate90 X) C ∧ 
  (¬superimposable (rotate90 X) A ∧
   ¬superimposable (rotate90 X) B ∧
   ¬superimposable (rotate90 X) D ∧
   ¬superimposable (rotate90 X) E) :=
sorry

end NUMINAMATH_CALUDE_C_is_rotated_X_l3045_304581


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3045_304549

-- Problem 1
theorem simplify_expression_1 (m n : ℝ) : 
  (2*m + n)^2 - (4*m + 3*n)*(m - n) = 8*m*n + 4*n^2 := by sorry

-- Problem 2
theorem simplify_expression_2 (x : ℝ) 
  (h1 : x ≠ 3) (h2 : 2*x^2 - 5*x - 3 ≠ 0) : 
  ((2*x + 1)*(3*x - 4) / (2*x^2 - 5*x - 3) - 1) / ((4*x^2 - 1) / (x - 3)) = 1 / (2*x + 1) := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3045_304549


namespace NUMINAMATH_CALUDE_no_integer_solution_l3045_304562

theorem no_integer_solution :
  ¬ ∃ (x y z : ℤ), 
    (x^6 + x^3 + x^3*y + y = 147^157) ∧ 
    (x^3 + x^3*y + y^2 + y + z^9 = 157^147) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3045_304562


namespace NUMINAMATH_CALUDE_false_or_false_is_false_l3045_304538

theorem false_or_false_is_false (p q : Prop) (hp : ¬p) (hq : ¬q) : ¬(p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_false_or_false_is_false_l3045_304538


namespace NUMINAMATH_CALUDE_parkway_soccer_players_l3045_304572

theorem parkway_soccer_players (total_students : ℕ) (boys : ℕ) (girls_not_playing : ℕ) :
  total_students = 420 →
  boys = 312 →
  girls_not_playing = 53 →
  ∃ (soccer_players : ℕ),
    soccer_players = 250 ∧
    (soccer_players : ℚ) * (78 / 100) = boys - (total_students - boys - girls_not_playing) :=
by sorry

end NUMINAMATH_CALUDE_parkway_soccer_players_l3045_304572


namespace NUMINAMATH_CALUDE_binomial_and_factorial_l3045_304531

theorem binomial_and_factorial : 
  (Nat.choose 10 5 = 252) ∧ (Nat.factorial (Nat.choose 10 5 - 5) = Nat.factorial 247) := by
  sorry

end NUMINAMATH_CALUDE_binomial_and_factorial_l3045_304531


namespace NUMINAMATH_CALUDE_equation_undefined_at_five_l3045_304592

theorem equation_undefined_at_five :
  ¬∃ (y : ℝ), (1 / (5 + 5) + 1 / (5 - 5) : ℝ) = y :=
sorry

end NUMINAMATH_CALUDE_equation_undefined_at_five_l3045_304592


namespace NUMINAMATH_CALUDE_cycling_time_difference_l3045_304593

-- Define the distances and speeds for each day
def monday_distance : ℝ := 3
def monday_speed : ℝ := 6
def tuesday_distance : ℝ := 4
def tuesday_speed : ℝ := 4
def thursday_distance : ℝ := 3
def thursday_speed : ℝ := 3
def saturday_distance : ℝ := 2
def saturday_speed : ℝ := 8

-- Define the constant speed
def constant_speed : ℝ := 5

-- Define the total distance
def total_distance : ℝ := monday_distance + tuesday_distance + thursday_distance + saturday_distance

-- Theorem statement
theorem cycling_time_difference : 
  let actual_time := (monday_distance / monday_speed) + 
                     (tuesday_distance / tuesday_speed) + 
                     (thursday_distance / thursday_speed) + 
                     (saturday_distance / saturday_speed)
  let constant_time := total_distance / constant_speed
  ((actual_time - constant_time) * 60) = 21 := by
  sorry

end NUMINAMATH_CALUDE_cycling_time_difference_l3045_304593


namespace NUMINAMATH_CALUDE_root_sum_eighth_power_l3045_304597

theorem root_sum_eighth_power (r s : ℝ) : 
  (r^2 - 2*r*Real.sqrt 6 + 3 = 0) →
  (s^2 - 2*s*Real.sqrt 6 + 3 = 0) →
  r^8 + s^8 = 93474 := by
sorry

end NUMINAMATH_CALUDE_root_sum_eighth_power_l3045_304597


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l3045_304584

/-- Given that x is inversely proportional to y, prove that when x = 8 and y = 16,
    then x = -4 when y = -32 -/
theorem inverse_proportion_problem (x y : ℝ) (c : ℝ) 
    (h1 : x * y = c)  -- x is inversely proportional to y
    (h2 : 8 * 16 = c) -- When x = 8, y = 16
    (h3 : y = -32)    -- Given y = -32
    : x = -4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l3045_304584


namespace NUMINAMATH_CALUDE_lexus_cars_sold_l3045_304551

def total_cars : ℕ := 300

def audi_percent : ℚ := 10 / 100
def toyota_percent : ℚ := 25 / 100
def bmw_percent : ℚ := 15 / 100
def acura_percent : ℚ := 30 / 100

def other_brands_percent : ℚ := audi_percent + toyota_percent + bmw_percent + acura_percent

def lexus_percent : ℚ := 1 - other_brands_percent

theorem lexus_cars_sold : 
  ⌊(lexus_percent * total_cars : ℚ)⌋ = 60 := by
  sorry

end NUMINAMATH_CALUDE_lexus_cars_sold_l3045_304551


namespace NUMINAMATH_CALUDE_cosine_identity_l3045_304561

theorem cosine_identity (α : ℝ) :
  3 - 4 * Real.cos (4 * α - 3 * Real.pi) - Real.cos (5 * Real.pi + 8 * α) = 8 * (Real.cos (2 * α))^4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_identity_l3045_304561


namespace NUMINAMATH_CALUDE_parabola_p_value_l3045_304514

/-- Given a parabola with equation y^2 = 2px and axis of symmetry x = -1, prove that p = 2 -/
theorem parabola_p_value (p : ℝ) : 
  (∀ x y : ℝ, y^2 = 2*p*x) → 
  (∀ y : ℝ, y^2 = -2*p) → 
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_p_value_l3045_304514


namespace NUMINAMATH_CALUDE_triangle_options_l3045_304541

/-- Represents a triangle with side lengths a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Checks if a triangle is right-angled -/
def isRightAngled (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2 ∨ t.a^2 + t.c^2 = t.b^2 ∨ t.b^2 + t.c^2 = t.a^2 ∨
  t.A = 90 ∨ t.B = 90 ∨ t.C = 90

theorem triangle_options (t : Triangle) :
  (t.b^2 = t.a^2 - t.c^2 → isRightAngled t) ∧
  (t.a / t.b = 3 / 4 ∧ t.a / t.c = 3 / 5 ∧ t.b / t.c = 4 / 5 → isRightAngled t) ∧
  (t.C = t.A - t.B → isRightAngled t) ∧
  (t.A / t.B = 3 / 4 ∧ t.A / t.C = 3 / 5 ∧ t.B / t.C = 4 / 5 → ¬isRightAngled t) :=
by sorry


end NUMINAMATH_CALUDE_triangle_options_l3045_304541
