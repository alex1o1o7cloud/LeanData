import Mathlib

namespace NUMINAMATH_CALUDE_collinear_probability_5x4_l3598_359851

/-- A rectangular array of dots -/
structure DotArray :=
  (rows : ℕ)
  (cols : ℕ)

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of collinear sets of 4 dots in a DotArray -/
def collinearSets (arr : DotArray) : ℕ := sorry

/-- The probability of choosing 4 collinear dots from a DotArray -/
def collinearProbability (arr : DotArray) : ℚ :=
  (collinearSets arr : ℚ) / choose (arr.rows * arr.cols) 4

/-- The main theorem -/
theorem collinear_probability_5x4 :
  collinearProbability ⟨5, 4⟩ = 2 / 4845 := by sorry

end NUMINAMATH_CALUDE_collinear_probability_5x4_l3598_359851


namespace NUMINAMATH_CALUDE_intersecting_chords_theorem_l3598_359827

theorem intersecting_chords_theorem (chord1_segment1 chord1_segment2 chord2_ratio1 chord2_ratio2 : ℝ) :
  chord1_segment1 = 12 →
  chord1_segment2 = 18 →
  chord2_ratio1 = 3 →
  chord2_ratio2 = 8 →
  ∃ (chord2_length : ℝ),
    chord2_length = chord2_ratio1 / (chord2_ratio1 + chord2_ratio2) * chord2_length +
                    chord2_ratio2 / (chord2_ratio1 + chord2_ratio2) * chord2_length ∧
    chord1_segment1 * chord1_segment2 = (chord2_ratio1 / (chord2_ratio1 + chord2_ratio2) * chord2_length) *
                                        (chord2_ratio2 / (chord2_ratio1 + chord2_ratio2) * chord2_length) →
    chord2_length = 33 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_chords_theorem_l3598_359827


namespace NUMINAMATH_CALUDE_second_month_bill_l3598_359804

/-- Represents Elvin's telephone bill structure -/
structure TelephoneBill where
  internetCharge : ℝ
  callCharge : ℝ

/-- Calculates the total bill given internet and call charges -/
def totalBill (bill : TelephoneBill) : ℝ :=
  bill.internetCharge + bill.callCharge

theorem second_month_bill 
  (januaryBill : TelephoneBill) 
  (h1 : totalBill januaryBill = 40) 
  (secondMonthBill : TelephoneBill) 
  (h2 : secondMonthBill.internetCharge = januaryBill.internetCharge)
  (h3 : secondMonthBill.callCharge = 2 * januaryBill.callCharge) :
  totalBill secondMonthBill = 40 + januaryBill.callCharge := by
  sorry

#check second_month_bill

end NUMINAMATH_CALUDE_second_month_bill_l3598_359804


namespace NUMINAMATH_CALUDE_rachel_removed_bottle_caps_l3598_359831

/-- The number of bottle caps Rachel removed from a jar --/
def bottleCapsRemoved (originalCount remainingCount : ℕ) : ℕ :=
  originalCount - remainingCount

/-- Theorem: The number of bottle caps Rachel removed is equal to the difference
    between the original number and the remaining number of bottle caps --/
theorem rachel_removed_bottle_caps :
  bottleCapsRemoved 87 40 = 47 := by
  sorry

end NUMINAMATH_CALUDE_rachel_removed_bottle_caps_l3598_359831


namespace NUMINAMATH_CALUDE_coefficient_of_3_squared_x_squared_l3598_359836

/-- Definition of a coefficient in an algebraic term -/
def is_coefficient (c : ℝ) (term : ℝ → ℝ) : Prop :=
  ∃ (f : ℝ → ℝ), ∀ x, term x = c * f x

/-- The coefficient of 3^2 * x^2 is 3^2 -/
theorem coefficient_of_3_squared_x_squared :
  is_coefficient (3^2) (λ x => 3^2 * x^2) :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_3_squared_x_squared_l3598_359836


namespace NUMINAMATH_CALUDE_no_function_satisfies_inequality_l3598_359812

/-- There does not exist a function satisfying the given inequality for all real numbers. -/
theorem no_function_satisfies_inequality :
  ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, (f x + f y) / 2 ≥ f ((x + y) / 2) + |x - y| := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_inequality_l3598_359812


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_180_4620_l3598_359803

theorem gcd_lcm_sum_180_4620 : 
  Nat.gcd 180 4620 + Nat.lcm 180 4620 = 13920 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_180_4620_l3598_359803


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3598_359885

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_product : a 1 * a 7 = 36) :
  a 4 = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3598_359885


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3598_359813

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 1 → x^2 > 1) ↔ (∃ x : ℝ, x > 1 ∧ x^2 ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3598_359813


namespace NUMINAMATH_CALUDE_water_in_pool_is_34_l3598_359819

/-- Calculates the amount of water in Carol's pool after five hours of filling and leaking -/
def water_in_pool : ℕ :=
  let first_hour : ℕ := 8
  let next_two_hours : ℕ := 10 * 2
  let fourth_hour : ℕ := 14
  let leak : ℕ := 8
  (first_hour + next_two_hours + fourth_hour) - leak

/-- Theorem stating that the amount of water in the pool after five hours is 34 gallons -/
theorem water_in_pool_is_34 : water_in_pool = 34 := by
  sorry

end NUMINAMATH_CALUDE_water_in_pool_is_34_l3598_359819


namespace NUMINAMATH_CALUDE_book_purchase_change_l3598_359864

/-- Calculates the change received when buying two items with given prices and paying with a given amount. -/
def calculate_change (price1 : ℝ) (price2 : ℝ) (payment : ℝ) : ℝ :=
  payment - (price1 + price2)

/-- Theorem stating that buying two books priced at 5.5£ and 6.5£ with a 20£ bill results in 8£ change. -/
theorem book_purchase_change : calculate_change 5.5 6.5 20 = 8 := by
  sorry

end NUMINAMATH_CALUDE_book_purchase_change_l3598_359864


namespace NUMINAMATH_CALUDE_equal_commission_l3598_359852

/-- The list price of the item -/
def list_price : ℝ := 34

/-- Alice's selling price -/
def alice_price (x : ℝ) : ℝ := x - 15

/-- Bob's selling price -/
def bob_price (x : ℝ) : ℝ := x - 25

/-- Alice's commission rate -/
def alice_rate : ℝ := 0.12

/-- Bob's commission rate -/
def bob_rate : ℝ := 0.25

/-- Alice's commission -/
def alice_commission (x : ℝ) : ℝ := alice_rate * alice_price x

/-- Bob's commission -/
def bob_commission (x : ℝ) : ℝ := bob_rate * bob_price x

theorem equal_commission :
  alice_commission list_price = bob_commission list_price :=
sorry

end NUMINAMATH_CALUDE_equal_commission_l3598_359852


namespace NUMINAMATH_CALUDE_pastries_count_l3598_359899

/-- The number of pastries made by Lola and Lulu -/
def total_pastries (lola_cupcakes lola_poptarts lola_pies lulu_cupcakes lulu_poptarts lulu_pies : ℕ) : ℕ :=
  lola_cupcakes + lola_poptarts + lola_pies + lulu_cupcakes + lulu_poptarts + lulu_pies

/-- Theorem stating the total number of pastries made by Lola and Lulu -/
theorem pastries_count : total_pastries 13 10 8 16 12 14 = 73 := by
  sorry

end NUMINAMATH_CALUDE_pastries_count_l3598_359899


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3598_359887

theorem line_passes_through_fixed_point :
  ∀ (a : ℝ), (3 * a - 1 + 1 - 3 * a = 0) := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3598_359887


namespace NUMINAMATH_CALUDE_regression_line_equation_l3598_359882

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a linear equation of the form y = mx + b -/
structure LinearEquation where
  slope : ℝ
  intercept : ℝ

/-- Given a regression line with slope 1.2 passing through (4,5), prove its equation is ŷ = 1.2x + 0.2 -/
theorem regression_line_equation 
  (slope : ℝ) 
  (center : Point)
  (h1 : slope = 1.2)
  (h2 : center = ⟨4, 5⟩)
  : ∃ (eq : LinearEquation), 
    eq.slope = slope ∧ 
    eq.intercept = 0.2 ∧ 
    center.y = eq.slope * center.x + eq.intercept := by
  sorry

end NUMINAMATH_CALUDE_regression_line_equation_l3598_359882


namespace NUMINAMATH_CALUDE_opposite_of_2023_l3598_359881

-- Define the concept of opposite for real numbers
def opposite (x : ℝ) : ℝ := -x

-- State the theorem
theorem opposite_of_2023 : opposite 2023 = -2023 := by
  -- The proof would go here, but we're skipping it as requested
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l3598_359881


namespace NUMINAMATH_CALUDE_dilution_proof_l3598_359825

/-- Given a solution of 12 ounces with 60% alcohol concentration, 
    adding 6 ounces of water results in a 40% alcohol solution -/
theorem dilution_proof (initial_volume : ℝ) (initial_concentration : ℝ) 
                       (water_added : ℝ) (final_concentration : ℝ) : 
  initial_volume = 12 ∧ 
  initial_concentration = 0.6 ∧ 
  water_added = 6 ∧ 
  final_concentration = 0.4 → 
  initial_volume * initial_concentration = 
  (initial_volume + water_added) * final_concentration := by
  sorry

#check dilution_proof

end NUMINAMATH_CALUDE_dilution_proof_l3598_359825


namespace NUMINAMATH_CALUDE_rachel_math_problems_l3598_359880

theorem rachel_math_problems (minutes_before_bed : ℕ) (problems_next_day : ℕ) (total_problems : ℕ)
  (h1 : minutes_before_bed = 12)
  (h2 : problems_next_day = 16)
  (h3 : total_problems = 76) :
  ∃ (problems_per_minute : ℕ),
    problems_per_minute * minutes_before_bed + problems_next_day = total_problems ∧
    problems_per_minute = 5 := by
  sorry

end NUMINAMATH_CALUDE_rachel_math_problems_l3598_359880


namespace NUMINAMATH_CALUDE_line_through_point_and_trisection_l3598_359868

/-- The line passing through (2,3) and one of the trisection points of the line segment
    joining (1,2) and (7,-4) has the equation 4x - 9y + 15 = 0 -/
theorem line_through_point_and_trisection :
  ∃ (t : ℝ) (x y : ℝ),
    -- Define the trisection point
    x = 1 + 2 * t * (7 - 1) ∧
    y = 2 + 2 * t * (-4 - 2) ∧
    0 ≤ t ∧ t ≤ 1 ∧
    -- The trisection point is on the line
    4 * x - 9 * y + 15 = 0 ∧
    -- The point (2,3) is on the line
    4 * 2 - 9 * 3 + 15 = 0 :=
sorry

end NUMINAMATH_CALUDE_line_through_point_and_trisection_l3598_359868


namespace NUMINAMATH_CALUDE_students_without_A_l3598_359823

theorem students_without_A (total : ℕ) (history : ℕ) (math : ℕ) (science : ℕ)
  (history_math : ℕ) (history_science : ℕ) (math_science : ℕ) (all_three : ℕ) :
  total = 45 →
  history = 11 →
  math = 16 →
  science = 9 →
  history_math = 5 →
  history_science = 3 →
  math_science = 4 →
  all_three = 2 →
  total - (history + math + science - history_math - history_science - math_science + all_three) = 19 :=
by sorry

end NUMINAMATH_CALUDE_students_without_A_l3598_359823


namespace NUMINAMATH_CALUDE_lucy_flour_purchase_l3598_359865

/-- Calculates the amount of flour needed to replenish stock --/
def flour_to_buy (initial : ℕ) (used : ℕ) (full_bag : ℕ) : ℕ :=
  let remaining := initial - used
  let after_spill := remaining / 2
  full_bag - after_spill

/-- Theorem: Given the initial conditions, Lucy needs to buy 370g of flour --/
theorem lucy_flour_purchase :
  flour_to_buy 500 240 500 = 370 := by
  sorry

end NUMINAMATH_CALUDE_lucy_flour_purchase_l3598_359865


namespace NUMINAMATH_CALUDE_combined_savings_equals_individual_savings_l3598_359826

-- Define the regular price of a window
def regular_price : ℕ := 120

-- Define the offer: for every 5 windows purchased, 2 are free
def offer (n : ℕ) : ℕ := (n / 5) * 2

-- Calculate the cost for a given number of windows with the offer
def cost_with_offer (n : ℕ) : ℕ :=
  regular_price * (n - offer n)

-- Calculate savings for a given number of windows
def savings (n : ℕ) : ℕ :=
  regular_price * n - cost_with_offer n

-- Dave's required windows
def dave_windows : ℕ := 9

-- Doug's required windows
def doug_windows : ℕ := 10

-- Combined windows
def combined_windows : ℕ := dave_windows + doug_windows

-- Theorem: Combined savings equals sum of individual savings
theorem combined_savings_equals_individual_savings :
  savings combined_windows = savings dave_windows + savings doug_windows :=
sorry

end NUMINAMATH_CALUDE_combined_savings_equals_individual_savings_l3598_359826


namespace NUMINAMATH_CALUDE_race_time_calculation_l3598_359814

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ

/-- Represents the race scenario -/
structure Race where
  distance : ℝ
  runner_a : Runner
  runner_b : Runner
  time_difference : ℝ
  distance_difference : ℝ

/-- The theorem to prove -/
theorem race_time_calculation (race : Race) 
  (h1 : race.distance = 1000)
  (h2 : race.time_difference = 10)
  (h3 : race.distance_difference = 20) :
  ∃ (t : ℝ), t = 490 ∧ t * race.runner_a.speed = race.distance :=
sorry

end NUMINAMATH_CALUDE_race_time_calculation_l3598_359814


namespace NUMINAMATH_CALUDE_binomial_prob_l3598_359866

/-- A random variable following a binomial distribution B(2,p) -/
def X (p : ℝ) : Type := Unit

/-- The probability that X is greater than or equal to 1 -/
def prob_X_geq_1 (p : ℝ) : ℝ := 1 - (1 - p)^2

/-- The theorem stating that if P(X ≥ 1) = 5/9 for X ~ B(2,p), then p = 1/3 -/
theorem binomial_prob (p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 1) :
  prob_X_geq_1 p = 5/9 → p = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_prob_l3598_359866


namespace NUMINAMATH_CALUDE_vector_problem_l3598_359830

/-- Given two non-collinear vectors e₁ and e₂ in a real vector space,
    prove that if CB = e₁ + 3e₂, CD = 2e₁ - e₂, BF = 3e₁ - ke₂,
    and points B, D, and F are collinear, then k = 12. -/
theorem vector_problem (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (e₁ e₂ : V) (hne : ¬ ∃ (r : ℝ), e₁ = r • e₂) 
  (CB CD BF : V)
  (hCB : CB = e₁ + 3 • e₂)
  (hCD : CD = 2 • e₁ - e₂)
  (k : ℝ)
  (hBF : BF = 3 • e₁ - k • e₂)
  (hcollinear : ∃ (t : ℝ), BF = t • (CD - CB)) :
  k = 12 := by sorry

end NUMINAMATH_CALUDE_vector_problem_l3598_359830


namespace NUMINAMATH_CALUDE_second_week_collection_l3598_359810

def total_goal : ℕ := 500
def first_week : ℕ := 158
def cans_needed : ℕ := 83

theorem second_week_collection : 
  total_goal - first_week - cans_needed = 259 := by
  sorry

end NUMINAMATH_CALUDE_second_week_collection_l3598_359810


namespace NUMINAMATH_CALUDE_sin_arccos_tan_arcsin_product_one_l3598_359832

theorem sin_arccos_tan_arcsin_product_one :
  ∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧
  x₁ ≠ x₂ ∧
  (∀ (x : ℝ), x > 0 → Real.sin (Real.arccos (Real.tan (Real.arcsin x))) = x → (x = x₁ ∨ x = x₂)) ∧
  x₁ * x₂ = 1 :=
sorry

end NUMINAMATH_CALUDE_sin_arccos_tan_arcsin_product_one_l3598_359832


namespace NUMINAMATH_CALUDE_constant_e_value_l3598_359853

theorem constant_e_value (x y e : ℝ) 
  (h1 : x / (2 * y) = 5 / e)
  (h2 : (7 * x + 4 * y) / (x - 2 * y) = 13) :
  e = 2 := by
  sorry

end NUMINAMATH_CALUDE_constant_e_value_l3598_359853


namespace NUMINAMATH_CALUDE_digit_product_over_21_l3598_359857

theorem digit_product_over_21 (c d : ℕ) : 
  (c < 10 ∧ d < 10) → -- c and d are base-10 digits
  (7 * 7 * 7 + 6 * 7 + 5 = 400 + 10 * c + d) → -- 765₇ = 4cd₁₀
  (c * d : ℚ) / 21 = 9 / 7 := by
  sorry

end NUMINAMATH_CALUDE_digit_product_over_21_l3598_359857


namespace NUMINAMATH_CALUDE_loss_percentage_is_five_percent_l3598_359860

def original_price : ℚ := 490
def sale_price : ℚ := 465.5

def loss_amount : ℚ := original_price - sale_price

def loss_percentage : ℚ := (loss_amount / original_price) * 100

theorem loss_percentage_is_five_percent :
  loss_percentage = 5 := by
  sorry

end NUMINAMATH_CALUDE_loss_percentage_is_five_percent_l3598_359860


namespace NUMINAMATH_CALUDE_unique_digit_subtraction_l3598_359872

theorem unique_digit_subtraction (A B C D : Nat) :
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 →
  1000 * A + 100 * B + 10 * C + D - 989 = 109 →
  1000 * A + 100 * B + 10 * C + D = 1908 := by
sorry

end NUMINAMATH_CALUDE_unique_digit_subtraction_l3598_359872


namespace NUMINAMATH_CALUDE_polynomial_product_no_x_terms_l3598_359889

theorem polynomial_product_no_x_terms
  (a b : ℚ)
  (h1 : a ≠ 0)
  (h2 : ∀ x : ℚ, (a * x^2 + b * x + 1) * (3 * x - 2) = 3 * a * x^3 - 2) :
  a = 9/4 ∧ b = 3/2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_product_no_x_terms_l3598_359889


namespace NUMINAMATH_CALUDE_compound_composition_l3598_359834

/-- Represents the number of atoms of each element in the compound -/
structure Compound where
  h : ℕ
  c : ℕ
  o : ℕ

/-- Atomic weights of elements in g/mol -/
def atomic_weight (element : String) : ℝ :=
  match element with
  | "H" => 1
  | "C" => 12
  | "O" => 16
  | _ => 0

/-- Calculate the molecular weight of a compound -/
def molecular_weight (comp : Compound) : ℝ :=
  comp.h * atomic_weight "H" + comp.c * atomic_weight "C" + comp.o * atomic_weight "O"

/-- The main theorem to prove -/
theorem compound_composition :
  ∃ (comp : Compound), comp.h = 2 ∧ comp.o = 3 ∧ molecular_weight comp = 62 ∧ comp.c = 1 :=
by sorry

end NUMINAMATH_CALUDE_compound_composition_l3598_359834


namespace NUMINAMATH_CALUDE_doctor_team_formations_l3598_359877

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

theorem doctor_team_formations :
  let total_doctors : ℕ := 9
  let male_doctors : ℕ := 5
  let female_doctors : ℕ := 4
  let team_size : ℕ := 3
  let one_male_two_female : ℕ := choose male_doctors 1 * choose female_doctors 2
  let two_male_one_female : ℕ := choose male_doctors 2 * choose female_doctors 1
  one_male_two_female + two_male_one_female = 70 :=
sorry

end NUMINAMATH_CALUDE_doctor_team_formations_l3598_359877


namespace NUMINAMATH_CALUDE_peas_soybean_mixture_ratio_l3598_359829

/-- Proves that the ratio of peas to soybean in a mixture costing Rs. 19/kg is 2:1,
    given that peas cost Rs. 16/kg and soybean costs Rs. 25/kg. -/
theorem peas_soybean_mixture_ratio : 
  ∀ (x y : ℝ), 
    x > 0 → y > 0 →
    16 * x + 25 * y = 19 * (x + y) →
    x / y = 2 := by
  sorry

end NUMINAMATH_CALUDE_peas_soybean_mixture_ratio_l3598_359829


namespace NUMINAMATH_CALUDE_dictionary_page_count_l3598_359891

/-- Count the occurrences of digit 1 in a number -/
def countOnesInNumber (n : ℕ) : ℕ := sorry

/-- Count the occurrences of digit 1 in a range of numbers from 1 to n -/
def countOnesInRange (n : ℕ) : ℕ := sorry

/-- The number of pages in the dictionary -/
def dictionaryPages : ℕ := 3152

/-- The total count of digit 1 appearances -/
def totalOnesCount : ℕ := 1988

theorem dictionary_page_count :
  countOnesInRange dictionaryPages = totalOnesCount ∧
  ∀ m : ℕ, m < dictionaryPages → countOnesInRange m < totalOnesCount :=
sorry

end NUMINAMATH_CALUDE_dictionary_page_count_l3598_359891


namespace NUMINAMATH_CALUDE_room_painting_time_l3598_359820

theorem room_painting_time 
  (alice_rate : ℝ) 
  (bob_rate : ℝ) 
  (carla_rate : ℝ) 
  (t : ℝ) 
  (h_alice : alice_rate = 1 / 6) 
  (h_bob : bob_rate = 1 / 8) 
  (h_carla : carla_rate = 1 / 12) 
  (h_combined_work : (alice_rate + bob_rate + carla_rate) * t = 1) : 
  (1 / 6 + 1 / 8 + 1 / 12) * t = 1 := by
  sorry

end NUMINAMATH_CALUDE_room_painting_time_l3598_359820


namespace NUMINAMATH_CALUDE_factorial_fraction_simplification_l3598_359828

theorem factorial_fraction_simplification :
  (4 * Nat.factorial 6 + 20 * Nat.factorial 5) / Nat.factorial 7 = 22 / 21 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_simplification_l3598_359828


namespace NUMINAMATH_CALUDE_a_gt_2_sufficient_not_necessary_l3598_359800

theorem a_gt_2_sufficient_not_necessary :
  (∀ a : ℝ, a > 2 → 2^a - a - 1 > 0) ∧
  (∃ a : ℝ, a ≤ 2 ∧ 2^a - a - 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_a_gt_2_sufficient_not_necessary_l3598_359800


namespace NUMINAMATH_CALUDE_power_sum_problem_l3598_359855

theorem power_sum_problem (a b x y : ℝ) 
  (h1 : 2*a*x + 3*b*y = 6)
  (h2 : 2*a*x^2 + 3*b*y^2 = 14)
  (h3 : 2*a*x^3 + 3*b*y^3 = 33)
  (h4 : 2*a*x^4 + 3*b*y^4 = 87) :
  2*a*x^5 + 3*b*y^5 = 528 := by
sorry

end NUMINAMATH_CALUDE_power_sum_problem_l3598_359855


namespace NUMINAMATH_CALUDE_compound_weight_l3598_359843

/-- Given a compound with a molecular weight of 2670 grams/mole, 
    prove that the total weight of 10 moles of this compound is 26700 grams. -/
theorem compound_weight (molecular_weight : ℝ) (moles : ℝ) : 
  molecular_weight = 2670 → moles = 10 → moles * molecular_weight = 26700 := by
  sorry

end NUMINAMATH_CALUDE_compound_weight_l3598_359843


namespace NUMINAMATH_CALUDE_distribute_5_4_l3598_359895

/-- The number of ways to distribute n indistinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 cousins into 4 rooms -/
theorem distribute_5_4 : distribute 5 4 = 67 := by sorry

end NUMINAMATH_CALUDE_distribute_5_4_l3598_359895


namespace NUMINAMATH_CALUDE_max_x_squared_minus_y_squared_l3598_359808

theorem max_x_squared_minus_y_squared (x y : ℝ) 
  (h : 2 * (x^3 + y^3) = x^2 + y^2) : 
  ∀ a b : ℝ, 2 * (a^3 + b^3) = a^2 + b^2 → x^2 - y^2 ≤ a^2 - b^2 := by
sorry

end NUMINAMATH_CALUDE_max_x_squared_minus_y_squared_l3598_359808


namespace NUMINAMATH_CALUDE_max_non_managers_l3598_359801

/-- The maximum number of non-managers in a department with 9 managers, 
    given that the ratio of managers to non-managers must be greater than 7:37 -/
theorem max_non_managers (managers : ℕ) (non_managers : ℕ) : 
  managers = 9 →
  (managers : ℚ) / non_managers > 7 / 37 →
  non_managers ≤ 47 :=
by sorry

end NUMINAMATH_CALUDE_max_non_managers_l3598_359801


namespace NUMINAMATH_CALUDE_place_value_ratio_l3598_359873

/-- The number we're analyzing -/
def number : ℚ := 25684.2057

/-- The place value of the digit 6 in the number -/
def place_value_6 : ℚ := 1000

/-- The place value of the digit 2 in the number -/
def place_value_2 : ℚ := 0.1

/-- Theorem stating the relationship between the place values -/
theorem place_value_ratio : place_value_6 / place_value_2 = 10000 := by
  sorry

end NUMINAMATH_CALUDE_place_value_ratio_l3598_359873


namespace NUMINAMATH_CALUDE_infinitely_many_non_prime_n4_plus_k_l3598_359850

theorem infinitely_many_non_prime_n4_plus_k :
  ∀ N : ℕ, ∃ k : ℕ, k > N ∧ ∀ n : ℕ, ¬ Prime (n^4 + k) := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_non_prime_n4_plus_k_l3598_359850


namespace NUMINAMATH_CALUDE_final_salt_concentration_l3598_359840

/-- Represents the volume of salt solution in arbitrary units -/
def initialVolume : ℝ := 30

/-- Represents the initial concentration of salt in the solution -/
def initialConcentration : ℝ := 0.15

/-- Represents the volume ratio of the large ball -/
def largeBallRatio : ℝ := 10

/-- Represents the volume ratio of the medium ball -/
def mediumBallRatio : ℝ := 5

/-- Represents the volume ratio of the small ball -/
def smallBallRatio : ℝ := 3

/-- Represents the overflow percentage caused by the small ball -/
def overflowPercentage : ℝ := 0.1

/-- Theorem stating that the final salt concentration is 10% -/
theorem final_salt_concentration :
  let totalOverflow := smallBallRatio + mediumBallRatio + largeBallRatio
  let remainingVolume := initialVolume - totalOverflow
  let initialSaltAmount := initialVolume * initialConcentration
  (initialSaltAmount / initialVolume) * 100 = 10 := by
  sorry


end NUMINAMATH_CALUDE_final_salt_concentration_l3598_359840


namespace NUMINAMATH_CALUDE_original_workers_is_seven_l3598_359806

/-- Represents the work scenario described in the problem -/
structure WorkScenario where
  planned_days : ℕ
  absent_workers : ℕ
  actual_days : ℕ

/-- Calculates the original number of workers given a work scenario -/
def original_workers (scenario : WorkScenario) : ℕ :=
  (scenario.absent_workers * scenario.actual_days) / (scenario.actual_days - scenario.planned_days)

/-- The specific work scenario from the problem -/
def problem_scenario : WorkScenario :=
  { planned_days := 8
  , absent_workers := 3
  , actual_days := 14 }

/-- Theorem stating that the original number of workers in the problem scenario is 7 -/
theorem original_workers_is_seven :
  original_workers problem_scenario = 7 := by
  sorry

end NUMINAMATH_CALUDE_original_workers_is_seven_l3598_359806


namespace NUMINAMATH_CALUDE_abs_diff_of_abs_l3598_359859

theorem abs_diff_of_abs : ∀ a b : ℝ, 
  (abs a = 3 ∧ abs b = 5) → abs (abs (a + b) - abs (a - b)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_of_abs_l3598_359859


namespace NUMINAMATH_CALUDE_solve_equation_l3598_359816

theorem solve_equation : ∀ x : ℝ, 2 * 3 * 4 = 6 * x → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3598_359816


namespace NUMINAMATH_CALUDE_square_tiles_problem_l3598_359838

/-- 
Given a square area tiled with congruent square tiles,
if the total number of tiles on the two diagonals is 25,
then the total number of tiles covering the entire square area is 169.
-/
theorem square_tiles_problem (n : ℕ) : 
  n > 0 → 
  2 * n - 1 = 25 → 
  n ^ 2 = 169 := by
  sorry

end NUMINAMATH_CALUDE_square_tiles_problem_l3598_359838


namespace NUMINAMATH_CALUDE_total_participants_is_260_l3598_359874

/-- Represents the voting scenario for a school dance -/
structure VotingScenario where
  initial_oct22_percent : ℝ
  initial_oct29_percent : ℝ
  additional_votes : ℕ
  final_oct29_percent : ℝ

/-- Calculates the total number of participants in the voting -/
def total_participants (scenario : VotingScenario) : ℕ :=
  sorry

/-- Theorem stating that the total number of participants is 260 -/
theorem total_participants_is_260 (scenario : VotingScenario) 
  (h1 : scenario.initial_oct22_percent = 0.35)
  (h2 : scenario.initial_oct29_percent = 0.65)
  (h3 : scenario.additional_votes = 80)
  (h4 : scenario.final_oct29_percent = 0.45) :
  total_participants scenario = 260 := by
  sorry

end NUMINAMATH_CALUDE_total_participants_is_260_l3598_359874


namespace NUMINAMATH_CALUDE_combine_like_terms_l3598_359862

-- Define the theorem
theorem combine_like_terms (a b : ℝ) : 3 * a^2 * b - 4 * b * a^2 = -a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_combine_like_terms_l3598_359862


namespace NUMINAMATH_CALUDE_solve_for_A_l3598_359879

theorem solve_for_A (x₁ x₂ A : ℂ) : 
  x₁ ≠ x₂ →
  x₁ * (x₁ + 1) = A →
  x₂ * (x₂ + 1) = A →
  x₁^4 + 3*x₁^3 + 5*x₁ = x₂^4 + 3*x₂^3 + 5*x₂ →
  A = -7 := by sorry

end NUMINAMATH_CALUDE_solve_for_A_l3598_359879


namespace NUMINAMATH_CALUDE_smallest_even_sum_fourteen_is_achievable_l3598_359817

def S : Finset Int := {8, -4, 3, 27, 10}

def isValidSum (x y z : Int) : Prop :=
  x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ Even (x + y + z)

theorem smallest_even_sum :
  ∀ x y z, isValidSum x y z → x + y + z ≥ 14 :=
by sorry

theorem fourteen_is_achievable :
  ∃ x y z, isValidSum x y z ∧ x + y + z = 14 :=
by sorry

end NUMINAMATH_CALUDE_smallest_even_sum_fourteen_is_achievable_l3598_359817


namespace NUMINAMATH_CALUDE_robot_walk_distance_l3598_359815

/-- Represents a rectangular field with robot paths -/
structure RobotField where
  length : ℕ
  width : ℕ
  path_width : ℕ
  b_distance : ℕ

/-- Calculates the total distance walked by the robot -/
def total_distance (field : RobotField) : ℕ :=
  let outer_loop := 2 * (field.length + field.width) - 1
  let second_loop := 2 * (field.length - 2 + field.width - 2) - 1
  let third_loop := 2 * (field.length - 4 + field.width - 4) - 1
  let fourth_loop := 2 * (field.length - 6 + field.width - 6) - 1
  let final_segment := field.length - field.path_width - field.b_distance
  outer_loop + second_loop + third_loop + fourth_loop + final_segment

/-- Theorem stating the total distance walked by the robot -/
theorem robot_walk_distance (field : RobotField) 
    (h1 : field.length = 16)
    (h2 : field.width = 8)
    (h3 : field.path_width = 1)
    (h4 : field.b_distance = 1) : 
  total_distance field = 154 := by
  sorry

end NUMINAMATH_CALUDE_robot_walk_distance_l3598_359815


namespace NUMINAMATH_CALUDE_salary_changes_l3598_359802

theorem salary_changes (S : ℝ) (S_pos : S > 0) :
  S * (1 + 0.3) * (1 - 0.2) * (1 + 0.1) * (1 - 0.25) = S * 1.04 := by
  sorry

end NUMINAMATH_CALUDE_salary_changes_l3598_359802


namespace NUMINAMATH_CALUDE_shelf_filling_theorem_l3598_359894

/-- Represents the thickness of a programming book -/
def t : ℝ := sorry

/-- The number of programming books that can fill the shelf -/
def P : ℕ := sorry

/-- The number of biology books in a combination that can fill the shelf -/
def B : ℕ := sorry

/-- The number of physics books in a combination that can fill the shelf -/
def F : ℕ := sorry

/-- The number of programming books in a combination with biology books that can fill the shelf -/
def R : ℕ := sorry

/-- The number of biology books in a combination with programming books that can fill the shelf -/
def C : ℕ := sorry

/-- The number of programming books that alone would fill the shelf (to be proven) -/
def Q : ℕ := sorry

/-- The length of the shelf -/
def shelf_length : ℝ := P * t

/-- Theorem stating that Q equals R + 2C -/
theorem shelf_filling_theorem (h1 : P * t = shelf_length)
                               (h2 : 2 * B * t + 3 * F * t = shelf_length)
                               (h3 : R * t + 2 * C * t = shelf_length)
                               (h4 : Q * t = shelf_length)
                               (h5 : P ≠ B ∧ P ≠ F ∧ P ≠ R ∧ P ≠ C ∧ B ≠ F ∧ B ≠ R ∧ B ≠ C ∧ F ≠ R ∧ F ≠ C ∧ R ≠ C)
                               (h6 : P > 0 ∧ B > 0 ∧ F > 0 ∧ R > 0 ∧ C > 0) :
  Q = R + 2 * C :=
sorry

end NUMINAMATH_CALUDE_shelf_filling_theorem_l3598_359894


namespace NUMINAMATH_CALUDE_caravan_feet_heads_difference_l3598_359854

/-- Represents the number of feet for each animal type -/
def feet_per_animal : Nat → Nat
| 0 => 2  -- Hens
| 1 => 4  -- Goats
| 2 => 4  -- Camels
| 3 => 2  -- Keepers
| _ => 0  -- Other (shouldn't occur)

/-- Calculates the total number of feet for a given animal type and count -/
def total_feet (animal_type : Nat) (count : Nat) : Nat :=
  count * feet_per_animal animal_type

/-- Theorem: In a caravan with 60 hens, 35 goats, 6 camels, and 10 keepers,
    the difference between the total number of feet and the total number of heads is 193 -/
theorem caravan_feet_heads_difference :
  let hens := 60
  let goats := 35
  let camels := 6
  let keepers := 10
  let total_heads := hens + goats + camels + keepers
  let total_feet := total_feet 0 hens + total_feet 1 goats + total_feet 2 camels + total_feet 3 keepers
  total_feet - total_heads = 193 := by
  sorry

end NUMINAMATH_CALUDE_caravan_feet_heads_difference_l3598_359854


namespace NUMINAMATH_CALUDE_six_by_six_grid_shaded_half_l3598_359805

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)
  (shaded_per_row : ℕ)

/-- Calculates the percentage of shaded area in a grid -/
def shaded_percentage (g : Grid) : ℚ :=
  (g.size * g.shaded_per_row : ℚ) / (g.size * g.size)

/-- The main theorem: for a 6x6 grid with 3 shaded squares per row,
    the shaded percentage is 50% -/
theorem six_by_six_grid_shaded_half :
  let g : Grid := { size := 6, shaded_per_row := 3 }
  shaded_percentage g = 1/2 := by sorry

end NUMINAMATH_CALUDE_six_by_six_grid_shaded_half_l3598_359805


namespace NUMINAMATH_CALUDE_line_relationship_l3598_359844

-- Define the concept of lines in 3D space
variable (Line : Type)

-- Define the parallel relationship between lines
variable (parallel : Line → Line → Prop)

-- Define the intersecting relationship between lines
variable (intersecting : Line → Line → Prop)

-- Define the skew relationship between lines
variable (skew : Line → Line → Prop)

-- Define the theorem
theorem line_relationship (a b c : Line) 
  (h1 : parallel a c) 
  (h2 : ¬ parallel b c) : 
  intersecting a b ∨ skew a b :=
sorry

end NUMINAMATH_CALUDE_line_relationship_l3598_359844


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l3598_359841

theorem line_slope_intercept_product (m b : ℚ) : 
  m > 0 → b < 0 → m = 3/4 → b = -2/3 → -1 < m * b ∧ m * b < 0 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l3598_359841


namespace NUMINAMATH_CALUDE_max_value_expression_l3598_359849

theorem max_value_expression (x k : ℕ) (hx : x > 0) (hk : k > 0) : 
  let y := k * x
  ∃ (max : ℚ), max = 2 ∧ ∀ (x' k' : ℕ), x' > 0 → k' > 0 → 
    let y' := k' * x'
    (x' + y')^2 / (x'^2 + y'^2 : ℚ) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l3598_359849


namespace NUMINAMATH_CALUDE_product_121_54_l3598_359842

theorem product_121_54 : 121 * 54 = 6534 := by
  sorry

end NUMINAMATH_CALUDE_product_121_54_l3598_359842


namespace NUMINAMATH_CALUDE_investment_gain_percentage_l3598_359809

/-- Calculate the overall gain percentage for an investment portfolio --/
theorem investment_gain_percentage
  (stock_initial : ℝ)
  (artwork_initial : ℝ)
  (crypto_initial : ℝ)
  (stock_return : ℝ)
  (artwork_return : ℝ)
  (crypto_return_rub : ℝ)
  (rub_to_rs_rate : ℝ)
  (artwork_tax_rate : ℝ)
  (crypto_fee_rate : ℝ)
  (h1 : stock_initial = 5000)
  (h2 : artwork_initial = 10000)
  (h3 : crypto_initial = 15000)
  (h4 : stock_return = 6000)
  (h5 : artwork_return = 12000)
  (h6 : crypto_return_rub = 17000)
  (h7 : rub_to_rs_rate = 1.03)
  (h8 : artwork_tax_rate = 0.05)
  (h9 : crypto_fee_rate = 0.02) :
  let total_initial := stock_initial + artwork_initial + crypto_initial
  let artwork_net_return := artwork_return * (1 - artwork_tax_rate)
  let crypto_return_rs := crypto_return_rub * rub_to_rs_rate
  let crypto_net_return := crypto_return_rs * (1 - crypto_fee_rate)
  let total_return := stock_return + artwork_net_return + crypto_net_return
  let gain_percentage := (total_return - total_initial) / total_initial * 100
  ∃ ε > 0, |gain_percentage - 15.20| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_investment_gain_percentage_l3598_359809


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3598_359811

/-- An isosceles triangle with a median to the leg dividing the perimeter -/
structure IsoscelesTriangleWithMedian where
  /-- Length of the leg of the isosceles triangle -/
  leg : ℝ
  /-- Length of the base of the isosceles triangle -/
  base : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : leg > 0
  /-- The median to the leg divides the perimeter into two parts -/
  medianDivides : leg + leg + base = 27
  /-- One part of the divided perimeter is 15 -/
  part1 : leg + leg / 2 = 15 ∨ leg / 2 + base = 15

/-- The theorem stating the possible base lengths of the isosceles triangle -/
theorem isosceles_triangle_base_length (t : IsoscelesTriangleWithMedian) :
  t.base = 7 ∨ t.base = 11 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3598_359811


namespace NUMINAMATH_CALUDE_empty_boxes_count_l3598_359835

theorem empty_boxes_count (total boxes_with_markers boxes_with_crayons boxes_with_both : ℕ) 
  (h1 : total = 15)
  (h2 : boxes_with_markers = 9)
  (h3 : boxes_with_crayons = 4)
  (h4 : boxes_with_both = 5) :
  total - (boxes_with_markers + boxes_with_crayons - boxes_with_both) = 7 := by
  sorry

end NUMINAMATH_CALUDE_empty_boxes_count_l3598_359835


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3598_359818

/-- Given a principal amount P and an interest rate R (as a percentage),
    if the amount after 2 years is 780 and after 7 years is 1020,
    then the principal amount P is 684. -/
theorem simple_interest_problem (P R : ℚ) : 
  P + (P * R * 2) / 100 = 780 →
  P + (P * R * 7) / 100 = 1020 →
  P = 684 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3598_359818


namespace NUMINAMATH_CALUDE_statement_1_false_statement_2_false_statement_3_false_statement_4_true_l3598_359871

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Line → Prop)
variable (parallelLP : Line → Plane → Prop)
variable (parallelPP : Plane → Plane → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

variable (m n : Line)
variable (α β : Plane)

-- Statement ①
theorem statement_1_false :
  ¬(parallelLP m α → parallelLP n α → parallel m n) :=
sorry

-- Statement ②
theorem statement_2_false :
  ¬(subset m α → subset n α → parallelLP m β → parallelLP n β → parallelPP α β) :=
sorry

-- Statement ③
theorem statement_3_false :
  ¬(perpendicular α β → subset m α → perpendicularLP m β) :=
sorry

-- Statement ④
theorem statement_4_true :
  perpendicular α β → perpendicularLP m β → ¬(subset m α) → parallelLP m α :=
sorry

end NUMINAMATH_CALUDE_statement_1_false_statement_2_false_statement_3_false_statement_4_true_l3598_359871


namespace NUMINAMATH_CALUDE_green_shirt_pairs_l3598_359867

theorem green_shirt_pairs (red_students green_students total_students total_pairs red_red_pairs : ℕ) 
  (h1 : red_students = 63)
  (h2 : green_students = 69)
  (h3 : total_students = red_students + green_students)
  (h4 : total_pairs = 66)
  (h5 : red_red_pairs = 26)
  (h6 : total_students = 2 * total_pairs) :
  green_students - (total_pairs - red_red_pairs - (red_students - 2 * red_red_pairs)) = 29 := by
  sorry

end NUMINAMATH_CALUDE_green_shirt_pairs_l3598_359867


namespace NUMINAMATH_CALUDE_centroid_altitude_distance_l3598_359822

/-- Triangle XYZ with sides a, b, c and centroid G -/
structure Triangle where
  a : ℝ  -- side XY
  b : ℝ  -- side XZ
  c : ℝ  -- side YZ
  G : ℝ × ℝ  -- centroid coordinates

/-- The foot of the altitude from a point to a line segment -/
def altitudeFoot (point : ℝ × ℝ) (segment : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Theorem: In a triangle with sides 12, 15, and 23, the distance from the centroid
    to the foot of the altitude from the centroid to the longest side is 40/23 -/
theorem centroid_altitude_distance (t : Triangle) 
    (h1 : t.a = 12) (h2 : t.b = 15) (h3 : t.c = 23) : 
    let Q := altitudeFoot t.G (⟨0, 0⟩, ⟨t.c, 0⟩)  -- Assuming YZ is on x-axis
    distance t.G Q = 40 / 23 := by
  sorry

end NUMINAMATH_CALUDE_centroid_altitude_distance_l3598_359822


namespace NUMINAMATH_CALUDE_price_increase_theorem_l3598_359856

theorem price_increase_theorem (original_price : ℝ) (original_price_pos : original_price > 0) :
  let first_increase := original_price * 1.2
  let second_increase := first_increase * 1.15
  let total_increase := second_increase - original_price
  (total_increase / original_price) * 100 = 38 := by
sorry

end NUMINAMATH_CALUDE_price_increase_theorem_l3598_359856


namespace NUMINAMATH_CALUDE_february_highest_percentage_difference_l3598_359890

/-- Represents the sales data for a vendor in a given month -/
structure SalesData where
  quantity : Nat
  price : Float

/-- Calculates the revenue from sales data -/
def revenue (data : SalesData) : Float :=
  data.quantity.toFloat * data.price

/-- Calculates the percentage difference between two revenues -/
def percentageDifference (r1 r2 : Float) : Float :=
  (max r1 r2 - min r1 r2) / (min r1 r2) * 100

/-- Represents a month -/
inductive Month
  | January | February | March | April | May | June

/-- Andy's sales data for each month -/
def andySales : Month → SalesData
  | .January => ⟨100, 2⟩
  | .February => ⟨150, 1.5⟩
  | .March => ⟨120, 2.5⟩
  | .April => ⟨80, 4⟩
  | .May => ⟨140, 1.75⟩
  | .June => ⟨110, 3⟩

/-- Bella's sales data for each month -/
def bellaSales : Month → SalesData
  | .January => ⟨90, 2.2⟩
  | .February => ⟨100, 1.75⟩
  | .March => ⟨80, 3⟩
  | .April => ⟨85, 3.5⟩
  | .May => ⟨135, 2⟩
  | .June => ⟨160, 2.5⟩

/-- Theorem: February has the highest percentage difference in revenue -/
theorem february_highest_percentage_difference :
  ∀ m : Month, m ≠ Month.February →
    percentageDifference (revenue (andySales Month.February)) (revenue (bellaSales Month.February)) ≥
    percentageDifference (revenue (andySales m)) (revenue (bellaSales m)) :=
by sorry

end NUMINAMATH_CALUDE_february_highest_percentage_difference_l3598_359890


namespace NUMINAMATH_CALUDE_volume_ratio_of_pyramids_l3598_359898

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a triangular pyramid -/
structure TriangularPyramid where
  apex : Point3D
  base1 : Point3D
  base2 : Point3D
  base3 : Point3D

/-- Calculates the volume of a triangular pyramid -/
def volumeOfTriangularPyramid (pyramid : TriangularPyramid) : ℝ :=
  sorry

/-- Given two points, returns a point that divides the line segment in a given ratio -/
def divideSegment (p1 p2 : Point3D) (ratio : ℝ) : Point3D :=
  sorry

theorem volume_ratio_of_pyramids (P A B C : Point3D) : 
  let PABC := TriangularPyramid.mk P A B C
  let M := divideSegment P C (1/3)
  let N := divideSegment P B (2/3)
  let PAMN := TriangularPyramid.mk P A M N
  (volumeOfTriangularPyramid PAMN) / (volumeOfTriangularPyramid PABC) = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_of_pyramids_l3598_359898


namespace NUMINAMATH_CALUDE_unique_a_with_prime_roots_l3598_359824

theorem unique_a_with_prime_roots : ∃! a : ℕ+, 
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ 
  (2 : ℝ) * p^2 - 30 * p + (a : ℝ) = 0 ∧
  (2 : ℝ) * q^2 - 30 * q + (a : ℝ) = 0 ∧
  a = 52 := by
sorry

end NUMINAMATH_CALUDE_unique_a_with_prime_roots_l3598_359824


namespace NUMINAMATH_CALUDE_power_72_equals_m3n2_l3598_359892

theorem power_72_equals_m3n2 (a m n : ℝ) (h1 : 2^a = m) (h2 : 3^a = n) : 72^a = m^3 * n^2 := by
  sorry

end NUMINAMATH_CALUDE_power_72_equals_m3n2_l3598_359892


namespace NUMINAMATH_CALUDE_right_triangle_in_sets_l3598_359884

/-- Checks if three numbers can form a right-angled triangle --/
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- The given sets of numbers --/
def number_sets : List (ℕ × ℕ × ℕ) :=
  [(1, 2, 3), (2, 3, 4), (3, 4, 5), (9, 13, 17)]

theorem right_triangle_in_sets :
  ∃! (a b c : ℕ), (a, b, c) ∈ number_sets ∧ is_right_triangle a b c :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_in_sets_l3598_359884


namespace NUMINAMATH_CALUDE_base_nine_ones_triangular_l3598_359876

theorem base_nine_ones_triangular (k : ℕ+) : ∃ n : ℕ, (9^k.val - 1) / 8 = n * (n + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_base_nine_ones_triangular_l3598_359876


namespace NUMINAMATH_CALUDE_candy_mixture_problem_l3598_359878

/-- Given two types of candy mixed to produce a specific mixture, 
    prove the amount of the second type of candy. -/
theorem candy_mixture_problem (X Y : ℝ) : 
  X + Y = 10 →
  3.50 * X + 4.30 * Y = 40 →
  Y = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_candy_mixture_problem_l3598_359878


namespace NUMINAMATH_CALUDE_max_knights_and_courtiers_l3598_359837

/-- Represents the number of people at each table -/
structure TableCounts where
  king : ℕ
  courtiers : ℕ
  knights : ℕ

/-- Checks if the table counts are valid according to the problem constraints -/
def is_valid_table_counts (tc : TableCounts) : Prop :=
  tc.king = 7 ∧ 
  12 ≤ tc.courtiers ∧ tc.courtiers ≤ 18 ∧
  10 ≤ tc.knights ∧ tc.knights ≤ 20

/-- The rule that the sum of a knight's portion and a courtier's portion equals the king's portion -/
def satisfies_portion_rule (tc : TableCounts) : Prop :=
  (1 : ℚ) / tc.courtiers + (1 : ℚ) / tc.knights = (1 : ℚ) / tc.king

/-- The main theorem stating the maximum number of knights and corresponding courtiers -/
theorem max_knights_and_courtiers :
  ∃ (tc : TableCounts), 
    is_valid_table_counts tc ∧ 
    satisfies_portion_rule tc ∧
    tc.knights = 14 ∧ 
    tc.courtiers = 14 ∧
    (∀ (tc' : TableCounts), 
      is_valid_table_counts tc' ∧ 
      satisfies_portion_rule tc' → 
      tc'.knights ≤ tc.knights) :=
by sorry

end NUMINAMATH_CALUDE_max_knights_and_courtiers_l3598_359837


namespace NUMINAMATH_CALUDE_problem_solution_l3598_359883

def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem problem_solution :
  let a₁ := 5
  let d := 3
  let aₙ := 38
  let n := (aₙ - a₁) / d + 1
  let a := arithmetic_sum a₁ d n
  let b := sum_of_digits a
  let c := b ^ 2
  let d := c / 3
  d = 75 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3598_359883


namespace NUMINAMATH_CALUDE_min_value_cubic_expression_l3598_359897

theorem min_value_cubic_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 15 * x - y = 22) :
  x^3 + y^3 - x^2 - y^2 ≥ 1 ∧
  (x^3 + y^3 - x^2 - y^2 = 1 ↔ x = 3/2 ∧ y = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_cubic_expression_l3598_359897


namespace NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l3598_359863

theorem integer_pairs_satisfying_equation : 
  {(x, y) : ℤ × ℤ | x * (x + 1) * (x + 2) * (x + 3) = y * (y + 1)} = 
  {(0, 0), (-1, 0), (-2, 0), (-3, 0), (0, -1), (-1, -1), (-2, -1), (-3, -1)} := by
  sorry

end NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l3598_359863


namespace NUMINAMATH_CALUDE_remainder_N_mod_45_l3598_359848

def N : ℕ := sorry

theorem remainder_N_mod_45 : N % 45 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_N_mod_45_l3598_359848


namespace NUMINAMATH_CALUDE_algebraic_expression_correct_l3598_359886

/-- The algebraic expression for "three times x minus the cube of y" -/
def algebraic_expression (x y : ℝ) : ℝ := 3 * x - y^3

/-- Theorem stating that the algebraic expression is correct -/
theorem algebraic_expression_correct (x y : ℝ) : 
  algebraic_expression x y = 3 * x - y^3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_correct_l3598_359886


namespace NUMINAMATH_CALUDE_pasha_wins_l3598_359875

/-- Represents the game state -/
structure GameState where
  n : ℕ  -- Number of tokens
  k : ℕ  -- Game parameter

/-- Represents a move in the game -/
inductive Move
  | pasha : Move
  | roma : Move

/-- Represents the result of the game -/
inductive GameResult
  | pashaWins : GameResult
  | romaWins : GameResult

/-- The game progression function -/
def playGame (state : GameState) (strategy : GameState → Move) : GameResult :=
  sorry

/-- Pasha's winning strategy -/
def pashaStrategy (state : GameState) : Move :=
  sorry

/-- Theorem stating that Pasha can ensure at least one token reaches the end -/
theorem pasha_wins (n k : ℕ) (h : n > k * 2^k) :
  ∃ (strategy : GameState → Move),
    playGame ⟨n, k⟩ strategy = GameResult.pashaWins :=
  sorry

end NUMINAMATH_CALUDE_pasha_wins_l3598_359875


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3598_359858

theorem inequality_solution_set (a : ℝ) (h : a < 0) :
  {x : ℝ | a * x - 1 > 0} = {x : ℝ | x < 1 / a} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3598_359858


namespace NUMINAMATH_CALUDE_sixteenth_selected_student_number_l3598_359821

/-- Represents a systematic sampling scheme. -/
structure SystematicSampling where
  totalStudents : ℕ
  numGroups : ℕ
  interval : ℕ
  firstSelected : ℕ

/-- Calculates the number of the nth selected student in a systematic sampling. -/
def nthSelectedStudent (s : SystematicSampling) (n : ℕ) : ℕ :=
  s.firstSelected + (n - 1) * s.interval

theorem sixteenth_selected_student_number
  (s : SystematicSampling)
  (h1 : s.totalStudents = 800)
  (h2 : s.numGroups = 50)
  (h3 : s.interval = s.totalStudents / s.numGroups)
  (h4 : nthSelectedStudent s 3 = 36) :
  nthSelectedStudent s 16 = 244 := by
  sorry

end NUMINAMATH_CALUDE_sixteenth_selected_student_number_l3598_359821


namespace NUMINAMATH_CALUDE_chord_intersection_diameter_segments_l3598_359839

theorem chord_intersection_diameter_segments (r : ℝ) (chord_length : ℝ) : 
  r = 6 → chord_length = 10 → ∃ (s₁ s₂ : ℝ), s₁ = 6 - Real.sqrt 11 ∧ s₂ = 6 + Real.sqrt 11 ∧ s₁ + s₂ = 2 * r :=
by sorry

end NUMINAMATH_CALUDE_chord_intersection_diameter_segments_l3598_359839


namespace NUMINAMATH_CALUDE_f_g_derivatives_neg_l3598_359846

-- Define f and g as functions from ℝ to ℝ
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
variable (hf : ∀ x, f (-x) = -f x)
variable (hg : ∀ x, g (-x) = g x)

-- Define the derivative properties for x > 0
variable (hf_deriv_pos : ∀ x, x > 0 → deriv f x > 0)
variable (hg_deriv_pos : ∀ x, x > 0 → deriv g x > 0)

-- State the theorem
theorem f_g_derivatives_neg (x : ℝ) (hx : x < 0) : 
  deriv f x > 0 ∧ deriv g x < 0 :=
sorry

end NUMINAMATH_CALUDE_f_g_derivatives_neg_l3598_359846


namespace NUMINAMATH_CALUDE_product_of_repeating_decimals_division_of_repeating_decimals_l3598_359833

-- Define the repeating decimals
def repeating_decimal_18 : ℚ := 2 / 11
def repeating_decimal_36 : ℚ := 4 / 11

-- Theorem for the product
theorem product_of_repeating_decimals :
  repeating_decimal_18 * repeating_decimal_36 = 8 / 121 := by
  sorry

-- Theorem for the division
theorem division_of_repeating_decimals :
  repeating_decimal_18 / repeating_decimal_36 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimals_division_of_repeating_decimals_l3598_359833


namespace NUMINAMATH_CALUDE_problem_solution_l3598_359870

/-- The base-74 representation of the number in the problem -/
def base_74_num : ℕ := 235935623

/-- Converts the base-74 number to its decimal equivalent modulo 15 -/
def decimal_mod_15 : ℕ := base_74_num % 15

theorem problem_solution (a : ℤ) (h1 : 0 ≤ a) (h2 : a ≤ 14) 
  (h3 : (decimal_mod_15 - a) % 15 = 0) : a = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3598_359870


namespace NUMINAMATH_CALUDE_john_uber_profit_l3598_359869

def uber_earnings : ℕ := 30000
def car_purchase_price : ℕ := 18000
def car_trade_in_value : ℕ := 6000

theorem john_uber_profit :
  uber_earnings - (car_purchase_price - car_trade_in_value) = 18000 :=
by sorry

end NUMINAMATH_CALUDE_john_uber_profit_l3598_359869


namespace NUMINAMATH_CALUDE_regular_toenails_in_jar_l3598_359893

def jar_capacity : ℕ := 100
def big_toenail_count : ℕ := 20
def remaining_space : ℕ := 20

def big_toenail_size : ℕ := 2
def regular_toenail_size : ℕ := 1

theorem regular_toenails_in_jar :
  ∃ (regular_count : ℕ),
    regular_count * regular_toenail_size +
    big_toenail_count * big_toenail_size +
    remaining_space * regular_toenail_size = jar_capacity ∧
    regular_count = 40 := by
  sorry

end NUMINAMATH_CALUDE_regular_toenails_in_jar_l3598_359893


namespace NUMINAMATH_CALUDE_range_of_2a_plus_3b_l3598_359896

theorem range_of_2a_plus_3b (a b : ℝ) 
  (h1 : -1 ≤ a + b ∧ a + b ≤ 1) 
  (h2 : -1 ≤ a - b ∧ a - b ≤ 1) : 
  (∀ x, 2*a + 3*b ≤ x → 3 ≤ x) ∧ 
  (∀ y, y ≤ 2*a + 3*b → y ≤ -3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_2a_plus_3b_l3598_359896


namespace NUMINAMATH_CALUDE_merchant_profit_l3598_359845

theorem merchant_profit (C S : ℝ) (h : 17 * C = 16 * S) :
  (S - C) / C * 100 = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_merchant_profit_l3598_359845


namespace NUMINAMATH_CALUDE_expression_equality_l3598_359807

theorem expression_equality : 
  (2011^2 * 2012 - 2013) / Nat.factorial 2012 + 
  (2013^2 * 2014 - 2015) / Nat.factorial 2014 = 
  1 / Nat.factorial 2009 + 1 / Nat.factorial 2010 - 
  1 / Nat.factorial 2013 - 1 / Nat.factorial 2014 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3598_359807


namespace NUMINAMATH_CALUDE_set_intersection_example_l3598_359847

theorem set_intersection_example : 
  let A : Set ℕ := {1, 2, 4}
  let B : Set ℕ := {2, 4, 6}
  A ∩ B = {2, 4} := by
sorry

end NUMINAMATH_CALUDE_set_intersection_example_l3598_359847


namespace NUMINAMATH_CALUDE_rectangle_dimensions_quadratic_equation_l3598_359888

theorem rectangle_dimensions_quadratic_equation 
  (L W : ℝ) 
  (h1 : L + W = 15) 
  (h2 : L * W = 2 * W^2) : 
  (L = (15 + Real.sqrt 25) / 2 ∧ W = (15 - Real.sqrt 25) / 2) ∨ 
  (L = (15 - Real.sqrt 25) / 2 ∧ W = (15 + Real.sqrt 25) / 2) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_quadratic_equation_l3598_359888


namespace NUMINAMATH_CALUDE_caleb_spent_66_50_l3598_359861

/-- The total amount spent on hamburgers -/
def total_spent (total_burgers : ℕ) (single_cost double_cost : ℚ) (double_count : ℕ) : ℚ :=
  let single_count := total_burgers - double_count
  double_count * double_cost + single_count * single_cost

/-- Theorem stating that Caleb spent $66.50 on hamburgers -/
theorem caleb_spent_66_50 :
  total_spent 50 1 (3/2) 33 = 133/2 := by
  sorry

end NUMINAMATH_CALUDE_caleb_spent_66_50_l3598_359861
