import Mathlib

namespace NUMINAMATH_CALUDE_square_on_hypotenuse_l618_61891

theorem square_on_hypotenuse (a b : ℝ) (ha : a = 9) (hb : b = 12) :
  let c := Real.sqrt (a^2 + b^2)
  let s := (a * b * c) / (a^2 + b^2)
  s = 45 / 8 := by sorry

end NUMINAMATH_CALUDE_square_on_hypotenuse_l618_61891


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l618_61894

/-- Given two vectors in ℝ², prove that the magnitude of their difference is 5 -/
theorem vector_difference_magnitude (a b : ℝ × ℝ) : 
  a = (2, 1) → b = (-2, 4) → ‖a - b‖ = 5 := by sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l618_61894


namespace NUMINAMATH_CALUDE_percentage_passed_all_subjects_l618_61822

/-- Percentage of students who failed in Hindi -/
def A : ℝ := 30

/-- Percentage of students who failed in English -/
def B : ℝ := 45

/-- Percentage of students who failed in Math -/
def C : ℝ := 25

/-- Percentage of students who failed in Science -/
def D : ℝ := 40

/-- Percentage of students who failed in both Hindi and English -/
def AB : ℝ := 12

/-- Percentage of students who failed in both Hindi and Math -/
def AC : ℝ := 15

/-- Percentage of students who failed in both Hindi and Science -/
def AD : ℝ := 18

/-- Percentage of students who failed in both English and Math -/
def BC : ℝ := 20

/-- Percentage of students who failed in both English and Science -/
def BD : ℝ := 22

/-- Percentage of students who failed in both Math and Science -/
def CD : ℝ := 24

/-- Percentage of students who failed in all four subjects -/
def ABCD : ℝ := 10

/-- The total percentage -/
def total : ℝ := 100

theorem percentage_passed_all_subjects :
  total - (A + B + C + D - (AB + AC + AD + BC + BD + CD) + ABCD) = 61 := by
  sorry

end NUMINAMATH_CALUDE_percentage_passed_all_subjects_l618_61822


namespace NUMINAMATH_CALUDE_last_digit_sum_l618_61853

theorem last_digit_sum (n : ℕ) : 
  (2^2 + 20^20 + 200^200 + 2006^2006) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_sum_l618_61853


namespace NUMINAMATH_CALUDE_original_number_is_35_l618_61884

-- Define a two-digit number type
def TwoDigitNumber := { n : ℕ // n ≥ 10 ∧ n < 100 }

-- Define functions to get tens and units digits
def tens_digit (n : TwoDigitNumber) : ℕ := n.val / 10
def units_digit (n : TwoDigitNumber) : ℕ := n.val % 10

-- Define a function to swap digits
def swap_digits (n : TwoDigitNumber) : TwoDigitNumber :=
  ⟨10 * (units_digit n) + (tens_digit n), by sorry⟩

-- Theorem statement
theorem original_number_is_35 (n : TwoDigitNumber) 
  (h1 : tens_digit n + units_digit n = 8)
  (h2 : (swap_digits n).val = n.val + 18) : 
  n.val = 35 := by sorry

end NUMINAMATH_CALUDE_original_number_is_35_l618_61884


namespace NUMINAMATH_CALUDE_addison_sunday_ticket_sales_l618_61808

/-- Proves that Addison sold 78 raffle tickets on Sunday given the conditions of the problem -/
theorem addison_sunday_ticket_sales : 
  ∀ (friday saturday sunday : ℕ),
  friday = 181 →
  saturday = 2 * friday →
  saturday = sunday + 284 →
  sunday = 78 := by sorry

end NUMINAMATH_CALUDE_addison_sunday_ticket_sales_l618_61808


namespace NUMINAMATH_CALUDE_steves_speed_steves_speed_proof_l618_61882

/-- Calculates Steve's speed during John's final push in a race --/
theorem steves_speed (john_initial_behind : ℝ) (john_speed : ℝ) (john_time : ℝ) (john_final_ahead : ℝ) : ℝ :=
  let john_distance := john_speed * john_time
  let steve_distance := john_distance - (john_initial_behind + john_final_ahead)
  steve_distance / john_time

/-- Proves that Steve's speed during John's final push was 3.8 m/s --/
theorem steves_speed_proof :
  steves_speed 15 4.2 42.5 2 = 3.8 := by
  sorry

end NUMINAMATH_CALUDE_steves_speed_steves_speed_proof_l618_61882


namespace NUMINAMATH_CALUDE_response_change_difference_l618_61865

theorem response_change_difference (initial_yes initial_no final_yes final_no : ℚ) :
  initial_yes = 40/100 →
  initial_no = 60/100 →
  final_yes = 80/100 →
  final_no = 20/100 →
  initial_yes + initial_no = 1 →
  final_yes + final_no = 1 →
  ∃ (min_change max_change : ℚ),
    (∀ (change : ℚ), change ≥ min_change ∧ change ≤ max_change) ∧
    max_change - min_change = 20/100 :=
by sorry

end NUMINAMATH_CALUDE_response_change_difference_l618_61865


namespace NUMINAMATH_CALUDE_area_between_circles_l618_61866

/-- The area between two externally tangent circles and their circumscribing circle -/
theorem area_between_circles (r₁ r₂ : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 5) : 
  let R := (r₁ + r₂) / 2
  π * R^2 - π * r₁^2 - π * r₂^2 = 40 * π := by sorry

end NUMINAMATH_CALUDE_area_between_circles_l618_61866


namespace NUMINAMATH_CALUDE_inequality_proof_l618_61875

theorem inequality_proof :
  (∀ m n p : ℝ, m > n ∧ n > 0 ∧ p > 0 → n / m < (n + p) / (m + p)) ∧
  (∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b →
    c / (a + b) + a / (b + c) + b / (c + a) < 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l618_61875


namespace NUMINAMATH_CALUDE_relay_race_total_time_l618_61850

/-- The total time for a relay race with four athletes -/
def relay_race_time (athlete1_time athlete2_time athlete3_time athlete4_time : ℕ) : ℕ :=
  athlete1_time + athlete2_time + athlete3_time + athlete4_time

/-- Theorem stating the total time for the relay race is 200 seconds -/
theorem relay_race_total_time : 
  ∀ (athlete1_time : ℕ),
    athlete1_time = 55 →
    ∀ (athlete2_time : ℕ),
      athlete2_time = athlete1_time + 10 →
      ∀ (athlete3_time : ℕ),
        athlete3_time = athlete2_time - 15 →
        ∀ (athlete4_time : ℕ),
          athlete4_time = athlete1_time - 25 →
          relay_race_time athlete1_time athlete2_time athlete3_time athlete4_time = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_relay_race_total_time_l618_61850


namespace NUMINAMATH_CALUDE_problem_1_l618_61836

theorem problem_1 : (1/3 - 3/4 + 5/6) / (1/12) = 5 := by sorry

end NUMINAMATH_CALUDE_problem_1_l618_61836


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l618_61843

/-- An isosceles triangle with perimeter 10 and one side length 2 -/
structure IsoscelesTriangle where
  perimeter : ℝ
  side_length : ℝ
  perimeter_eq : perimeter = 10
  side_length_eq : side_length = 2

/-- The base length of the isosceles triangle -/
def base_length (t : IsoscelesTriangle) : ℝ := 4

theorem isosceles_triangle_base_length (t : IsoscelesTriangle) :
  base_length t = 4 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l618_61843


namespace NUMINAMATH_CALUDE_gcd_bound_for_special_numbers_l618_61844

/-- Given two 2019-digit numbers a and b with specific non-zero digit patterns,
    prove that their greatest common divisor has at most 14 digits. -/
theorem gcd_bound_for_special_numbers (a b : ℕ) : 
  (∃ A B C D : ℕ,
    a = A * 10^2014 + B ∧ 
    b = C * 10^2014 + D ∧
    10^4 < A ∧ A < 10^5 ∧
    10^6 < B ∧ B < 10^7 ∧
    10^4 < C ∧ C < 10^5 ∧
    10^8 < D ∧ D < 10^9) →
  Nat.gcd a b < 10^14 :=
sorry

end NUMINAMATH_CALUDE_gcd_bound_for_special_numbers_l618_61844


namespace NUMINAMATH_CALUDE_constant_e_value_l618_61810

theorem constant_e_value (x y e : ℝ) 
  (h1 : x / (2 * y) = 3 / e) 
  (h2 : (7 * x + 4 * y) / (x - 2 * y) = 25) : 
  e = 2 := by sorry

end NUMINAMATH_CALUDE_constant_e_value_l618_61810


namespace NUMINAMATH_CALUDE_sphere_volume_l618_61898

theorem sphere_volume (r : ℝ) (h : 4 * π * r^2 = 2 * Real.sqrt 3 * π * (2 * r)) :
  (4 / 3) * π * r^3 = 4 * Real.sqrt 3 * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_l618_61898


namespace NUMINAMATH_CALUDE_chocolate_distribution_l618_61803

/-- Represents the number of bags of chocolates initially bought by Robie -/
def initial_bags : ℕ := 3

/-- Represents the number of pieces of chocolate in each bag -/
def pieces_per_bag : ℕ := 30

/-- Represents the number of bags given to siblings -/
def bags_to_siblings : ℕ := 2

/-- Represents the number of Robie's siblings -/
def num_siblings : ℕ := 4

/-- Represents the percentage of chocolates received by the oldest sibling -/
def oldest_sibling_share : ℚ := 40 / 100

/-- Represents the percentage of chocolates received by the second oldest sibling -/
def second_oldest_sibling_share : ℚ := 30 / 100

/-- Represents the percentage of chocolates shared by the last two siblings -/
def youngest_siblings_share : ℚ := 30 / 100

/-- Represents the number of additional bags bought by Robie -/
def additional_bags : ℕ := 3

/-- Represents the discount percentage on the third additional bag -/
def discount_percentage : ℚ := 50 / 100

/-- Represents the cost of each non-discounted bag in dollars -/
def cost_per_bag : ℕ := 12

/-- Theorem stating the total amount spent, Robie's remaining chocolates, and siblings' remaining chocolates -/
theorem chocolate_distribution :
  let total_spent := initial_bags * cost_per_bag + 
                     (additional_bags - 1) * cost_per_bag + 
                     (1 - discount_percentage) * cost_per_bag
  let robie_remaining := (initial_bags - bags_to_siblings) * pieces_per_bag + 
                         additional_bags * pieces_per_bag
  let sibling_remaining := 0
  (total_spent = 66 ∧ robie_remaining = 90 ∧ sibling_remaining = 0) := by sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l618_61803


namespace NUMINAMATH_CALUDE_reggie_free_throws_l618_61824

/-- Represents the number of points for each type of shot --/
structure PointValues where
  layup : ℕ
  freeThrow : ℕ
  longShot : ℕ

/-- Represents the shots made by a player --/
structure ShotsMade where
  layups : ℕ
  freeThrows : ℕ
  longShots : ℕ

/-- Calculates the total points scored by a player --/
def calculatePoints (pv : PointValues) (sm : ShotsMade) : ℕ :=
  pv.layup * sm.layups + pv.freeThrow * sm.freeThrows + pv.longShot * sm.longShots

theorem reggie_free_throws 
  (pointValues : PointValues)
  (reggieShotsMade : ShotsMade)
  (brotherShotsMade : ShotsMade)
  (h1 : pointValues.layup = 1)
  (h2 : pointValues.freeThrow = 2)
  (h3 : pointValues.longShot = 3)
  (h4 : reggieShotsMade.layups = 3)
  (h5 : reggieShotsMade.longShots = 1)
  (h6 : brotherShotsMade.layups = 0)
  (h7 : brotherShotsMade.freeThrows = 0)
  (h8 : brotherShotsMade.longShots = 4)
  (h9 : calculatePoints pointValues brotherShotsMade = calculatePoints pointValues reggieShotsMade + 2) :
  reggieShotsMade.freeThrows = 2 := by
  sorry

#check reggie_free_throws

end NUMINAMATH_CALUDE_reggie_free_throws_l618_61824


namespace NUMINAMATH_CALUDE_column_addition_sum_l618_61895

theorem column_addition_sum : ∀ (w x y z : ℕ),
  w ≤ 9 ∧ x ≤ 9 ∧ y ≤ 9 ∧ z ≤ 9 →  -- digits are between 0 and 9
  w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →  -- digits are distinct
  y + w = 10 →  -- rightmost column
  x + y + 1 = 10 →  -- middle column
  w + z + 1 = 11 →  -- leftmost column
  w + x + y + z = 20 :=
by sorry

end NUMINAMATH_CALUDE_column_addition_sum_l618_61895


namespace NUMINAMATH_CALUDE_screen_area_difference_l618_61825

theorem screen_area_difference :
  let square_area (diagonal : ℝ) := diagonal^2 / 2
  (square_area 19 - square_area 17) = 36 := by sorry

end NUMINAMATH_CALUDE_screen_area_difference_l618_61825


namespace NUMINAMATH_CALUDE_subset_condition_1_subset_condition_2_l618_61863

-- Define the sets A and B
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 7}
def B (a : ℝ) : Set ℝ := {x | 3 - 2*a ≤ x ∧ x ≤ 2*a - 5}

-- Theorem for part 1
theorem subset_condition_1 (a : ℝ) : A ⊆ B a ↔ a ≥ 6 := by sorry

-- Theorem for part 2
theorem subset_condition_2 (a : ℝ) : B a ⊆ A ↔ 2 ≤ a ∧ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_subset_condition_1_subset_condition_2_l618_61863


namespace NUMINAMATH_CALUDE_cost_of_450_candies_l618_61828

/-- The cost of buying a specific number of chocolate candies -/
def cost_of_candies (candies_per_box : ℕ) (cost_per_box : ℚ) (total_candies : ℕ) : ℚ :=
  (total_candies / candies_per_box) * cost_per_box

/-- Theorem stating the cost of 450 chocolate candies -/
theorem cost_of_450_candies :
  cost_of_candies 30 (7.5 : ℚ) 450 = 112.5 := by
  sorry

#eval cost_of_candies 30 (7.5 : ℚ) 450

end NUMINAMATH_CALUDE_cost_of_450_candies_l618_61828


namespace NUMINAMATH_CALUDE_square_area_error_l618_61858

theorem square_area_error (S : ℝ) (S' : ℝ) (A : ℝ) (A' : ℝ) : 
  S > 0 →
  S' = S * 1.04 →
  A = S^2 →
  A' = S'^2 →
  (A' - A) / A * 100 = 8.16 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l618_61858


namespace NUMINAMATH_CALUDE_lucas_easter_eggs_problem_l618_61887

theorem lucas_easter_eggs_problem (blue_eggs green_eggs min_eggs : ℕ) 
  (h1 : blue_eggs = 30)
  (h2 : green_eggs = 42)
  (h3 : min_eggs = 5) :
  ∃ (basket_eggs : ℕ), 
    basket_eggs ≥ min_eggs ∧ 
    basket_eggs ∣ blue_eggs ∧ 
    basket_eggs ∣ green_eggs ∧
    ∀ (n : ℕ), n > basket_eggs → ¬(n ∣ blue_eggs ∧ n ∣ green_eggs) :=
by sorry

end NUMINAMATH_CALUDE_lucas_easter_eggs_problem_l618_61887


namespace NUMINAMATH_CALUDE_jane_yellow_sheets_l618_61877

/-- The number of old, yellow sheets of drawing paper Jane has -/
def yellowSheets (totalSheets brownSheets : ℕ) : ℕ :=
  totalSheets - brownSheets

theorem jane_yellow_sheets : 
  let totalSheets : ℕ := 55
  let brownSheets : ℕ := 28
  yellowSheets totalSheets brownSheets = 27 := by
  sorry

end NUMINAMATH_CALUDE_jane_yellow_sheets_l618_61877


namespace NUMINAMATH_CALUDE_negation_p_false_necessary_not_sufficient_l618_61838

theorem negation_p_false_necessary_not_sufficient (p q : Prop) :
  (∃ (h : p ∧ q), ¬¬p) ∧ 
  (∃ (h : ¬¬p), ¬(p ∧ q)) := by
sorry

end NUMINAMATH_CALUDE_negation_p_false_necessary_not_sufficient_l618_61838


namespace NUMINAMATH_CALUDE_family_income_proof_l618_61814

/-- Proves that the initial average monthly income of a family is 840 given the conditions --/
theorem family_income_proof (initial_members : ℕ) (deceased_income new_average : ℚ) :
  initial_members = 4 →
  deceased_income = 1410 →
  new_average = 650 →
  (initial_members : ℚ) * (initial_members * new_average + deceased_income) / initial_members = 840 :=
by sorry

end NUMINAMATH_CALUDE_family_income_proof_l618_61814


namespace NUMINAMATH_CALUDE_ace_diamond_king_probability_l618_61861

-- Define the structure of a standard deck
def StandardDeck : Type := Fin 52

-- Define the properties of cards
def isAce : StandardDeck → Prop := sorry
def isDiamond : StandardDeck → Prop := sorry
def isKing : StandardDeck → Prop := sorry

-- Define the draw function
def draw : ℕ → (StandardDeck → Prop) → ℚ := sorry

-- Theorem statement
theorem ace_diamond_king_probability :
  draw 1 isAce * draw 2 isDiamond * draw 3 isKing = 1 / 663 := by sorry

end NUMINAMATH_CALUDE_ace_diamond_king_probability_l618_61861


namespace NUMINAMATH_CALUDE_g_expression_l618_61801

theorem g_expression (x : ℝ) (g : ℝ → ℝ) :
  (2 * x^5 + 4 * x^3 - 3 * x + g x = 7 * x^3 + 5 * x - 2) →
  (g x = -2 * x^5 + 3 * x^3 + 8 * x - 2) := by
  sorry

end NUMINAMATH_CALUDE_g_expression_l618_61801


namespace NUMINAMATH_CALUDE_northern_village_conscription_l618_61880

/-- The number of people to be conscripted from the northern village -/
def northern_conscription (total_population : ℕ) (northern_population : ℕ) (total_conscription : ℕ) : ℕ :=
  (northern_population * total_conscription) / total_population

theorem northern_village_conscription :
  northern_conscription 22500 8100 300 = 108 := by
sorry

end NUMINAMATH_CALUDE_northern_village_conscription_l618_61880


namespace NUMINAMATH_CALUDE_f_minimum_and_inequality_l618_61879

noncomputable def f (x : ℝ) : ℝ := Real.exp (x - 1) - Real.log x

theorem f_minimum_and_inequality :
  (∀ x > 0, f x ≥ 1) ∧
  (∃ x > 0, f x = 1) ∧
  (∀ x > 0, x * (Real.exp x) * f x + (x * Real.exp x - 1) * Real.log x - Real.exp x + 1/2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_f_minimum_and_inequality_l618_61879


namespace NUMINAMATH_CALUDE_baseball_average_runs_l618_61832

/-- Represents the scoring pattern of a baseball team over a series of games -/
structure ScoringPattern where
  games : ℕ
  oneRun : ℕ
  fourRuns : ℕ
  fiveRuns : ℕ

/-- Calculates the average runs per game given a scoring pattern -/
def averageRuns (pattern : ScoringPattern) : ℚ :=
  (pattern.oneRun * 1 + pattern.fourRuns * 4 + pattern.fiveRuns * 5) / pattern.games

/-- Theorem stating that for the given scoring pattern, the average runs per game is 4 -/
theorem baseball_average_runs :
  let pattern : ScoringPattern := {
    games := 6,
    oneRun := 1,
    fourRuns := 2,
    fiveRuns := 3
  }
  averageRuns pattern = 4 := by sorry

end NUMINAMATH_CALUDE_baseball_average_runs_l618_61832


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_to_83_l618_61868

theorem last_three_digits_of_7_to_83 :
  7^83 ≡ 886 [ZMOD 1000] := by sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_to_83_l618_61868


namespace NUMINAMATH_CALUDE_min_ratio_T2_T1_l618_61849

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle in 2D space -/
structure Triangle :=
  (A B C : Point)

/-- Checks if a triangle is acute -/
def is_acute (t : Triangle) : Prop := sorry

/-- Calculates the area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- Represents the altitude of a triangle -/
structure Altitude :=
  (base : Point) (foot : Point)

/-- Calculates the projection of a point onto a line -/
def project (p : Point) (l : Point × Point) : Point := sorry

/-- Calculates the area of T_1 as defined in the problem -/
def area_T1 (t : Triangle) (AD BE CF : Altitude) : ℝ := sorry

/-- Calculates the area of T_2 as defined in the problem -/
def area_T2 (t : Triangle) (AD BE CF : Altitude) : ℝ := sorry

/-- The main theorem: The ratio T_2/T_1 is always greater than or equal to 25 for any acute triangle -/
theorem min_ratio_T2_T1 (t : Triangle) (AD BE CF : Altitude) :
  is_acute t →
  (area_T2 t AD BE CF) / (area_T1 t AD BE CF) ≥ 25 := by
  sorry

end NUMINAMATH_CALUDE_min_ratio_T2_T1_l618_61849


namespace NUMINAMATH_CALUDE_fraction_inequality_l618_61888

theorem fraction_inequality (a b c d p q : ℕ+) 
  (h1 : a * d - b * c = 1)
  (h2 : (a : ℚ) / b > (p : ℚ) / q)
  (h3 : (p : ℚ) / q > (c : ℚ) / d) : 
  q ≥ b + d ∧ (q = b + d → p = a + c) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l618_61888


namespace NUMINAMATH_CALUDE_roof_length_width_difference_l618_61809

/-- Given a rectangular roof with length 4 times its width and an area of 900 square feet,
    prove that the difference between the length and width is 24√5 feet. -/
theorem roof_length_width_difference (w : ℝ) (h1 : w > 0) (h2 : 5 * w * w = 900) :
  5 * w - w = 24 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_roof_length_width_difference_l618_61809


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l618_61821

/-- Given arithmetic sequences a and b with sums S and T respectively, 
    if S_n / T_n = (2n-1) / (3n+2) for all n, then a_7 / b_7 = 25 / 41 -/
theorem arithmetic_sequence_ratio 
  (a b : ℕ → ℚ) 
  (S T : ℕ → ℚ) 
  (h1 : ∀ n, S n = (n / 2) * (a 1 + a n)) 
  (h2 : ∀ n, T n = (n / 2) * (b 1 + b n)) 
  (h3 : ∀ n, S n / T n = (2 * n - 1) / (3 * n + 2)) : 
  a 7 / b 7 = 25 / 41 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l618_61821


namespace NUMINAMATH_CALUDE_planes_parallel_l618_61833

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (non_coincident : Line → Line → Prop)
variable (plane_non_coincident : Plane → Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel
  (l m : Line) (α β γ : Plane)
  (h1 : non_coincident l m)
  (h2 : plane_non_coincident α β γ)
  (h3 : perpendicular l α)
  (h4 : perpendicular m β)
  (h5 : parallel l m) :
  plane_parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_l618_61833


namespace NUMINAMATH_CALUDE_circular_pond_area_l618_61845

theorem circular_pond_area (AB CD : ℝ) (h1 : AB = 20) (h2 : CD = 12) : 
  let R := CD
  let A := π * R^2
  A = 244 * π := by sorry

end NUMINAMATH_CALUDE_circular_pond_area_l618_61845


namespace NUMINAMATH_CALUDE_smallest_integer_for_quadratic_inequality_l618_61860

theorem smallest_integer_for_quadratic_inequality :
  ∃ n : ℤ, (∀ m : ℤ, m^2 - 13*m + 40 ≤ 0 → n ≤ m) ∧ (n^2 - 13*n + 40 ≤ 0) ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_for_quadratic_inequality_l618_61860


namespace NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l618_61834

theorem smallest_part_of_proportional_division (total : ℝ) (a b c d : ℝ) 
  (h_total : total = 80)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (h_prop : b = 3 * a ∧ c = 5 * a ∧ d = 7 * a)
  (h_sum : a + b + c + d = total) :
  a = 5 := by
sorry

end NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l618_61834


namespace NUMINAMATH_CALUDE_field_area_l618_61871

/-- The area of a rectangular field with specific fencing conditions -/
theorem field_area (L W : ℝ) : 
  L = 20 →  -- One side is 20 feet
  2 * W + L = 41 →  -- Total fencing is 41 feet
  L * W = 210 :=  -- Area of the field
by
  sorry

end NUMINAMATH_CALUDE_field_area_l618_61871


namespace NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l618_61885

-- Define sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

-- Theorem 1: B ⊆ A iff m ∈ (-∞, 3]
theorem subset_condition (m : ℝ) : B m ⊆ A ↔ m ≤ 3 := by sorry

-- Theorem 2: A ∩ B = ∅ iff m ∈ (-∞, 2) ∪ (4, +∞)
theorem disjoint_condition (m : ℝ) : A ∩ B m = ∅ ↔ m < 2 ∨ m > 4 := by sorry

end NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l618_61885


namespace NUMINAMATH_CALUDE_angle_between_vectors_l618_61869

/-- The angle between two vectors in R² -/
def angle (v w : ℝ × ℝ) : ℝ := sorry

theorem angle_between_vectors (a b : ℝ × ℝ) (h1 : a = (1, 1)) (h2 : b = (-1, 2)) :
  angle (a + b) a = π / 4 := by sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l618_61869


namespace NUMINAMATH_CALUDE_scientific_notation_180_million_l618_61839

/-- Proves that 180 million in scientific notation is equal to 1.8 × 10^8 -/
theorem scientific_notation_180_million :
  (180000000 : ℝ) = 1.8 * (10 ^ 8) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_180_million_l618_61839


namespace NUMINAMATH_CALUDE_undeveloped_sections_l618_61830

/-- Proves that the number of undeveloped sections is 3 given the specified conditions -/
theorem undeveloped_sections
  (section_area : ℝ)
  (total_undeveloped_area : ℝ)
  (h1 : section_area = 2435)
  (h2 : total_undeveloped_area = 7305) :
  total_undeveloped_area / section_area = 3 := by
  sorry

end NUMINAMATH_CALUDE_undeveloped_sections_l618_61830


namespace NUMINAMATH_CALUDE_quadratic_equivalent_forms_l618_61890

theorem quadratic_equivalent_forms : ∀ x : ℝ, x^2 - 2*x - 1 = (x - 1)^2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equivalent_forms_l618_61890


namespace NUMINAMATH_CALUDE_infinitely_many_primes_4k_plus_3_l618_61873

theorem infinitely_many_primes_4k_plus_3 :
  ∀ (S : Finset Nat), (∀ p ∈ S, Nat.Prime p ∧ ∃ k, p = 4 * k + 3) →
  ∃ q, Nat.Prime q ∧ (∃ m, q = 4 * m + 3) ∧ q ∉ S :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_4k_plus_3_l618_61873


namespace NUMINAMATH_CALUDE_center_value_is_31_l618_61876

/-- An arithmetic sequence -/
def ArithmeticSequence (a : Fin 5 → ℕ) : Prop :=
  ∃ d, ∀ i : Fin 4, a (i + 1) = a i + d

/-- A 5x5 array where each row and column is an arithmetic sequence -/
def ArithmeticArray (A : Fin 5 → Fin 5 → ℕ) : Prop :=
  (∀ i, ArithmeticSequence (λ j => A i j)) ∧
  (∀ j, ArithmeticSequence (λ i => A i j))

theorem center_value_is_31 (A : Fin 5 → Fin 5 → ℕ) 
  (h_array : ArithmeticArray A)
  (h_first_row : A 0 0 = 1 ∧ A 0 4 = 25)
  (h_last_row : A 4 0 = 17 ∧ A 4 4 = 81) :
  A 2 2 = 31 := by
  sorry

end NUMINAMATH_CALUDE_center_value_is_31_l618_61876


namespace NUMINAMATH_CALUDE_lcm_eight_fifteen_l618_61881

theorem lcm_eight_fifteen : Nat.lcm 8 15 = 120 := by sorry

end NUMINAMATH_CALUDE_lcm_eight_fifteen_l618_61881


namespace NUMINAMATH_CALUDE_checkerboard_tiling_l618_61896

/-- The size of the checkerboard -/
def boardSize : Nat := 8

/-- The length of a trimino -/
def triminoLength : Nat := 3

/-- The width of a trimino -/
def triminoWidth : Nat := 1

/-- The area of the checkerboard -/
def boardArea : Nat := boardSize * boardSize

/-- The area of a trimino -/
def triminoArea : Nat := triminoLength * triminoWidth

theorem checkerboard_tiling (boardSize triminoLength triminoWidth : Nat) :
  ¬(boardArea % triminoArea = 0) ∧
  ((boardArea - 1) % triminoArea = 0) := by
  sorry

#check checkerboard_tiling

end NUMINAMATH_CALUDE_checkerboard_tiling_l618_61896


namespace NUMINAMATH_CALUDE_profit_at_55_profit_price_relationship_optimal_price_l618_61812

-- Define the constants and variables
def sales_cost : ℝ := 40
def initial_price : ℝ := 50
def initial_volume : ℝ := 500
def volume_decrease_rate : ℝ := 10

-- Define the sales volume function
def sales_volume (price : ℝ) : ℝ :=
  initial_volume - volume_decrease_rate * (price - initial_price)

-- Define the profit function
def profit (price : ℝ) : ℝ :=
  (price - sales_cost) * sales_volume price

-- Theorem 1: Monthly sales profit at $55 per kilogram
theorem profit_at_55 :
  profit 55 = 6750 := by sorry

-- Theorem 2: Relationship between profit and price
theorem profit_price_relationship (price : ℝ) :
  profit price = -10 * price^2 + 1400 * price - 40000 := by sorry

-- Theorem 3: Optimal price for $8000 profit without exceeding $10000 cost
theorem optimal_price :
  ∃ (price : ℝ),
    profit price = 8000 ∧
    sales_volume price * sales_cost ≤ 10000 ∧
    price = 80 := by sorry

end NUMINAMATH_CALUDE_profit_at_55_profit_price_relationship_optimal_price_l618_61812


namespace NUMINAMATH_CALUDE_permutations_with_non_adjacent_yellow_eq_11760_l618_61811

/-- The number of permutations of 3 green, 2 red, 2 white, and 3 yellow balls
    where no two yellow balls are adjacent -/
def permutations_with_non_adjacent_yellow : ℕ :=
  let green : ℕ := 3
  let red : ℕ := 2
  let white : ℕ := 2
  let yellow : ℕ := 3
  let non_yellow : ℕ := green + red + white
  let gaps : ℕ := non_yellow + 1
  (Nat.factorial non_yellow / (Nat.factorial green * Nat.factorial red * Nat.factorial white)) *
  (Nat.choose gaps yellow)

theorem permutations_with_non_adjacent_yellow_eq_11760 :
  permutations_with_non_adjacent_yellow = 11760 := by
  sorry

end NUMINAMATH_CALUDE_permutations_with_non_adjacent_yellow_eq_11760_l618_61811


namespace NUMINAMATH_CALUDE_alloy_composition_ratio_l618_61806

/-- Given two alloys A and B, with known compositions and mixture properties,
    prove that the ratio of tin to copper in alloy B is 1:4. -/
theorem alloy_composition_ratio :
  -- Define the masses of alloys
  ∀ (mass_A mass_B : ℝ),
  -- Define the ratio of lead to tin in alloy A
  ∀ (lead_ratio tin_ratio : ℝ),
  -- Define the total amount of tin in the mixture
  ∀ (total_tin : ℝ),
  -- Conditions
  mass_A = 60 →
  mass_B = 100 →
  lead_ratio = 3 →
  tin_ratio = 2 →
  total_tin = 44 →
  -- Calculate tin in alloy A
  let tin_A := (tin_ratio / (lead_ratio + tin_ratio)) * mass_A
  -- Calculate tin in alloy B
  let tin_B := total_tin - tin_A
  -- Calculate copper in alloy B
  let copper_B := mass_B - tin_B
  -- Prove the ratio
  tin_B / copper_B = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_alloy_composition_ratio_l618_61806


namespace NUMINAMATH_CALUDE_max_sum_theorem_l618_61847

theorem max_sum_theorem (x y z v w : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) (pos_v : v > 0) (pos_w : w > 0)
  (sum_sq : x^2 + y^2 + z^2 + v^2 + w^2 = 2025) : 
  ∃ (N x_N y_N z_N v_N w_N : ℝ),
    (∀ a b c d e : ℝ, a > 0 → b > 0 → c > 0 → d > 0 → e > 0 → 
      a^2 + b^2 + c^2 + d^2 + e^2 = 2025 → 
      a*c + 3*b*c + 5*c*d + 2*c*e ≤ N) ∧
    x_N > 0 ∧ y_N > 0 ∧ z_N > 0 ∧ v_N > 0 ∧ w_N > 0 ∧
    x_N^2 + y_N^2 + z_N^2 + v_N^2 + w_N^2 = 2025 ∧
    x_N*z_N + 3*y_N*z_N + 5*z_N*v_N + 2*z_N*w_N = N ∧
    N + x_N + y_N + z_N + v_N + w_N = 55 + 3037.5 * Real.sqrt 13 + 5 * Real.sqrt 202.5 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_theorem_l618_61847


namespace NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l618_61883

/-- Represents the speed of a swimmer in still water and the speed of the stream -/
structure SwimmerSpeeds where
  manSpeed : ℝ
  streamSpeed : ℝ

/-- Calculates the effective speed given the swimmer's speed and stream speed -/
def effectiveSpeed (speeds : SwimmerSpeeds) (isDownstream : Bool) : ℝ :=
  if isDownstream then speeds.manSpeed + speeds.streamSpeed else speeds.manSpeed - speeds.streamSpeed

/-- Theorem stating the speed of the man in still water given the problem conditions -/
theorem swimmer_speed_in_still_water
  (downstream_distance : ℝ)
  (upstream_distance : ℝ)
  (time : ℝ)
  (h_downstream : downstream_distance = 28)
  (h_upstream : upstream_distance = 12)
  (h_time : time = 2)
  (speeds : SwimmerSpeeds)
  (h_downstream_speed : effectiveSpeed speeds true = downstream_distance / time)
  (h_upstream_speed : effectiveSpeed speeds false = upstream_distance / time) :
  speeds.manSpeed = 10 := by
sorry


end NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l618_61883


namespace NUMINAMATH_CALUDE_paco_cookies_eaten_l618_61813

/-- Represents the number of cookies Paco ate -/
structure CookiesEaten where
  sweet : ℕ
  salty : ℕ

/-- Proves that if Paco ate 20 sweet cookies and 14 more salty cookies than sweet cookies,
    then he ate 34 salty cookies. -/
theorem paco_cookies_eaten (cookies : CookiesEaten) 
  (h1 : cookies.sweet = 20) 
  (h2 : cookies.salty = cookies.sweet + 14) : 
  cookies.salty = 34 := by
  sorry

#check paco_cookies_eaten

end NUMINAMATH_CALUDE_paco_cookies_eaten_l618_61813


namespace NUMINAMATH_CALUDE_milk_for_pizza_dough_l618_61878

/-- Calculates the amount of milk needed for a given amount of flour, based on a milk-to-flour ratio -/
def milk_needed (milk_ratio : ℚ) (flour_ratio : ℚ) (flour_amount : ℚ) : ℚ :=
  (milk_ratio / flour_ratio) * flour_amount

/-- Proves that 160 mL of milk is needed for 800 mL of flour, given the ratio of 40 mL milk to 200 mL flour -/
theorem milk_for_pizza_dough : milk_needed 40 200 800 = 160 := by
  sorry

end NUMINAMATH_CALUDE_milk_for_pizza_dough_l618_61878


namespace NUMINAMATH_CALUDE_lost_card_number_l618_61818

theorem lost_card_number (n : ℕ) (h1 : n > 0) (h2 : (n * (n + 1)) / 2 - 101 ∈ Finset.range (n + 1)) : 
  (n * (n + 1)) / 2 - 101 = 4 :=
sorry

end NUMINAMATH_CALUDE_lost_card_number_l618_61818


namespace NUMINAMATH_CALUDE_specific_grid_has_nine_triangles_l618_61815

/-- Represents the structure of the triangular grid with an additional triangle -/
structure TriangularGrid :=
  (bottom_row : Nat)
  (middle_row : Nat)
  (top_row : Nat)
  (additional : Nat)

/-- Counts the total number of triangles in the given grid structure -/
def count_triangles (grid : TriangularGrid) : Nat :=
  sorry

/-- Theorem stating that the specific grid structure has 9 triangles in total -/
theorem specific_grid_has_nine_triangles :
  let grid := TriangularGrid.mk 3 2 1 1
  count_triangles grid = 9 :=
by sorry

end NUMINAMATH_CALUDE_specific_grid_has_nine_triangles_l618_61815


namespace NUMINAMATH_CALUDE_distance_between_foci_l618_61872

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + (y + 5)^2) + Real.sqrt ((x + 6)^2 + (y - 9)^2) = 24

-- Define the foci
def focus1 : ℝ × ℝ := (4, -5)
def focus2 : ℝ × ℝ := (-6, 9)

-- Theorem statement
theorem distance_between_foci :
  ∃ (x y : ℝ), ellipse_equation x y →
  Real.sqrt ((focus1.1 - focus2.1)^2 + (focus1.2 - focus2.2)^2) = 2 * Real.sqrt 74 :=
sorry

end NUMINAMATH_CALUDE_distance_between_foci_l618_61872


namespace NUMINAMATH_CALUDE_polynomial_positivity_l618_61892

theorem polynomial_positivity (P : ℕ → ℝ) 
  (h0 : P 0 > 0)
  (h1 : P 1 > P 0)
  (h2 : P 2 > 2 * P 1 - P 0)
  (h3 : P 3 > 3 * P 2 - 3 * P 1 + P 0)
  (h4 : ∀ n : ℕ, P (n + 4) > 4 * P (n + 3) - 6 * P (n + 2) + 4 * P (n + 1) - P n) :
  ∀ n : ℕ, n > 0 → P n > 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_positivity_l618_61892


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l618_61867

theorem ceiling_floor_sum : ⌈(7 : ℚ) / 3⌉ + ⌊-(7 : ℚ) / 3⌋ = 0 := by sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l618_61867


namespace NUMINAMATH_CALUDE_matt_work_time_l618_61874

theorem matt_work_time (total_time together_time matt_remaining_time : ℝ) 
  (h1 : total_time = 20)
  (h2 : together_time = 12)
  (h3 : matt_remaining_time = 10) : 
  (total_time * matt_remaining_time) / (total_time - together_time) = 25 := by
  sorry

#check matt_work_time

end NUMINAMATH_CALUDE_matt_work_time_l618_61874


namespace NUMINAMATH_CALUDE_sixDigitPermutationsCount_l618_61846

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The factorial of a natural number -/
def factorial (n : ℕ) : ℕ := sorry

/-- The number of permutations of n elements -/
def permutations (n : ℕ) : ℕ := factorial n

/-- The number of ways to arrange k objects in n positions -/
def arrangements (n k : ℕ) : ℕ := (permutations n) / (factorial (n - k))

/-- The number of 6-digit permutations using x, y, and z with given conditions -/
def sixDigitPermutations : ℕ :=
  let xTwice := choose 6 2 * arrangements 4 2  -- x appears twice
  let xThrice := choose 6 3 * arrangements 3 1 -- x appears thrice
  let yOnce := choose 4 1                      -- y appears once
  let yThrice := choose 4 3                    -- y appears thrice
  let zTwice := 1                              -- z appears twice (only one way)
  (xTwice + xThrice) * (yOnce + yThrice) * zTwice

theorem sixDigitPermutationsCount : sixDigitPermutations = 60 := by
  sorry

end NUMINAMATH_CALUDE_sixDigitPermutationsCount_l618_61846


namespace NUMINAMATH_CALUDE_probability_one_correct_l618_61827

/-- The number of options for each multiple-choice question -/
def num_options : ℕ := 4

/-- The number of questions -/
def num_questions : ℕ := 2

/-- The number of correct answers needed -/
def correct_answers : ℕ := 1

/-- The probability of getting exactly one answer correct out of two multiple-choice questions,
    each with 4 options and only one correct answer, when answers are randomly selected -/
theorem probability_one_correct :
  (num_options - 1) * num_questions / (num_options ^ num_questions) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_correct_l618_61827


namespace NUMINAMATH_CALUDE_box_length_l618_61842

/-- The length of a rectangular box given its filling rate, width, depth, and filling time. -/
theorem box_length (fill_rate : ℝ) (width depth : ℝ) (fill_time : ℝ) :
  fill_rate = 3 →
  width = 4 →
  depth = 3 →
  fill_time = 20 →
  fill_rate * fill_time / (width * depth) = 5 := by
sorry

end NUMINAMATH_CALUDE_box_length_l618_61842


namespace NUMINAMATH_CALUDE_magic_shop_cost_correct_l618_61893

/-- Calculates the total cost for Tom and his friend at the magic shop --/
def magic_shop_cost (trick_deck_price : ℚ) (gimmick_coin_price : ℚ) 
  (trick_deck_count : ℕ) (gimmick_coin_count : ℕ) 
  (trick_deck_discount : ℚ) (gimmick_coin_discount : ℚ) 
  (sales_tax : ℚ) : ℚ :=
  let total_trick_decks := 2 * trick_deck_count * trick_deck_price
  let total_gimmick_coins := 2 * gimmick_coin_count * gimmick_coin_price
  let discounted_trick_decks := 
    if trick_deck_count > 2 then total_trick_decks * (1 - trick_deck_discount) 
    else total_trick_decks
  let discounted_gimmick_coins := 
    if gimmick_coin_count > 3 then total_gimmick_coins * (1 - gimmick_coin_discount) 
    else total_gimmick_coins
  let total_after_discounts := discounted_trick_decks + discounted_gimmick_coins
  let total_with_tax := total_after_discounts * (1 + sales_tax)
  total_with_tax

theorem magic_shop_cost_correct : 
  magic_shop_cost 8 12 3 4 (1/10) (1/20) (7/100) = 14381/100 := by
  sorry

end NUMINAMATH_CALUDE_magic_shop_cost_correct_l618_61893


namespace NUMINAMATH_CALUDE_f_greater_g_when_x_greater_two_sum_greater_four_when_f_equal_l618_61835

noncomputable def f (x : ℝ) : ℝ := (x - 1) / Real.exp (x - 1)

noncomputable def g (x : ℝ) : ℝ := f (4 - x)

theorem f_greater_g_when_x_greater_two :
  ∀ x : ℝ, x > 2 → f x > g x :=
sorry

theorem sum_greater_four_when_f_equal :
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f x₁ = f x₂ → x₁ + x₂ > 4 :=
sorry

end NUMINAMATH_CALUDE_f_greater_g_when_x_greater_two_sum_greater_four_when_f_equal_l618_61835


namespace NUMINAMATH_CALUDE_prob_event_a_is_one_third_l618_61820

/-- Represents a glove --/
inductive Glove
| Left : Glove
| Right : Glove

/-- Represents a color --/
inductive Color
| Red : Color
| Blue : Color
| Yellow : Color

/-- Represents a pair of gloves --/
def GlovePair := Color × Glove × Glove

/-- The set of all possible glove pairs --/
def allGlovePairs : Finset GlovePair :=
  sorry

/-- The event of selecting two gloves --/
def twoGloveSelection := GlovePair × GlovePair

/-- The event of selecting one left and one right glove of different colors --/
def eventA : Set twoGloveSelection :=
  sorry

/-- The probability of event A --/
def probEventA : ℚ :=
  sorry

theorem prob_event_a_is_one_third :
  probEventA = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_prob_event_a_is_one_third_l618_61820


namespace NUMINAMATH_CALUDE_hit_frequency_l618_61857

theorem hit_frequency (total_shots : ℕ) (hits : ℕ) (h1 : total_shots = 20) (h2 : hits = 15) :
  (hits : ℚ) / total_shots = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_hit_frequency_l618_61857


namespace NUMINAMATH_CALUDE_sequence_sum_2000_l618_61823

def sequence_sum (n : ℕ) : ℤ :=
  let group_sum := -1
  let num_groups := n / 6
  num_groups * group_sum

theorem sequence_sum_2000 :
  sequence_sum 2000 = -334 :=
sorry

end NUMINAMATH_CALUDE_sequence_sum_2000_l618_61823


namespace NUMINAMATH_CALUDE_complex_modulus_l618_61829

theorem complex_modulus (z : ℂ) (h : (2 - Complex.I) * z = Complex.I) : Complex.abs z = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l618_61829


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l618_61817

theorem quadratic_root_problem (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + m*x - 6
  f (-6) = 0 → ∃ (x : ℝ), x ≠ -6 ∧ f x = 0 ∧ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l618_61817


namespace NUMINAMATH_CALUDE_unique_solution_implies_any_real_l618_61800

theorem unique_solution_implies_any_real (a : ℝ) : 
  (∃! x : ℝ, x^2 - 2*a*x + a^2 = 0) → ∀ b : ℝ, ∃ a : ℝ, a = b :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_implies_any_real_l618_61800


namespace NUMINAMATH_CALUDE_cube_root_to_square_l618_61851

theorem cube_root_to_square (y : ℝ) : 
  (y + 5) ^ (1/3 : ℝ) = 3 → (y + 5)^2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_to_square_l618_61851


namespace NUMINAMATH_CALUDE_function_value_at_four_l618_61855

/-- Given a function f where f(2x) = 3x^2 + 1 for all x, prove that f(4) = 13 -/
theorem function_value_at_four (f : ℝ → ℝ) (h : ∀ x, f (2 * x) = 3 * x^2 + 1) : f 4 = 13 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_four_l618_61855


namespace NUMINAMATH_CALUDE_monotone_function_characterization_l618_61889

/-- A monotone function from integers to integers -/
def MonotoneIntFunction (f : ℤ → ℤ) : Prop :=
  ∀ x y : ℤ, x ≤ y → f x ≤ f y

/-- The functional equation that f must satisfy -/
def SatisfiesFunctionalEquation (f : ℤ → ℤ) : Prop :=
  ∀ x y : ℤ, f (x^2005 + y^2005) = (f x)^2005 + (f y)^2005

/-- The main theorem statement -/
theorem monotone_function_characterization (f : ℤ → ℤ) 
  (hm : MonotoneIntFunction f) (hf : SatisfiesFunctionalEquation f) :
  (∀ x : ℤ, f x = x) ∨ (∀ x : ℤ, f x = -x) :=
sorry

end NUMINAMATH_CALUDE_monotone_function_characterization_l618_61889


namespace NUMINAMATH_CALUDE_bernard_red_notebooks_l618_61805

theorem bernard_red_notebooks 
  (blue_notebooks : ℕ) 
  (white_notebooks : ℕ) 
  (notebooks_left : ℕ) 
  (notebooks_given : ℕ) : 
  blue_notebooks = 17 → 
  white_notebooks = 19 → 
  notebooks_left = 5 → 
  notebooks_given = 46 → 
  blue_notebooks + white_notebooks + (notebooks_given + notebooks_left - (blue_notebooks + white_notebooks)) = 
    notebooks_given + notebooks_left := by
  sorry

end NUMINAMATH_CALUDE_bernard_red_notebooks_l618_61805


namespace NUMINAMATH_CALUDE_value_of_z_l618_61831

theorem value_of_z (x y z : ℝ) (hx : x = 3) (hy : y = 2 * x) (hz : z = 3 * y) : z = 18 := by
  sorry

end NUMINAMATH_CALUDE_value_of_z_l618_61831


namespace NUMINAMATH_CALUDE_range_characterization_range_closed_under_multiplication_l618_61807

def p (m n : ℤ) : ℤ := 2 * m^2 - 6 * m * n + 5 * n^2

def in_range (k : ℤ) : Prop := ∃ m n : ℤ, p m n = k

def range_set : Set ℤ := {1, 2, 4, 5, 8, 9, 10, 13, 16, 17, 18, 20, 25, 26, 29, 32, 34, 36, 37, 40, 41, 45, 49, 50, 52, 53, 58, 61, 64, 65, 68, 72, 73, 74, 80, 81, 82, 85, 89, 90, 97, 98, 100}

theorem range_characterization :
  ∀ k : ℤ, k ∈ range_set ↔ (in_range k ∧ 1 ≤ k ∧ k ≤ 100) :=
sorry

theorem range_closed_under_multiplication :
  ∀ a b c d : ℤ, in_range (a^2 + b^2) → in_range (c^2 + d^2) →
    in_range ((a^2 + b^2) * (c^2 + d^2)) :=
sorry

end NUMINAMATH_CALUDE_range_characterization_range_closed_under_multiplication_l618_61807


namespace NUMINAMATH_CALUDE_value_of_c_l618_61862

theorem value_of_c : 1996 * 19971997 - 1995 * 19961996 = 3995992 := by
  sorry

end NUMINAMATH_CALUDE_value_of_c_l618_61862


namespace NUMINAMATH_CALUDE_total_chairs_bought_l618_61856

def living_room_chairs : ℕ := 3
def kitchen_chairs : ℕ := 6

theorem total_chairs_bought : living_room_chairs + kitchen_chairs = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_chairs_bought_l618_61856


namespace NUMINAMATH_CALUDE_mr_green_potato_yield_l618_61848

/-- Calculates the expected potato yield for a rectangular garden -/
def expected_potato_yield (length_steps : ℕ) (width_steps : ℕ) (step_length : ℚ) (yield_per_sqft : ℚ) : ℚ :=
  (length_steps : ℚ) * step_length * (width_steps : ℚ) * step_length * yield_per_sqft

/-- Theorem: The expected potato yield for Mr. Green's garden is 2109.375 pounds -/
theorem mr_green_potato_yield :
  expected_potato_yield 18 25 (5/2) (3/4) = 2109375/1000 := by
  sorry

end NUMINAMATH_CALUDE_mr_green_potato_yield_l618_61848


namespace NUMINAMATH_CALUDE_x_fourth_coefficient_l618_61826

def binomial_coefficient (n k : ℕ) : ℤ :=
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

def expansion_coefficient (n r : ℕ) : ℤ :=
  (-1)^r * binomial_coefficient n r

theorem x_fourth_coefficient :
  expansion_coefficient 8 3 = -56 :=
by sorry

end NUMINAMATH_CALUDE_x_fourth_coefficient_l618_61826


namespace NUMINAMATH_CALUDE_complex_division_equality_l618_61897

theorem complex_division_equality : Complex.I = (3 + 2 * Complex.I) / (2 - 3 * Complex.I) := by
  sorry

end NUMINAMATH_CALUDE_complex_division_equality_l618_61897


namespace NUMINAMATH_CALUDE_solution_l618_61864

def problem (x y a : ℚ) : Prop :=
  (1 / 5) * x = (5 / 8) * y ∧
  y = 40 ∧
  x + a = 4 * y

theorem solution : ∃ x y a : ℚ, problem x y a ∧ a = 35 := by
  sorry

end NUMINAMATH_CALUDE_solution_l618_61864


namespace NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l618_61899

theorem least_positive_integer_divisible_by_four_primes : 
  ∃ (p₁ p₂ p₃ p₄ : Nat), Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
  p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
  (∀ n : Nat, n > 0 → (∃ (q₁ q₂ q₃ q₄ : Nat), Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧
    q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
    n % q₁ = 0 ∧ n % q₂ = 0 ∧ n % q₃ = 0 ∧ n % q₄ = 0) → n ≥ 210) ∧
  210 % p₁ = 0 ∧ 210 % p₂ = 0 ∧ 210 % p₃ = 0 ∧ 210 % p₄ = 0 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l618_61899


namespace NUMINAMATH_CALUDE_amanda_candy_bars_l618_61837

/-- Amanda's candy bar problem -/
theorem amanda_candy_bars :
  let initial_bars : ℕ := 7
  let first_day_given : ℕ := 3
  let second_day_given : ℕ := 4 * first_day_given
  let kept_for_self : ℕ := 22
  let bought_next_day : ℕ := kept_for_self + second_day_given - (initial_bars - first_day_given)
  bought_next_day = 30 := by sorry

end NUMINAMATH_CALUDE_amanda_candy_bars_l618_61837


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l618_61859

-- Define the fourth quadrant
def fourth_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi - Real.pi / 2 < α ∧ α < 2 * k * Real.pi

-- Define the second quadrant
def second_quadrant (α : Real) : Prop :=
  ∃ n : ℤ, (2 * n + 1) * Real.pi - Real.pi / 2 < α ∧ α < (2 * n + 1) * Real.pi

-- Define the fourth quadrant
def fourth_quadrant' (α : Real) : Prop :=
  ∃ n : ℤ, 2 * n * Real.pi - Real.pi / 2 < α ∧ α < 2 * n * Real.pi

-- Theorem statement
theorem half_angle_quadrant (α : Real) :
  fourth_quadrant α → (second_quadrant (α/2) ∨ fourth_quadrant' (α/2)) :=
by sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l618_61859


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l618_61852

theorem largest_solution_of_equation : 
  ∃ (x : ℝ), x = 6 ∧ 3 * x^2 + 18 * x - 84 = x * (x + 10) ∧
  ∀ (y : ℝ), 3 * y^2 + 18 * y - 84 = y * (y + 10) → y ≤ x :=
sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l618_61852


namespace NUMINAMATH_CALUDE_sum_of_squares_not_divisible_by_17_l618_61816

theorem sum_of_squares_not_divisible_by_17 (x y z : ℤ) :
  Nat.Coprime x.natAbs y.natAbs ∧
  Nat.Coprime x.natAbs z.natAbs ∧
  Nat.Coprime y.natAbs z.natAbs →
  (x + y + z) % 17 = 0 →
  (x * y * z) % 17 = 0 →
  (x^2 + y^2 + z^2) % 17 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_not_divisible_by_17_l618_61816


namespace NUMINAMATH_CALUDE_manufacturing_hours_worked_l618_61804

theorem manufacturing_hours_worked 
  (hourly_wage : ℝ) 
  (widget_bonus : ℝ) 
  (widgets_produced : ℕ) 
  (total_earnings : ℝ) 
  (h : ℝ) -- hours worked
  (hw : hourly_wage = 12.50)
  (wb : widget_bonus = 0.16)
  (wp : widgets_produced = 1250)
  (te : total_earnings = 700)
  (earnings_equation : hourly_wage * h + widget_bonus * ↑widgets_produced = total_earnings) :
  h = 40 := by
sorry

end NUMINAMATH_CALUDE_manufacturing_hours_worked_l618_61804


namespace NUMINAMATH_CALUDE_tan_80_plus_tan_40_minus_sqrt3_tan_80_tan_40_l618_61841

theorem tan_80_plus_tan_40_minus_sqrt3_tan_80_tan_40 :
  let t80 := Real.tan (80 * π / 180)
  let t40 := Real.tan (40 * π / 180)
  t80 + t40 - Real.sqrt 3 * t80 * t40 = -Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_tan_80_plus_tan_40_minus_sqrt3_tan_80_tan_40_l618_61841


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l618_61854

theorem cyclic_sum_inequality (k : ℕ) (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) 
  (h_sum : x + y + z = 1) : 
  (x^(k+2) / (x^(k+1) + y^k + z^k)) + 
  (y^(k+2) / (y^(k+1) + z^k + x^k)) + 
  (z^(k+2) / (z^(k+1) + x^k + y^k)) ≥ 1/7 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l618_61854


namespace NUMINAMATH_CALUDE_f_range_l618_61870

noncomputable def f (x : ℝ) : ℝ := (1/3) ^ (x^2 - 2*x)

theorem f_range :
  (∀ y ∈ Set.range f, 0 < y ∧ y ≤ 3) ∧
  (∀ ε > 0, ∃ x, |f x - 3| < ε ∧ f x > 0) :=
sorry

end NUMINAMATH_CALUDE_f_range_l618_61870


namespace NUMINAMATH_CALUDE_converse_proposition_l618_61802

theorem converse_proposition :
  let P : Prop := x ≥ 2 ∧ y ≥ 3
  let Q : Prop := x + y ≥ 5
  let original : Prop := P → Q
  let converse : Prop := Q → P
  converse = (x + y ≥ 5 → x ≥ 2 ∧ y ≥ 3) := by
sorry

end NUMINAMATH_CALUDE_converse_proposition_l618_61802


namespace NUMINAMATH_CALUDE_book_arrangement_l618_61819

theorem book_arrangement (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 3) :
  (Nat.factorial n) / ((Nat.factorial (n / k))^k * Nat.factorial k) =
  (Nat.factorial 30) / ((Nat.factorial 10)^3 * Nat.factorial 3) :=
by sorry

end NUMINAMATH_CALUDE_book_arrangement_l618_61819


namespace NUMINAMATH_CALUDE_correct_calculation_l618_61886

theorem correct_calculation (x : ℝ) : 2 * (x + 6) = 28 → 6 * x = 48 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l618_61886


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l618_61840

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 - 2*x < 0}

def B : Set ℝ := {x | x ≥ 1}

theorem intersection_A_complement_B : A ∩ Bᶜ = {x : ℝ | 0 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l618_61840
