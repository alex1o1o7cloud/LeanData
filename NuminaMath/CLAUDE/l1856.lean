import Mathlib

namespace NUMINAMATH_CALUDE_calorie_calculation_l1856_185604

/-- The number of calories in each cookie -/
def cookie_calories : ℕ := 50

/-- The number of cookies Jimmy eats -/
def cookies_eaten : ℕ := 7

/-- The number of crackers Jimmy eats -/
def crackers_eaten : ℕ := 10

/-- The total number of calories Jimmy consumes -/
def total_calories : ℕ := 500

/-- The number of calories in each cracker -/
def cracker_calories : ℕ := 15

theorem calorie_calculation :
  cookie_calories * cookies_eaten + cracker_calories * crackers_eaten = total_calories := by
  sorry

end NUMINAMATH_CALUDE_calorie_calculation_l1856_185604


namespace NUMINAMATH_CALUDE_circle_and_line_intersection_l1856_185693

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p | (p.1 - 1)^2 + p.2^2 = 25}

-- Define the line that contains the center of C
def center_line : Set (ℝ × ℝ) :=
  {p | 2 * p.1 - p.2 - 2 = 0}

-- Define the line l
def line_l (k : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 - 5 = k * (p.1 + 2)}

-- State the theorem
theorem circle_and_line_intersection
  (h1 : ((-3, 3) : ℝ × ℝ) ∈ circle_C)
  (h2 : ((1, -5) : ℝ × ℝ) ∈ circle_C)
  (h3 : ∃ c, c ∈ circle_C ∧ c ∈ center_line)
  (h4 : ∀ k > 0, ∃ A B, A ≠ B ∧ A ∈ circle_C ∧ B ∈ circle_C ∧ A ∈ line_l k ∧ B ∈ line_l k)
  (h5 : (-2, 5) ∈ line_l k) :
  (∀ p ∈ circle_C, (p.1 - 1)^2 + p.2^2 = 25) ∧
  (∀ k > 15/8, ∃ A B, A ≠ B ∧ A ∈ circle_C ∧ B ∈ circle_C ∧ A ∈ line_l k ∧ B ∈ line_l k) ∧
  (∀ k ≤ 15/8, ¬∃ A B, A ≠ B ∧ A ∈ circle_C ∧ B ∈ circle_C ∧ A ∈ line_l k ∧ B ∈ line_l k) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_line_intersection_l1856_185693


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1856_185612

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The theorem to prove -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 2 + a 4 + a 6 + a 8 + a 10 = 80 →
  a 6 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1856_185612


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l1856_185631

/-- For a geometric sequence with positive real terms, if a_1 = 1 and a_5 = 9, then a_3 = 3 -/
theorem geometric_sequence_third_term 
  (a : ℕ → ℝ) 
  (h_geom : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_pos : ∀ n, a n > 0) 
  (h_a1 : a 1 = 1) 
  (h_a5 : a 5 = 9) : 
  a 3 = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l1856_185631


namespace NUMINAMATH_CALUDE_smallest_square_area_l1856_185669

/-- The smallest square area containing two non-overlapping rectangles -/
theorem smallest_square_area (r1_width r1_height r2_width r2_height : ℕ) 
  (h1 : r1_width = 2 ∧ r1_height = 4)
  (h2 : r2_width = 3 ∧ r2_height = 5) : 
  (max (r1_width + r2_width) (max r1_height r2_height))^2 = 36 := by
  sorry

#check smallest_square_area

end NUMINAMATH_CALUDE_smallest_square_area_l1856_185669


namespace NUMINAMATH_CALUDE_women_to_total_ratio_in_salem_l1856_185661

/-- The population of Leesburg -/
def leesburg_population : ℕ := 58940

/-- The original population of Salem before people moved out -/
def salem_original_population : ℕ := 15 * leesburg_population

/-- The number of people who moved out of Salem -/
def people_moved_out : ℕ := 130000

/-- The new population of Salem after people moved out -/
def salem_new_population : ℕ := salem_original_population - people_moved_out

/-- The number of women living in Salem after the population change -/
def women_in_salem : ℕ := 377050

/-- The theorem stating the ratio of women to the total population in Salem -/
theorem women_to_total_ratio_in_salem :
  (women_in_salem : ℚ) / salem_new_population = 377050 / 754100 := by sorry

end NUMINAMATH_CALUDE_women_to_total_ratio_in_salem_l1856_185661


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1856_185656

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 1) :
  (1 / x + 1 / y) ≥ 5 + 3 * Real.sqrt 3 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 3 * y₀ = 1 ∧ 1 / x₀ + 1 / y₀ = 5 + 3 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1856_185656


namespace NUMINAMATH_CALUDE_total_sum_lent_is_2769_l1856_185691

/-- Calculates the total sum lent given the conditions of the problem -/
def totalSumLent (secondPart : ℕ) : ℕ :=
  let firstPart := (secondPart * 5) / 8
  firstPart + secondPart

/-- Proves that the total sum lent is 2769 given the problem conditions -/
theorem total_sum_lent_is_2769 :
  totalSumLent 1704 = 2769 := by
  sorry

#eval totalSumLent 1704

end NUMINAMATH_CALUDE_total_sum_lent_is_2769_l1856_185691


namespace NUMINAMATH_CALUDE_S_intersect_T_eq_T_l1856_185667

def S : Set Int := {s | ∃ n : Int, s = 2 * n + 1}
def T : Set Int := {t | ∃ n : Int, t = 4 * n + 1}

theorem S_intersect_T_eq_T : S ∩ T = T := by
  sorry

end NUMINAMATH_CALUDE_S_intersect_T_eq_T_l1856_185667


namespace NUMINAMATH_CALUDE_fixed_point_of_arithmetic_sequence_l1856_185687

/-- If k, -1, and b form an arithmetic sequence, then the line y = kx + b passes through the point (1, -2). -/
theorem fixed_point_of_arithmetic_sequence (k b : ℝ) : 
  (k - (-1) = (-1) - b) → 
  ∃ (y : ℝ), y = k * 1 + b ∧ y = -2 :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_of_arithmetic_sequence_l1856_185687


namespace NUMINAMATH_CALUDE_cube_less_than_triple_l1856_185606

theorem cube_less_than_triple : ∃! (x : ℤ), x^3 < 3*x :=
by sorry

end NUMINAMATH_CALUDE_cube_less_than_triple_l1856_185606


namespace NUMINAMATH_CALUDE_num_factors_1320_eq_32_l1856_185605

/-- The number of distinct, positive factors of 1320 -/
def num_factors_1320 : ℕ := sorry

/-- 1320 is equal to its prime factorization -/
axiom prime_fact_1320 : 1320 = 2^3 * 3 * 11 * 5

/-- Theorem: The number of distinct, positive factors of 1320 is 32 -/
theorem num_factors_1320_eq_32 : num_factors_1320 = 32 := by sorry

end NUMINAMATH_CALUDE_num_factors_1320_eq_32_l1856_185605


namespace NUMINAMATH_CALUDE_rotation_180_maps_points_l1856_185648

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Rotates a point 180 degrees clockwise around the origin -/
def rotate180 (p : Point) : Point :=
  { x := -p.x, y := -p.y }

/-- The theorem stating that rotating the given points 180 degrees clockwise
    results in the expected transformed points -/
theorem rotation_180_maps_points :
  let C : Point := { x := 3, y := -2 }
  let D : Point := { x := 2, y := -5 }
  let C' : Point := { x := -3, y := 2 }
  let D' : Point := { x := -2, y := 5 }
  rotate180 C = C' ∧ rotate180 D = D' := by
  sorry

end NUMINAMATH_CALUDE_rotation_180_maps_points_l1856_185648


namespace NUMINAMATH_CALUDE_product_sum_equality_l1856_185627

theorem product_sum_equality : 25 * 13 * 2 + 15 * 13 * 7 = 2015 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_equality_l1856_185627


namespace NUMINAMATH_CALUDE_solution_transformation_l1856_185682

theorem solution_transformation (k : ℤ) (x y : ℤ) 
  (h1 : ∃ n : ℤ, 15 * k = n) 
  (h2 : x^2 - 2*y^2 = k) : 
  ∃ t u : ℤ, t^2 - 2*u^2 = -k ∧ 
  ((t = x + 2*y ∧ u = x + y) ∨ (t = x - 2*y ∧ u = x - y)) := by
  sorry

end NUMINAMATH_CALUDE_solution_transformation_l1856_185682


namespace NUMINAMATH_CALUDE_circle_circumference_with_inscribed_rectangle_l1856_185618

/-- The circumference of a circle in which a rectangle with dimensions 10 cm by 24 cm
    is inscribed is equal to 26π cm. -/
theorem circle_circumference_with_inscribed_rectangle : 
  let rectangle_width : ℝ := 10
  let rectangle_height : ℝ := 24
  let diagonal : ℝ := (rectangle_width ^ 2 + rectangle_height ^ 2).sqrt
  let circumference : ℝ := π * diagonal
  circumference = 26 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_circumference_with_inscribed_rectangle_l1856_185618


namespace NUMINAMATH_CALUDE_eighteen_power_equality_l1856_185626

theorem eighteen_power_equality (m n : ℤ) (P Q : ℝ) 
  (hP : P = 2^m) (hQ : Q = 3^n) : 
  18^(m+n) = P^(m+n) * Q^(2*(m+n)) := by
  sorry

end NUMINAMATH_CALUDE_eighteen_power_equality_l1856_185626


namespace NUMINAMATH_CALUDE_fruit_distribution_ways_l1856_185665

def num_apples : ℕ := 2
def num_pears : ℕ := 3
def num_days : ℕ := 5

theorem fruit_distribution_ways :
  (Nat.choose num_days num_apples) = 10 := by
  sorry

end NUMINAMATH_CALUDE_fruit_distribution_ways_l1856_185665


namespace NUMINAMATH_CALUDE_statue_selling_price_l1856_185690

/-- The selling price of a statue given its cost and profit percentage -/
def selling_price (cost : ℝ) (profit_percentage : ℝ) : ℝ :=
  cost * (1 + profit_percentage)

/-- Theorem: The selling price of a statue that costs $536 and is sold at a 25% profit is $670 -/
theorem statue_selling_price : 
  selling_price 536 0.25 = 670 := by
  sorry

end NUMINAMATH_CALUDE_statue_selling_price_l1856_185690


namespace NUMINAMATH_CALUDE_total_days_calculation_l1856_185668

/-- Represents the total number of days needed to listen to a record collection. -/
def total_days (x y z t : ℕ) : ℕ := (x + y + z) * t

/-- Theorem stating that the total days needed to listen to the entire collection
    is the product of the total number of records and the time per record. -/
theorem total_days_calculation (x y z t : ℕ) : 
  total_days x y z t = (x + y + z) * t := by sorry

end NUMINAMATH_CALUDE_total_days_calculation_l1856_185668


namespace NUMINAMATH_CALUDE_sheet_difference_l1856_185613

theorem sheet_difference : ∀ (tommy jimmy : ℕ), 
  jimmy = 32 →
  jimmy + 40 = tommy + 30 →
  tommy - jimmy = 10 := by
    sorry

end NUMINAMATH_CALUDE_sheet_difference_l1856_185613


namespace NUMINAMATH_CALUDE_range_of_a_for_unique_integer_solution_l1856_185697

/-- Given a system of inequalities, prove the range of a for which there is exactly one integer solution. -/
theorem range_of_a_for_unique_integer_solution (a : ℝ) : 
  (∃! (x : ℤ), (x^3 + 3*x^2 - x - 3 > 0) ∧ 
                (x^2 - 2*a*x - 1 ≤ 0) ∧ 
                (a > 0)) ↔ 
  (3/4 ≤ a ∧ a < 4/3) := by
sorry

end NUMINAMATH_CALUDE_range_of_a_for_unique_integer_solution_l1856_185697


namespace NUMINAMATH_CALUDE_tropicenglish_word_count_l1856_185607

/-- Represents a letter in Tropicenglish -/
inductive TropicLetter
| A | M | O | P | T

/-- Represents whether a letter is a vowel or consonant -/
def isVowel : TropicLetter → Bool
  | TropicLetter.A => true
  | TropicLetter.O => true
  | _ => false

/-- A Tropicenglish word is a list of TropicLetters -/
def TropicWord := List TropicLetter

/-- Checks if a word is valid in Tropicenglish -/
def isValidWord (word : TropicWord) : Bool :=
  let consonantsBetweenVowels (w : TropicWord) : Bool :=
    -- Implementation details omitted
    sorry
  word.length == 6 && consonantsBetweenVowels word

/-- Counts the number of valid 6-letter Tropicenglish words -/
def countValidWords : Nat :=
  -- Implementation details omitted
  sorry

/-- The main theorem to prove -/
theorem tropicenglish_word_count : 
  ∃ (n : Nat), n < 1000 ∧ countValidWords % 1000 = n :=
sorry

end NUMINAMATH_CALUDE_tropicenglish_word_count_l1856_185607


namespace NUMINAMATH_CALUDE_add_like_terms_l1856_185624

theorem add_like_terms (a : ℝ) : 3 * a + 2 * a = 5 * a := by sorry

end NUMINAMATH_CALUDE_add_like_terms_l1856_185624


namespace NUMINAMATH_CALUDE_no_valid_n_l1856_185644

theorem no_valid_n : ¬∃ (n : ℕ), 
  (n > 0) ∧ 
  (1000 ≤ n / 4) ∧ (n / 4 ≤ 9999) ∧ 
  (1000 ≤ 4 * n) ∧ (4 * n ≤ 9999) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_n_l1856_185644


namespace NUMINAMATH_CALUDE_amelia_win_probability_l1856_185602

/-- Represents a player in the coin-tossing game -/
inductive Player
  | Amelia
  | Blaine
  | Calvin

/-- The probability of getting heads for each player -/
def headsProbability (p : Player) : ℚ :=
  match p with
  | Player.Amelia => 1/4
  | Player.Blaine => 1/3
  | Player.Calvin => 1/2

/-- The order of players in the game -/
def playerOrder : List Player := [Player.Amelia, Player.Blaine, Player.Calvin]

/-- The probability of Amelia winning the game -/
def ameliaWinProbability : ℚ := 1/3

/-- Theorem stating that Amelia's probability of winning is 1/3 -/
theorem amelia_win_probability : ameliaWinProbability = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_amelia_win_probability_l1856_185602


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1856_185638

def M : Set ℕ := {1, 2, 3, 5}
def N : Set ℕ := {2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1856_185638


namespace NUMINAMATH_CALUDE_chemical_solution_replacement_exists_l1856_185696

theorem chemical_solution_replacement_exists : 
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ 
  (1 - x)^5 * 0.5 + x * (0.6 + 0.65 + 0.7 + 0.75 + 0.8) = 0.63 := by
  sorry

end NUMINAMATH_CALUDE_chemical_solution_replacement_exists_l1856_185696


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1856_185675

theorem quadratic_equation_solution (m : ℝ) : 
  (∀ x, (m - 1) * x^2 + 5 * x + m^2 - 3 * m + 2 = 0) → 
  m^2 - 3 * m + 2 = 0 → 
  m - 1 ≠ 0 → 
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1856_185675


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1856_185653

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (9 - 2 * x) = 8 → x = -55/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1856_185653


namespace NUMINAMATH_CALUDE_custom_mul_solution_l1856_185616

/-- Custom multiplication operation -/
def custom_mul (a b : ℕ) : ℕ := 2 * a + b^2

/-- Theorem stating that if a * 3 = 21 under the custom multiplication, then a = 6 -/
theorem custom_mul_solution :
  ∃ a : ℕ, custom_mul a 3 = 21 ∧ a = 6 :=
by sorry

end NUMINAMATH_CALUDE_custom_mul_solution_l1856_185616


namespace NUMINAMATH_CALUDE_percentage_proof_l1856_185614

/-- The percentage of students who scored in the 70%-79% range -/
def percentage_in_range (total_students : ℕ) (students_in_range : ℕ) : ℚ :=
  students_in_range / total_students

/-- Proof that the percentage of students who scored in the 70%-79% range is 8/33 -/
theorem percentage_proof : 
  let total_students : ℕ := 33
  let students_in_range : ℕ := 8
  percentage_in_range total_students students_in_range = 8 / 33 := by
  sorry

end NUMINAMATH_CALUDE_percentage_proof_l1856_185614


namespace NUMINAMATH_CALUDE_max_sum_squared_distances_l1856_185609

theorem max_sum_squared_distances (a b c d : Fin 4 → ℝ) 
  (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hc : ‖c‖ = 1) (hd : ‖d‖ = 1) :
  (‖a - b‖^2 + ‖a - c‖^2 + ‖a - d‖^2 + ‖b - c‖^2 + ‖b - d‖^2 + ‖c - d‖^2) ≤ 24 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_squared_distances_l1856_185609


namespace NUMINAMATH_CALUDE_marble_difference_l1856_185660

theorem marble_difference (total : ℕ) (bag_a : ℕ) (bag_b : ℕ) : 
  total = 72 → bag_a = 42 → bag_b = total - bag_a → bag_a - bag_b = 12 := by
  sorry

end NUMINAMATH_CALUDE_marble_difference_l1856_185660


namespace NUMINAMATH_CALUDE_terrys_test_score_l1856_185699

theorem terrys_test_score (total_problems : ℕ) (total_score : ℕ) 
  (correct_points : ℕ) (incorrect_points : ℕ) :
  total_problems = 25 →
  total_score = 85 →
  correct_points = 4 →
  incorrect_points = 1 →
  ∃ (correct incorrect : ℕ),
    correct + incorrect = total_problems ∧
    correct_points * correct - incorrect_points * incorrect = total_score ∧
    incorrect = 3 := by
  sorry

end NUMINAMATH_CALUDE_terrys_test_score_l1856_185699


namespace NUMINAMATH_CALUDE_ellipse_equation_and_fixed_point_l1856_185637

structure Ellipse where
  center : ℝ × ℝ
  a : ℝ
  b : ℝ

def pointOnEllipse (E : Ellipse) (p : ℝ × ℝ) : Prop :=
  (p.1 - E.center.1)^2 / E.a^2 + (p.2 - E.center.2)^2 / E.b^2 = 1

def Line (p q : ℝ × ℝ) := {r : ℝ × ℝ | ∃ t, r = (1 - t) • p + t • q}

theorem ellipse_equation_and_fixed_point 
  (E : Ellipse)
  (h_center : E.center = (0, 0))
  (h_A : pointOnEllipse E (0, -2))
  (h_B : pointOnEllipse E (3/2, -1)) :
  (E.a^2 = 3 ∧ E.b^2 = 4) ∧
  ∀ (P M N T H : ℝ × ℝ),
    P = (1, -2) →
    pointOnEllipse E M →
    pointOnEllipse E N →
    M ∈ Line P N →
    T.1 = M.1 ∧ T ∈ Line (0, -2) (3/2, -1) →
    H.1 - T.1 = T.1 - M.1 ∧ H.2 = T.2 →
    (0, -2) ∈ Line H N :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_and_fixed_point_l1856_185637


namespace NUMINAMATH_CALUDE_intersection_complement_equals_set_l1856_185670

def I : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 6}
def N : Set ℕ := {2, 3, 4}

theorem intersection_complement_equals_set : M ∩ (I \ N) = {1, 6} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_set_l1856_185670


namespace NUMINAMATH_CALUDE_parallelogram_condition_l1856_185600

/-- The condition for the existence of a parallelogram inscribed in an ellipse and tangent to a circle -/
theorem parallelogram_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (∀ (x y : ℝ), x^2 + y^2 = 1 →
    ∃ (P : ℝ × ℝ), P.1^2 / a^2 + P.2^2 / b^2 = 1 ∧
      ∃ (Q R S : ℝ × ℝ),
        Q.1^2 / a^2 + Q.2^2 / b^2 = 1 ∧
        R.1^2 / a^2 + R.2^2 / b^2 = 1 ∧
        S.1^2 / a^2 + S.2^2 / b^2 = 1 ∧
        (P.1 - Q.1) * (R.1 - S.1) + (P.2 - Q.2) * (R.2 - S.2) = 0 ∧
        (P.1 - R.1) * (Q.1 - S.1) + (P.2 - R.2) * (Q.2 - S.2) = 0 ∧
        ((P.1 - x)^2 + (P.2 - y)^2 = 1 ∨
         (Q.1 - x)^2 + (Q.2 - y)^2 = 1 ∨
         (R.1 - x)^2 + (R.2 - y)^2 = 1 ∨
         (S.1 - x)^2 + (S.2 - y)^2 = 1)) ↔
  1 / a^2 + 1 / b^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_condition_l1856_185600


namespace NUMINAMATH_CALUDE_cheese_slices_lcm_l1856_185672

theorem cheese_slices_lcm : 
  let cheddar_slices : ℕ := 12
  let swiss_slices : ℕ := 28
  let gouda_slices : ℕ := 18
  Nat.lcm (Nat.lcm cheddar_slices swiss_slices) gouda_slices = 252 := by
  sorry

end NUMINAMATH_CALUDE_cheese_slices_lcm_l1856_185672


namespace NUMINAMATH_CALUDE_min_f_1998_l1856_185622

/-- A function satisfying the given property -/
def SpecialFunction (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, f (n^2 * f m) = m * (f n)^2

/-- The theorem stating the minimum value of f(1998) -/
theorem min_f_1998 (f : ℕ → ℕ) (hf : SpecialFunction f) : 
  (∀ g : ℕ → ℕ, SpecialFunction g → f 1998 ≤ g 1998) → f 1998 = 120 :=
sorry

end NUMINAMATH_CALUDE_min_f_1998_l1856_185622


namespace NUMINAMATH_CALUDE_f_monotone_increasing_l1856_185686

-- Define the function f and its derivative
noncomputable def f : ℝ → ℝ := sorry

-- State the derivative condition
axiom f_derivative (x : ℝ) : deriv f x = x * (1 - x)

-- Theorem statement
theorem f_monotone_increasing : MonotoneOn f (Set.Icc 0 1) := by sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_l1856_185686


namespace NUMINAMATH_CALUDE_max_k_value_l1856_185647

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (h : 3 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) :
  k ≤ (Real.sqrt 7 - 1) / 2 := by sorry

end NUMINAMATH_CALUDE_max_k_value_l1856_185647


namespace NUMINAMATH_CALUDE_joan_grew_29_carrots_l1856_185642

/-- The number of carrots Joan grew -/
def joans_carrots : ℕ := sorry

/-- The number of watermelons Joan grew -/
def joans_watermelons : ℕ := 14

/-- The number of carrots Jessica grew -/
def jessicas_carrots : ℕ := 11

/-- The total number of carrots grown by Joan and Jessica -/
def total_carrots : ℕ := 40

/-- Theorem stating that Joan grew 29 carrots -/
theorem joan_grew_29_carrots : joans_carrots = 29 := by
  sorry

end NUMINAMATH_CALUDE_joan_grew_29_carrots_l1856_185642


namespace NUMINAMATH_CALUDE_fraction_inequality_l1856_185652

theorem fraction_inequality (x : ℝ) (h : x ≠ -5) :
  (x - 2) / (x + 5) ≥ 0 ↔ x < -5 ∨ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1856_185652


namespace NUMINAMATH_CALUDE_number_subtraction_problem_l1856_185640

theorem number_subtraction_problem (x y : ℝ) 
  (h1 : (x - 5) / 7 = 7)
  (h2 : (x - y) / 10 = 5) : 
  y = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_subtraction_problem_l1856_185640


namespace NUMINAMATH_CALUDE_two_numbers_difference_l1856_185674

theorem two_numbers_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 80) :
  |x - y| = 8 := by sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l1856_185674


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1856_185649

theorem complex_equation_solution (z : ℂ) : z * Complex.I = 1 + Complex.I → z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1856_185649


namespace NUMINAMATH_CALUDE_necessary_condition_not_sufficient_l1856_185632

def f (x : ℝ) := |x - 2| + |x + 3|

def proposition_p (a : ℝ) := ∃ x, f x < a

theorem necessary_condition (a : ℝ) :
  (¬ proposition_p a) → a ≥ 5 := by sorry

theorem not_sufficient (a : ℝ) :
  ∃ a, a ≥ 5 ∧ proposition_p a := by sorry

end NUMINAMATH_CALUDE_necessary_condition_not_sufficient_l1856_185632


namespace NUMINAMATH_CALUDE_class_test_problem_l1856_185663

theorem class_test_problem (p_first : ℝ) (p_second : ℝ) (p_neither : ℝ) 
  (h1 : p_first = 0.75)
  (h2 : p_second = 0.7)
  (h3 : p_neither = 0.2) :
  p_first + p_second - (1 - p_neither) = 0.65 := by
  sorry

end NUMINAMATH_CALUDE_class_test_problem_l1856_185663


namespace NUMINAMATH_CALUDE_quadratic_inequality_relations_l1856_185623

/-- 
Given a quadratic inequality ax^2 - bx + c > 0 with solution set (-1, 2),
prove the relationships between a, b, and c.
-/
theorem quadratic_inequality_relations (a b c : ℝ) : 
  (∀ x : ℝ, ax^2 - b*x + c > 0 ↔ -1 < x ∧ x < 2) →
  (a < 0 ∧ b = a ∧ c = -2*a) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_relations_l1856_185623


namespace NUMINAMATH_CALUDE_negative_seven_x_is_product_l1856_185635

theorem negative_seven_x_is_product : 
  ∀ x : ℝ, -7 * x = (-7) * x :=
by
  sorry

end NUMINAMATH_CALUDE_negative_seven_x_is_product_l1856_185635


namespace NUMINAMATH_CALUDE_base_value_l1856_185666

theorem base_value (some_base : ℕ) : 
  (1/2)^16 * (1/81)^8 = 1/(some_base^16) → some_base = 18 := by sorry

end NUMINAMATH_CALUDE_base_value_l1856_185666


namespace NUMINAMATH_CALUDE_ceiling_minus_x_zero_l1856_185636

theorem ceiling_minus_x_zero (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 0) : ⌈x⌉ - x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_minus_x_zero_l1856_185636


namespace NUMINAMATH_CALUDE_dollar_op_five_neg_two_l1856_185680

def dollar_op (x y : ℤ) : ℤ := x * (2 * y - 1) + 2 * x * y

theorem dollar_op_five_neg_two : dollar_op 5 (-2) = -45 := by
  sorry

end NUMINAMATH_CALUDE_dollar_op_five_neg_two_l1856_185680


namespace NUMINAMATH_CALUDE_average_score_calculation_l1856_185692

theorem average_score_calculation (total : ℝ) (male_ratio : ℝ) (male_avg : ℝ) (female_avg : ℝ)
  (h1 : male_ratio = 0.4)
  (h2 : male_avg = 75)
  (h3 : female_avg = 80) :
  (male_ratio * male_avg + (1 - male_ratio) * female_avg) = 78 := by
  sorry

end NUMINAMATH_CALUDE_average_score_calculation_l1856_185692


namespace NUMINAMATH_CALUDE_existence_of_x0_l1856_185679

theorem existence_of_x0 (a b : ℝ) :
  ∃ x₀ ∈ Set.Icc 1 9, |a * x₀ + b + 9 / x₀| ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_x0_l1856_185679


namespace NUMINAMATH_CALUDE_right_triangle_BD_length_l1856_185634

-- Define the triangle and its properties
structure RightTriangle where
  AB : ℝ
  AC : ℝ
  BC : ℝ
  D : ℝ  -- Represents the position of D on BC
  hAB : AB = 45
  hAC : AC = 60
  hBC : BC^2 = AB^2 + AC^2
  hD : 0 < D ∧ D < BC

-- Define the theorem
theorem right_triangle_BD_length (t : RightTriangle) : 
  let AD := (t.AB * t.AC) / t.BC
  let BD := Real.sqrt (t.AB^2 - AD^2)
  BD = 27 := by sorry

end NUMINAMATH_CALUDE_right_triangle_BD_length_l1856_185634


namespace NUMINAMATH_CALUDE_square_equation_solution_l1856_185678

theorem square_equation_solution : 
  ∀ x y : ℕ+, x^2 = y^2 + 7*y + 6 ↔ x = 6 ∧ y = 3 := by sorry

end NUMINAMATH_CALUDE_square_equation_solution_l1856_185678


namespace NUMINAMATH_CALUDE_egyptian_fraction_sum_l1856_185639

theorem egyptian_fraction_sum : ∃! (b₂ b₃ b₄ b₅ : ℤ),
  (3 : ℚ) / 5 = (b₂ : ℚ) / 2 + (b₃ : ℚ) / 6 + (b₄ : ℚ) / 24 + (b₅ : ℚ) / 120 ∧
  (0 ≤ b₂ ∧ b₂ < 2) ∧
  (0 ≤ b₃ ∧ b₃ < 3) ∧
  (0 ≤ b₄ ∧ b₄ < 4) ∧
  (0 ≤ b₅ ∧ b₅ < 5) ∧
  b₂ + b₃ + b₄ + b₅ = 5 := by
sorry

end NUMINAMATH_CALUDE_egyptian_fraction_sum_l1856_185639


namespace NUMINAMATH_CALUDE_ellipse_semi_minor_axis_l1856_185676

/-- Given an ellipse with specified center, focus, and endpoint of semi-major axis,
    prove that its semi-minor axis has length 2√3. -/
theorem ellipse_semi_minor_axis 
  (center : ℝ × ℝ)
  (focus : ℝ × ℝ)
  (semi_major_endpoint : ℝ × ℝ)
  (h_center : center = (2, -1))
  (h_focus : focus = (2, -3))
  (h_semi_major_endpoint : semi_major_endpoint = (2, 3)) :
  let c := Real.sqrt ((center.1 - focus.1)^2 + (center.2 - focus.2)^2)
  let a := Real.sqrt ((center.1 - semi_major_endpoint.1)^2 + (center.2 - semi_major_endpoint.2)^2)
  let b := Real.sqrt (a^2 - c^2)
  b = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_semi_minor_axis_l1856_185676


namespace NUMINAMATH_CALUDE_x_less_than_y_less_than_zero_l1856_185658

theorem x_less_than_y_less_than_zero (x y : ℝ) 
  (h1 : x^2 - y^2 > 2*x) 
  (h2 : x*y < y) : 
  x < y ∧ y < 0 := by
sorry

end NUMINAMATH_CALUDE_x_less_than_y_less_than_zero_l1856_185658


namespace NUMINAMATH_CALUDE_inequality_holds_l1856_185620

theorem inequality_holds (a b c : ℝ) (h : a > b) : a * |c| ≥ b * |c| := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l1856_185620


namespace NUMINAMATH_CALUDE_cone_volume_from_half_sector_l1856_185698

/-- The volume of a cone formed by rolling up a half-sector of a circle --/
theorem cone_volume_from_half_sector (r : ℝ) (h : r = 6) : 
  let base_radius : ℝ := r / 2
  let cone_height : ℝ := Real.sqrt (r^2 - base_radius^2)
  (1/3 : ℝ) * Real.pi * base_radius^2 * cone_height = 9 * Real.pi * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_from_half_sector_l1856_185698


namespace NUMINAMATH_CALUDE_rectangle_width_decrease_l1856_185641

theorem rectangle_width_decrease (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  let L' := 1.4 * L
  let W' := W * L / L'
  (W - W') / W = 2 / 7 := by sorry

end NUMINAMATH_CALUDE_rectangle_width_decrease_l1856_185641


namespace NUMINAMATH_CALUDE_train_length_calculation_l1856_185625

/-- Given a train that crosses a platform in 54 seconds and a signal pole in 18 seconds,
    where the platform length is 600.0000000000001 meters, prove that the length of the train
    is 300.00000000000005 meters. -/
theorem train_length_calculation (platform_crossing_time : ℝ) (pole_crossing_time : ℝ) (platform_length : ℝ)
    (h1 : platform_crossing_time = 54)
    (h2 : pole_crossing_time = 18)
    (h3 : platform_length = 600.0000000000001) :
    ∃ (train_length : ℝ), train_length = 300.00000000000005 :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1856_185625


namespace NUMINAMATH_CALUDE_total_interest_after_trebling_l1856_185688

/-- 
Given a principal amount and an interest rate, if the simple interest 
on the principal for 10 years is 700, and the principal is trebled after 5 years, 
then the total interest at the end of the tenth year is 1750.
-/
theorem total_interest_after_trebling (P R : ℝ) : 
  (P * R * 10) / 100 = 700 → 
  ((P * R * 5) / 100) + (((3 * P) * R * 5) / 100) = 1750 := by
  sorry

#check total_interest_after_trebling

end NUMINAMATH_CALUDE_total_interest_after_trebling_l1856_185688


namespace NUMINAMATH_CALUDE_fraction_meaningful_l1856_185630

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (x - 1) / (2 - x)) ↔ x ≠ 2 := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l1856_185630


namespace NUMINAMATH_CALUDE_first_train_length_l1856_185650

/-- The length of a train given its speed, the speed and length of an oncoming train, and the time they take to cross each other. -/
def trainLength (speed1 : ℝ) (speed2 : ℝ) (length2 : ℝ) (crossTime : ℝ) : ℝ :=
  (speed1 + speed2) * crossTime - length2

/-- Theorem stating the length of the first train given the problem conditions -/
theorem first_train_length :
  let speed1 := 120 * (1000 / 3600)  -- Convert 120 km/hr to m/s
  let speed2 := 80 * (1000 / 3600)   -- Convert 80 km/hr to m/s
  let length2 := 230.04              -- Length of the second train in meters
  let crossTime := 9                 -- Time to cross in seconds
  trainLength speed1 speed2 length2 crossTime = 269.96 := by
  sorry

end NUMINAMATH_CALUDE_first_train_length_l1856_185650


namespace NUMINAMATH_CALUDE_f_has_min_value_neg_ten_l1856_185677

-- Define the function
def f (x : ℝ) : ℝ := 4 * x^2 - 12 * x - 1

-- Theorem statement
theorem f_has_min_value_neg_ten :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min ∧ f x_min = -10 :=
by sorry

end NUMINAMATH_CALUDE_f_has_min_value_neg_ten_l1856_185677


namespace NUMINAMATH_CALUDE_batteries_in_controllers_l1856_185601

def batteries_problem (total flashlights toys controllers : ℕ) : Prop :=
  total = 19 ∧ flashlights = 2 ∧ toys = 15 ∧ total = flashlights + toys + controllers

theorem batteries_in_controllers :
  ∀ total flashlights toys controllers : ℕ,
    batteries_problem total flashlights toys controllers →
    controllers = 2 :=
by sorry

end NUMINAMATH_CALUDE_batteries_in_controllers_l1856_185601


namespace NUMINAMATH_CALUDE_solution_is_121_l1856_185673

/-- Sum of first n odd numbers -/
def sumOddNumbers (n : ℕ) : ℕ := n^2

/-- Sum of first n even numbers -/
def sumEvenNumbers (n : ℕ) : ℕ := n * (n + 1)

/-- The equation from the problem -/
def equation (n : ℕ) : Prop :=
  (sumOddNumbers n : ℚ) / (sumEvenNumbers n : ℚ) = 121 / 122

theorem solution_is_121 : ∃ (n : ℕ), n > 0 ∧ equation n ∧ n = 121 := by
  sorry

end NUMINAMATH_CALUDE_solution_is_121_l1856_185673


namespace NUMINAMATH_CALUDE_arrangement_theorem_l1856_185671

/-- The number of ways to arrange 2 men and 4 women in a row, 
    such that no two men or two women are adjacent -/
def arrangement_count : ℕ := 240

/-- The number of positions between and at the ends of the men -/
def women_positions : ℕ := 5

/-- The number of men -/
def num_men : ℕ := 2

/-- The number of women -/
def num_women : ℕ := 4

theorem arrangement_theorem : 
  arrangement_count = women_positions.choose num_women * num_women.factorial * num_men.factorial :=
sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l1856_185671


namespace NUMINAMATH_CALUDE_dance_relationship_l1856_185608

/-- The number of girls that the nth boy dances with -/
def girls_danced (n : ℕ) : ℕ := n + 7

/-- The relationship between the number of boys (b) and girls (g) at a school dance -/
theorem dance_relationship (b g : ℕ) : 
  (∀ n : ℕ, n ≤ b → girls_danced n ≤ g) → 
  girls_danced b = g → 
  b = g - 7 :=
by sorry

end NUMINAMATH_CALUDE_dance_relationship_l1856_185608


namespace NUMINAMATH_CALUDE_product_profit_properties_l1856_185664

/-- A product with given cost and sales characteristics -/
structure Product where
  cost_price : ℝ
  initial_price : ℝ
  initial_sales : ℝ
  sales_increase : ℝ

/-- Daily profit as a function of price decrease -/
def daily_profit (p : Product) (x : ℝ) : ℝ :=
  (p.initial_price - x - p.cost_price) * (p.initial_sales + p.sales_increase * x)

/-- Theorem stating the properties of the product and its profit function -/
theorem product_profit_properties (p : Product) 
  (h_cost : p.cost_price = 3.5)
  (h_initial_price : p.initial_price = 14.5)
  (h_initial_sales : p.initial_sales = 500)
  (h_sales_increase : p.sales_increase = 100) :
  (∀ x, 0 ≤ x ∧ x ≤ 11 → daily_profit p x = -100 * (x - 3)^2 + 6400) ∧
  (∃ max_profit, max_profit = 6400 ∧ 
    ∀ x, 0 ≤ x ∧ x ≤ 11 → daily_profit p x ≤ max_profit) ∧
  (∃ optimal_price, optimal_price = 11.5 ∧
    ∀ x, 0 ≤ x ∧ x ≤ 11 → 
      daily_profit p ((p.initial_price - optimal_price) : ℝ) ≥ daily_profit p x) :=
sorry

end NUMINAMATH_CALUDE_product_profit_properties_l1856_185664


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1856_185603

/-- A right triangle with perimeter 40, area 30, and one angle of 45 degrees has a hypotenuse of length 2√30 -/
theorem right_triangle_hypotenuse (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem for right triangle
  a + b + c = 40 →   -- Perimeter is 40
  a * b / 2 = 30 →   -- Area is 30
  a = b →            -- One angle is 45 degrees, so adjacent sides are equal
  c = 2 * Real.sqrt 30 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1856_185603


namespace NUMINAMATH_CALUDE_coefficient_x4_proof_l1856_185628

/-- The coefficient of x^4 in the expansion of (x - 1/(2x))^10 -/
def coefficient_x4 : ℤ := -15

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

theorem coefficient_x4_proof :
  coefficient_x4 = binomial 10 3 * (-1/2)^3 := by sorry

end NUMINAMATH_CALUDE_coefficient_x4_proof_l1856_185628


namespace NUMINAMATH_CALUDE_segments_in_proportion_l1856_185645

/-- A set of four line segments is in proportion if the product of the extremes
    equals the product of the means. -/
def is_in_proportion (a b c d : ℝ) : Prop :=
  a * d = b * c

/-- The set of line segments (2, 3, 4, 6) is in proportion. -/
theorem segments_in_proportion :
  is_in_proportion 2 3 4 6 := by
  sorry

end NUMINAMATH_CALUDE_segments_in_proportion_l1856_185645


namespace NUMINAMATH_CALUDE_expression_value_l1856_185695

theorem expression_value (x : ℝ) (h : x^2 + 2*x = 1) :
  (1 - x)^2 - (x + 3)*(3 - x) - (x - 3)*(x - 1) = -10 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1856_185695


namespace NUMINAMATH_CALUDE_divides_two_pow_plus_one_congruence_l1856_185617

theorem divides_two_pow_plus_one_congruence (p : ℕ) (n : ℤ) 
  (h_prime : Nat.Prime p) (h_p_gt_3 : p > 3) 
  (h_divides : n ∣ (2^p + 1) / 3) : 
  n ≡ 1 [ZMOD (2 * p)] := by
sorry

end NUMINAMATH_CALUDE_divides_two_pow_plus_one_congruence_l1856_185617


namespace NUMINAMATH_CALUDE_relay_race_time_l1856_185684

-- Define the runners and their properties
structure Runner where
  base_time : ℝ
  obstacle_time : ℝ
  handicap : ℝ

def rhonda : Runner :=
  { base_time := 24
  , obstacle_time := 2
  , handicap := 0.95 }

def sally : Runner :=
  { base_time := 26
  , obstacle_time := 5
  , handicap := 0.90 }

def diane : Runner :=
  { base_time := 21
  , obstacle_time := 21 * 0.1
  , handicap := 1.05 }

-- Calculate the final time for a runner
def final_time (runner : Runner) : ℝ :=
  (runner.base_time + runner.obstacle_time) * runner.handicap

-- Calculate the total time for the relay race
def relay_time : ℝ :=
  final_time rhonda + final_time sally + final_time diane

-- Theorem statement
theorem relay_race_time : relay_time = 76.855 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_time_l1856_185684


namespace NUMINAMATH_CALUDE_train_crossing_time_l1856_185611

/-- Given a train and a platform with specific dimensions, calculate the time taken for the train to cross the platform. -/
theorem train_crossing_time (train_length platform_length : ℝ) (time_cross_pole : ℝ) : 
  train_length = 300 → 
  platform_length = 285 → 
  time_cross_pole = 20 → 
  (train_length + platform_length) / (train_length / time_cross_pole) = 39 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1856_185611


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_foci_l1856_185655

theorem ellipse_hyperbola_foci (a b : ℝ) : 
  (∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 → (x = 0 ∧ y = 5) ∨ (x = 0 ∧ y = -5)) →
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → (x = 7 ∧ y = 0) ∨ (x = -7 ∧ y = 0)) →
  |a * b| = Real.sqrt 444 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_foci_l1856_185655


namespace NUMINAMATH_CALUDE_complex_magnitude_l1856_185657

/-- Given two complex numbers z₁ and z₂, where z₁/z₂ is purely imaginary,
    prove that the magnitude of z₁ is 10/3. -/
theorem complex_magnitude (a : ℝ) : 
  let z₁ : ℂ := a + 2*I
  let z₂ : ℂ := 3 - 4*I
  (∃ (b : ℝ), z₁ / z₂ = b*I) → Complex.abs z₁ = 10/3 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1856_185657


namespace NUMINAMATH_CALUDE_solve_proportion_l1856_185654

theorem solve_proportion (x y : ℚ) (h1 : x / y = 8 / 3) (h2 : y = 27) : x = 72 := by
  sorry

end NUMINAMATH_CALUDE_solve_proportion_l1856_185654


namespace NUMINAMATH_CALUDE_correct_forecast_interpretation_l1856_185659

-- Define the probability of rainfall
def rainfall_probability : ℝ := 0.9

-- Define the event of getting wet when going out without rain gear
def might_get_wet (p : ℝ) : Prop :=
  p > 0 ∧ p < 1

-- Theorem statement
theorem correct_forecast_interpretation :
  might_get_wet rainfall_probability := by
  sorry

end NUMINAMATH_CALUDE_correct_forecast_interpretation_l1856_185659


namespace NUMINAMATH_CALUDE_special_sequence_common_difference_l1856_185683

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  first_term : ℝ
  last_term : ℝ
  sum : ℝ
  num_terms : ℕ
  common_difference : ℝ

/-- Properties of the specific arithmetic sequence -/
def special_sequence : ArithmeticSequence where
  first_term := 5
  last_term := 50
  sum := 275
  num_terms := 10  -- Derived from the solution, but could be proved
  common_difference := 5  -- This is what we want to prove

/-- Theorem stating that the common difference of the special sequence is 5 -/
theorem special_sequence_common_difference :
  (special_sequence.common_difference = 5) ∧
  (special_sequence.first_term = 5) ∧
  (special_sequence.last_term = 50) ∧
  (special_sequence.sum = 275) := by
  sorry

#check special_sequence_common_difference

end NUMINAMATH_CALUDE_special_sequence_common_difference_l1856_185683


namespace NUMINAMATH_CALUDE_fixed_point_exponential_l1856_185621

theorem fixed_point_exponential (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  ∀ x : ℝ, (a^(x - 1) + 1 = 2) ↔ (x = 1) :=
sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_l1856_185621


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_seven_sqrt_two_over_six_l1856_185615

theorem sqrt_difference_equals_seven_sqrt_two_over_six :
  Real.sqrt (9 / 2) - Real.sqrt (2 / 9) = (7 * Real.sqrt 2) / 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_seven_sqrt_two_over_six_l1856_185615


namespace NUMINAMATH_CALUDE_box_dimensions_solution_l1856_185694

/-- Represents the dimensions of a box --/
structure BoxDimensions where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : a + c = 17
  h2 : a + b = 13
  h3 : b + c = 20
  h4 : a < b
  h5 : b < c

/-- Proves that the dimensions of the box are 5, 8, and 12 --/
theorem box_dimensions_solution (box : BoxDimensions) : 
  box.a = 5 ∧ box.b = 8 ∧ box.c = 12 := by
  sorry

end NUMINAMATH_CALUDE_box_dimensions_solution_l1856_185694


namespace NUMINAMATH_CALUDE_parabola_focus_l1856_185651

/-- A parabola is defined by its equation in the form y = ax^2, where a is a non-zero real number. -/
structure Parabola where
  a : ℝ
  a_nonzero : a ≠ 0

/-- The focus of a parabola is a point (h, k) where h and k are real numbers. -/
structure Focus where
  h : ℝ
  k : ℝ

/-- Given a parabola y = -1/8 * x^2, its focus is at the point (0, -2). -/
theorem parabola_focus (p : Parabola) (h : p.a = -1/8) : 
  ∃ (f : Focus), f.h = 0 ∧ f.k = -2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_l1856_185651


namespace NUMINAMATH_CALUDE_intersection_equals_sqrt_set_l1856_185646

-- Define the square S
def S : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define the set Ct for a given t
def C (t : ℝ) : Set (ℝ × ℝ) := {p ∈ S | p.1 / t + p.2 / (1 - t) ≥ 1}

-- Define the intersection of all Ct
def intersectionC : Set (ℝ × ℝ) := ⋂ t ∈ {t | 0 < t ∧ t < 1}, C t

-- Define the set of points (x, y) in S such that √x + √y ≥ 1
def sqrtSet : Set (ℝ × ℝ) := {p ∈ S | Real.sqrt p.1 + Real.sqrt p.2 ≥ 1}

-- State the theorem
theorem intersection_equals_sqrt_set : intersectionC = sqrtSet := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_sqrt_set_l1856_185646


namespace NUMINAMATH_CALUDE_tan_product_seventh_roots_l1856_185681

theorem tan_product_seventh_roots : 
  Real.tan (π / 7) * Real.tan (2 * π / 7) * Real.tan (3 * π / 7) = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_seventh_roots_l1856_185681


namespace NUMINAMATH_CALUDE_power_fraction_minus_one_l1856_185685

theorem power_fraction_minus_one : (5 / 3 : ℚ) ^ 4 - 1 = 544 / 81 := by sorry

end NUMINAMATH_CALUDE_power_fraction_minus_one_l1856_185685


namespace NUMINAMATH_CALUDE_bus_schedule_theorem_l1856_185643

def is_valid_interval (T : ℚ) : Prop :=
  T < 30 ∧
  T > 0 ∧
  ∀ k : ℤ, ∀ t₀ : ℚ, 0 ≤ t₀ ∧ t₀ < T →
    (¬ (0 ≤ (t₀ + k * T) % 60 ∧ (t₀ + k * T) % 60 < 5)) ∧
    (¬ (38 ≤ (t₀ + k * T) % 60 ∧ (t₀ + k * T) % 60 < 43))

def valid_intervals : Set ℚ := {20, 15, 12, 10, 7.5, 5 + 5/11}

theorem bus_schedule_theorem :
  ∀ T : ℚ, is_valid_interval T ↔ T ∈ valid_intervals :=
sorry

end NUMINAMATH_CALUDE_bus_schedule_theorem_l1856_185643


namespace NUMINAMATH_CALUDE_bug_movement_l1856_185629

/-- Probability of the bug being at the starting vertex after n moves -/
def Q (n : ℕ) : ℚ :=
  if n = 0 then 1
  else (1 / 3) * (1 - Q (n - 1))

/-- The bug's movement on a square -/
theorem bug_movement :
  Q 8 = 547 / 2187 :=
sorry

end NUMINAMATH_CALUDE_bug_movement_l1856_185629


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l1856_185689

theorem polynomial_evaluation (y : ℝ) (h1 : y > 0) (h2 : y^2 - 3*y - 9 = 0) :
  y^3 - 3*y^2 - 9*y + 5 = 5 := by
sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l1856_185689


namespace NUMINAMATH_CALUDE_bread_needed_for_field_trip_bread_needed_proof_l1856_185633

/-- Calculates the number of pieces of bread needed for a field trip --/
theorem bread_needed_for_field_trip 
  (sandwiches_per_student : ℕ) 
  (students_per_group : ℕ) 
  (number_of_groups : ℕ) 
  (bread_per_sandwich : ℕ) : ℕ :=
  let total_students := students_per_group * number_of_groups
  let total_sandwiches := total_students * sandwiches_per_student
  let total_bread := total_sandwiches * bread_per_sandwich
  total_bread

/-- Proves that 120 pieces of bread are needed for the field trip --/
theorem bread_needed_proof :
  bread_needed_for_field_trip 2 6 5 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_bread_needed_for_field_trip_bread_needed_proof_l1856_185633


namespace NUMINAMATH_CALUDE_class_mean_calculation_l1856_185619

theorem class_mean_calculation (total_students : ℕ) 
  (group1_students group2_students group3_students : ℕ)
  (group1_mean group2_mean group3_mean : ℚ) :
  total_students = group1_students + group2_students + group3_students →
  group1_students = 50 →
  group2_students = 8 →
  group3_students = 2 →
  group1_mean = 68 / 100 →
  group2_mean = 75 / 100 →
  group3_mean = 82 / 100 →
  (group1_students * group1_mean + group2_students * group2_mean + group3_students * group3_mean) / total_students = 694 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_class_mean_calculation_l1856_185619


namespace NUMINAMATH_CALUDE_dot_product_implies_x_value_l1856_185610

/-- Given vectors a and b, if their dot product is 1, then the second component of b is 1. -/
theorem dot_product_implies_x_value (a b : ℝ × ℝ) (h : a.1 * b.1 + a.2 * b.2 = 1) 
  (ha1 : a.1 = 1) (ha2 : a.2 = -1) (hb1 : b.1 = 2) : b.2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_implies_x_value_l1856_185610


namespace NUMINAMATH_CALUDE_bob_has_77_pennies_l1856_185662

/-- The number of pennies Alex currently has -/
def alex_pennies : ℕ := sorry

/-- The number of pennies Bob currently has -/
def bob_pennies : ℕ := sorry

/-- If Alex gives Bob three pennies, Bob will have four times as many pennies as Alex has -/
axiom condition1 : bob_pennies + 3 = 4 * (alex_pennies - 3)

/-- If Bob gives Alex two pennies, Bob will have three times as many pennies as Alex has -/
axiom condition2 : bob_pennies - 2 = 3 * (alex_pennies + 2)

/-- Bob currently has 77 pennies -/
theorem bob_has_77_pennies : bob_pennies = 77 := by sorry

end NUMINAMATH_CALUDE_bob_has_77_pennies_l1856_185662
