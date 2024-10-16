import Mathlib

namespace NUMINAMATH_CALUDE_lucca_ball_count_l859_85961

theorem lucca_ball_count :
  ∀ (lucca_balls : ℕ) (lucca_basketballs : ℕ) (lucien_basketballs : ℕ),
    lucca_basketballs = lucca_balls / 10 →
    lucien_basketballs = 40 →
    lucca_basketballs + lucien_basketballs = 50 →
    lucca_balls = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_lucca_ball_count_l859_85961


namespace NUMINAMATH_CALUDE_quadratic_integer_solution_count_l859_85963

theorem quadratic_integer_solution_count : ∃ (S : Finset ℚ),
  (∀ k ∈ S, |k| < 100 ∧ ∃ x : ℤ, 3 * x^2 + k * x + 8 = 0) ∧
  (∀ k : ℚ, |k| < 100 → (∃ x : ℤ, 3 * x^2 + k * x + 8 = 0) → k ∈ S) ∧
  Finset.card S = 40 :=
sorry

end NUMINAMATH_CALUDE_quadratic_integer_solution_count_l859_85963


namespace NUMINAMATH_CALUDE_ac_length_l859_85921

/-- A quadrilateral with diagonals intersecting at O --/
structure Quadrilateral :=
  (A B C D O : ℝ × ℝ)
  (OA : ℝ)
  (OC : ℝ)
  (OD : ℝ)
  (OB : ℝ)
  (BD : ℝ)
  (hOA : dist O A = OA)
  (hOC : dist O C = OC)
  (hOD : dist O D = OD)
  (hOB : dist O B = OB)
  (hBD : dist B D = BD)

/-- The theorem stating the length of AC in the given quadrilateral --/
theorem ac_length (q : Quadrilateral) 
  (h1 : q.OA = 6)
  (h2 : q.OC = 9)
  (h3 : q.OD = 6)
  (h4 : q.OB = 7)
  (h5 : q.BD = 10) :
  dist q.A q.C = 11.5 := by sorry

end NUMINAMATH_CALUDE_ac_length_l859_85921


namespace NUMINAMATH_CALUDE_felix_weight_lifting_l859_85997

/-- Felix's weight lifting problem -/
theorem felix_weight_lifting (felix_weight : ℝ) (felix_lift : ℝ) (brother_lift : ℝ)
  (h1 : felix_lift = 150)
  (h2 : brother_lift = 600)
  (h3 : brother_lift = 3 * (2 * felix_weight)) :
  felix_lift / felix_weight = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_felix_weight_lifting_l859_85997


namespace NUMINAMATH_CALUDE_roots_equation_t_value_l859_85939

theorem roots_equation_t_value (n s : ℝ) (u v : ℝ) : 
  u^2 - n*u + 6 = 0 →
  v^2 - n*v + 6 = 0 →
  (u + 2/v)^2 - s*(u + 2/v) + t = 0 →
  (v + 2/u)^2 - s*(v + 2/u) + t = 0 →
  t = 32/3 := by
sorry

end NUMINAMATH_CALUDE_roots_equation_t_value_l859_85939


namespace NUMINAMATH_CALUDE_sum_of_cubes_square_not_prime_product_l859_85983

theorem sum_of_cubes_square_not_prime_product (a b : ℕ+) (n : ℕ) :
  a^3 + b^3 = n^2 →
  ¬∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ a + b = p * q :=
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_square_not_prime_product_l859_85983


namespace NUMINAMATH_CALUDE_fraction_multiplication_l859_85930

theorem fraction_multiplication : (1/2 + 5/6 - 7/12) * (-36) = -27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l859_85930


namespace NUMINAMATH_CALUDE_cupboard_books_count_l859_85948

theorem cupboard_books_count :
  ∃! x : ℕ, x ≤ 400 ∧
    x % 4 = 1 ∧
    x % 5 = 1 ∧
    x % 6 = 1 ∧
    x % 7 = 0 ∧
    x = 301 := by
  sorry

end NUMINAMATH_CALUDE_cupboard_books_count_l859_85948


namespace NUMINAMATH_CALUDE_system_solution_condition_l859_85927

/-- The system of equations has a solution for any a if and only if 0 ≤ b ≤ 2. -/
theorem system_solution_condition (b : ℝ) :
  (∀ a : ℝ, ∃ x y : ℝ, x^2 - 2*x + y^2 = 0 ∧ a*x + y = a*b) ↔ 0 ≤ b ∧ b ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_condition_l859_85927


namespace NUMINAMATH_CALUDE_product_of_large_numbers_l859_85955

theorem product_of_large_numbers : (4 * 10^6) * (8 * 10^6) = 3.2 * 10^13 := by
  sorry

end NUMINAMATH_CALUDE_product_of_large_numbers_l859_85955


namespace NUMINAMATH_CALUDE_apple_difference_l859_85996

theorem apple_difference (initial_apples remaining_apples : ℕ) 
  (h1 : initial_apples = 46)
  (h2 : remaining_apples = 14) : 
  initial_apples - remaining_apples = 32 := by
sorry

end NUMINAMATH_CALUDE_apple_difference_l859_85996


namespace NUMINAMATH_CALUDE_car_speed_problem_l859_85910

/-- Proves that Car B's speed is 50 mph given the problem conditions -/
theorem car_speed_problem (speed_A speed_B initial_distance overtake_time final_distance : ℝ) :
  speed_A = 58 ∧ 
  initial_distance = 16 ∧ 
  overtake_time = 3 ∧ 
  final_distance = 8 ∧
  speed_A * overtake_time = speed_B * overtake_time + initial_distance + final_distance →
  speed_B = 50 := by
sorry

end NUMINAMATH_CALUDE_car_speed_problem_l859_85910


namespace NUMINAMATH_CALUDE_unique_perfect_square_sum_l859_85953

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def distinct_perfect_square_sum (a b c : ℕ) : Prop :=
  is_perfect_square a ∧ is_perfect_square b ∧ is_perfect_square c ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a + b + c = 100

theorem unique_perfect_square_sum : 
  ∃! (abc : ℕ × ℕ × ℕ), distinct_perfect_square_sum abc.1 abc.2.1 abc.2.2 :=
sorry

end NUMINAMATH_CALUDE_unique_perfect_square_sum_l859_85953


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l859_85926

/-- Given a line L1 with equation mx - m^2y = 1 passing through point P(2, 1),
    prove that the perpendicular line L2 at P has equation x + y - 3 = 0 -/
theorem perpendicular_line_equation (m : ℝ) :
  (∀ x y, m * x - m^2 * y = 1 → x = 2 ∧ y = 1) →
  (∀ x y, x + y - 3 = 0 ↔ 
    (m * x - m^2 * y = 1 → 
      (x - 2) * (x - 2) + (y - 1) * (y - 1) = 
      (2 - 2) * (2 - 2) + (1 - 1) * (1 - 1))) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l859_85926


namespace NUMINAMATH_CALUDE_total_air_removed_l859_85943

def air_removal_fractions : List Rat := [1/3, 1/4, 1/5, 1/6, 1/7]

def remaining_air (fractions : List Rat) : Rat :=
  fractions.foldl (fun acc f => acc * (1 - f)) 1

theorem total_air_removed (fractions : List Rat) :
  fractions = air_removal_fractions →
  1 - remaining_air fractions = 5/7 := by
  sorry

end NUMINAMATH_CALUDE_total_air_removed_l859_85943


namespace NUMINAMATH_CALUDE_books_in_pile_A_l859_85908

/-- Given three piles of books with the following properties:
  - The total number of books is 240
  - Pile A has 30 more than three times the books in pile B
  - Pile C has 15 fewer books than pile B
  Prove that pile A contains 165 books. -/
theorem books_in_pile_A (total : ℕ) (books_B : ℕ) (books_A : ℕ) (books_C : ℕ) : 
  total = 240 →
  books_A = 3 * books_B + 30 →
  books_C = books_B - 15 →
  books_A + books_B + books_C = total →
  books_A = 165 := by
sorry

end NUMINAMATH_CALUDE_books_in_pile_A_l859_85908


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l859_85920

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 31 = 22 % 31 ∧
  ∀ (y : ℕ), y > 0 → (5 * y) % 31 = 22 % 31 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l859_85920


namespace NUMINAMATH_CALUDE_correct_systematic_sample_l859_85937

def systematicSample (totalItems : Nat) (sampleSize : Nat) : List Nat :=
  sorry

theorem correct_systematic_sample :
  let totalItems : Nat := 50
  let sampleSize : Nat := 5
  let samplingInterval : Nat := totalItems / sampleSize
  let sample := systematicSample totalItems sampleSize
  samplingInterval = 10 ∧ sample = [7, 17, 27, 37, 47] := by sorry

end NUMINAMATH_CALUDE_correct_systematic_sample_l859_85937


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l859_85946

theorem complex_modulus_problem (i : ℂ) (h : i * i = -1) :
  let z : ℂ := 2 * i - 5 / (2 - i)
  Complex.abs z = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l859_85946


namespace NUMINAMATH_CALUDE_dot_product_zero_l859_85934

-- Define the circle
def Circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the line on which P lies
def Line (x y : ℝ) : Prop := x - y + 1 = 0

-- Define points A and B
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the dot product of two vectors
def dotProduct (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem dot_product_zero (P : ℝ × ℝ) (h : Line P.1 P.2) :
  dotProduct (P.1 - A.1, P.2 - A.2) (P.1 - B.1, P.2 - B.2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_zero_l859_85934


namespace NUMINAMATH_CALUDE_river_round_trip_time_l859_85998

/-- The time taken for a round trip on a river with given conditions -/
theorem river_round_trip_time
  (rower_speed : ℝ)
  (river_speed : ℝ)
  (distance : ℝ)
  (h1 : rower_speed = 6)
  (h2 : river_speed = 1)
  (h3 : distance = 2.916666666666667)
  : (distance / (rower_speed - river_speed)) + (distance / (rower_speed + river_speed)) = 1 := by
  sorry

#eval (2.916666666666667 / (6 - 1)) + (2.916666666666667 / (6 + 1))

end NUMINAMATH_CALUDE_river_round_trip_time_l859_85998


namespace NUMINAMATH_CALUDE_grid_routes_equal_binomial_coefficient_l859_85962

def grid_width : ℕ := 10
def grid_height : ℕ := 5

def num_routes : ℕ := Nat.choose (grid_width + grid_height) grid_height

theorem grid_routes_equal_binomial_coefficient :
  num_routes = Nat.choose (grid_width + grid_height) grid_height :=
by sorry

end NUMINAMATH_CALUDE_grid_routes_equal_binomial_coefficient_l859_85962


namespace NUMINAMATH_CALUDE_base_ten_solution_l859_85936

/-- Converts a number from base b to decimal --/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldl (fun acc d => acc * b + d) 0

/-- Checks if the equation 253_b + 146_b = 410_b holds for a given base b --/
def equation_holds (b : Nat) : Prop :=
  to_decimal [2, 5, 3] b + to_decimal [1, 4, 6] b = to_decimal [4, 1, 0] b

theorem base_ten_solution :
  ∃ (b : Nat), b > 9 ∧ equation_holds b ∧ ∀ (x : Nat), x ≠ b → ¬(equation_holds x) :=
sorry

end NUMINAMATH_CALUDE_base_ten_solution_l859_85936


namespace NUMINAMATH_CALUDE_roberts_reading_l859_85967

/-- Given Robert's reading rate and book length, calculate the maximum number of complete books he can read in a given time. -/
theorem roberts_reading (reading_rate : ℕ) (book_length : ℕ) (available_time : ℕ) :
  reading_rate > 0 →
  book_length > 0 →
  available_time > 0 →
  reading_rate = 120 →
  book_length = 360 →
  available_time = 8 →
  (available_time * reading_rate) / book_length = 2 :=
by sorry

end NUMINAMATH_CALUDE_roberts_reading_l859_85967


namespace NUMINAMATH_CALUDE_bowl_game_score_l859_85979

/-- Given the scores of Noa, Phillip, and Lucy in a bowl game, prove their total score. -/
theorem bowl_game_score (noa_score : ℕ) (phillip_score : ℕ) (lucy_score : ℕ) 
  (h1 : noa_score = 30)
  (h2 : phillip_score = 2 * noa_score)
  (h3 : lucy_score = (3 : ℕ) / 2 * phillip_score) :
  noa_score + phillip_score + lucy_score = 180 := by
  sorry

end NUMINAMATH_CALUDE_bowl_game_score_l859_85979


namespace NUMINAMATH_CALUDE_game_result_l859_85947

def f (n : ℕ) : ℕ :=
  if n % 3 = 0 then 12
  else if n % 2 = 0 then 4
  else 0

def allie_rolls : List ℕ := [2, 6, 3, 1, 6]
def betty_rolls : List ℕ := [4, 6, 3, 5]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map f |>.sum

theorem game_result : 
  total_points allie_rolls * total_points betty_rolls = 1120 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l859_85947


namespace NUMINAMATH_CALUDE_isosceles_triangle_quadratic_roots_l859_85992

theorem isosceles_triangle_quadratic_roots (a b c m : ℝ) : 
  a = 5 →
  b ≠ c →
  (b = a ∨ c = a) →
  b > 0 ∧ c > 0 →
  (b * b + (m + 2) * b + (6 - m) = 0) ∧ 
  (c * c + (m + 2) * c + (6 - m) = 0) →
  m = -10 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_quadratic_roots_l859_85992


namespace NUMINAMATH_CALUDE_compound_interest_problem_l859_85923

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

/-- Total amount calculation -/
def total_amount (principal : ℝ) (interest : ℝ) : ℝ :=
  principal + interest

theorem compound_interest_problem (P : ℝ) :
  compound_interest P 0.06 2 = 370.80 →
  total_amount P 370.80 = 3370.80 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_problem_l859_85923


namespace NUMINAMATH_CALUDE_hard_drives_sold_l859_85977

/-- Represents the number of hard drives sold -/
def num_hard_drives : ℕ := 14

/-- Represents the total earnings from all items -/
def total_earnings : ℕ := 8960

/-- Theorem stating that the number of hard drives sold is 14 -/
theorem hard_drives_sold : 
  10 * 600 + 8 * 200 + 4 * 60 + num_hard_drives * 80 = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_hard_drives_sold_l859_85977


namespace NUMINAMATH_CALUDE_inequality_solution_set_min_mn_value_l859_85951

def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

theorem inequality_solution_set (x : ℝ) :
  (f 1 x ≥ 4 - |x + 1|) ↔ (x ≤ -2 ∨ x ≥ 2) := by sorry

theorem min_mn_value (a m n : ℝ) :
  (∀ x, f a x ≤ 1 ↔ 0 ≤ x ∧ x ≤ 2) →
  m > 0 →
  n > 0 →
  1/m + 1/(2*n) = a →
  ∀ k, m*n ≤ k → 2 ≤ k := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_min_mn_value_l859_85951


namespace NUMINAMATH_CALUDE_fraction_integer_iff_p_in_range_l859_85982

theorem fraction_integer_iff_p_in_range (p : ℕ+) :
  (∃ (k : ℕ+), (4 * p + 17 : ℚ) / (3 * p - 7 : ℚ) = k) ↔ 3 ≤ p ∧ p ≤ 40 := by
sorry

end NUMINAMATH_CALUDE_fraction_integer_iff_p_in_range_l859_85982


namespace NUMINAMATH_CALUDE_ratio_p_to_q_l859_85957

def total_ways : ℕ := 6^24

def ways_p : ℕ := Nat.choose 6 2 * Nat.choose 24 2 * Nat.choose 22 6 * 
                  Nat.choose 16 4 * Nat.choose 12 4 * Nat.choose 8 4 * Nat.choose 4 4

def ways_q : ℕ := Nat.choose 6 2 * Nat.choose 24 3 * Nat.choose 21 3 * 
                  Nat.choose 18 4 * Nat.choose 14 4 * Nat.choose 10 4 * Nat.choose 6 4

def p : ℚ := ways_p / total_ways
def q : ℚ := ways_q / total_ways

theorem ratio_p_to_q : p / q = ways_p / ways_q := by
  sorry

end NUMINAMATH_CALUDE_ratio_p_to_q_l859_85957


namespace NUMINAMATH_CALUDE_distance_at_speed1_l859_85944

def total_distance : ℝ := 250
def speed1 : ℝ := 40
def speed2 : ℝ := 60
def total_time : ℝ := 5.2

theorem distance_at_speed1 (x : ℝ) 
  (h1 : x / speed1 + (total_distance - x) / speed2 = total_time) :
  x = 124 := by
  sorry

end NUMINAMATH_CALUDE_distance_at_speed1_l859_85944


namespace NUMINAMATH_CALUDE_cabbage_increase_l859_85949

/-- Represents a square garden where cabbages are grown -/
structure CabbageGarden where
  side : ℕ  -- Side length of the square garden

/-- The number of cabbages in a garden -/
def num_cabbages (g : CabbageGarden) : ℕ := g.side * g.side

/-- Theorem: If the number of cabbages increased by 199 from last year to this year,
    and the garden remained square-shaped, then the number of cabbages this year is 10,000 -/
theorem cabbage_increase (last_year this_year : CabbageGarden) :
  num_cabbages this_year = num_cabbages last_year + 199 →
  num_cabbages this_year = 10000 := by
  sorry

#check cabbage_increase

end NUMINAMATH_CALUDE_cabbage_increase_l859_85949


namespace NUMINAMATH_CALUDE_integer_root_of_cubic_l859_85909

-- Define the polynomial
def cubic_polynomial (p q : ℚ) (x : ℝ) : ℝ := x^3 + p*x + q

-- State the theorem
theorem integer_root_of_cubic (p q : ℚ) : 
  (∃ (n : ℤ), cubic_polynomial p q n = 0) →
  (cubic_polynomial p q (3 - Real.sqrt 5) = 0) →
  (∃ (n : ℤ), cubic_polynomial p q n = 0 ∧ n = -6) :=
by sorry

end NUMINAMATH_CALUDE_integer_root_of_cubic_l859_85909


namespace NUMINAMATH_CALUDE_managers_salary_l859_85973

theorem managers_salary (num_employees : ℕ) (avg_salary : ℚ) (avg_increase : ℚ) :
  num_employees = 15 →
  avg_salary = 1800 →
  avg_increase = 150 →
  let total_salary := num_employees * avg_salary
  let new_avg_salary := avg_salary + avg_increase
  let new_total_salary := (num_employees + 1) * new_avg_salary
  new_total_salary - total_salary = 4200 := by
  sorry

end NUMINAMATH_CALUDE_managers_salary_l859_85973


namespace NUMINAMATH_CALUDE_smallest_non_prime_non_square_no_small_factors_l859_85935

theorem smallest_non_prime_non_square_no_small_factors : ∃ n : ℕ,
  n = 5183 ∧
  (∀ m : ℕ, m < n →
    (Nat.Prime m → m ≥ 70) ∧
    (¬ Nat.Prime n) ∧
    (∀ k : ℕ, k * k ≠ n)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_non_prime_non_square_no_small_factors_l859_85935


namespace NUMINAMATH_CALUDE_millet_sunflower_exceed_half_on_tuesday_l859_85911

/-- Represents the proportion of seeds in the feeder -/
structure SeedMix where
  millet : ℝ
  sunflower : ℝ
  other : ℝ

/-- Calculates the next day's seed mix based on consumption and refilling -/
def nextDayMix (mix : SeedMix) : SeedMix :=
  { millet := 0.2 + 0.75 * mix.millet,
    sunflower := 0.3 + 0.5 * mix.sunflower,
    other := 0.5 }

/-- The initial seed mix on Monday -/
def initialMix : SeedMix :=
  { millet := 0.2, sunflower := 0.3, other := 0.5 }

/-- Theorem: On Tuesday, millet and sunflower seeds combined exceed 50% of total seeds -/
theorem millet_sunflower_exceed_half_on_tuesday :
  let tuesdayMix := nextDayMix initialMix
  tuesdayMix.millet + tuesdayMix.sunflower > 0.5 := by
  sorry


end NUMINAMATH_CALUDE_millet_sunflower_exceed_half_on_tuesday_l859_85911


namespace NUMINAMATH_CALUDE_adjustment_ways_l859_85942

def front_row : ℕ := 4
def back_row : ℕ := 8
def students_to_move : ℕ := 2

def ways_to_select : ℕ := Nat.choose back_row students_to_move
def ways_to_insert : ℕ := Nat.factorial (front_row + students_to_move) / Nat.factorial front_row

theorem adjustment_ways : 
  ways_to_select * ways_to_insert = 840 := by sorry

end NUMINAMATH_CALUDE_adjustment_ways_l859_85942


namespace NUMINAMATH_CALUDE_preimage_of_5_1_l859_85964

/-- The mapping f that transforms a point (x, y) to (x+y, 2x-y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, 2 * p.1 - p.2)

/-- Theorem stating that the pre-image of (5, 1) under f is (2, 3) -/
theorem preimage_of_5_1 : f (2, 3) = (5, 1) := by sorry

end NUMINAMATH_CALUDE_preimage_of_5_1_l859_85964


namespace NUMINAMATH_CALUDE_evaluate_polynomial_l859_85959

theorem evaluate_polynomial : 7^3 - 4 * 7^2 + 6 * 7 - 2 = 187 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_polynomial_l859_85959


namespace NUMINAMATH_CALUDE_negative_seven_minus_seven_l859_85980

theorem negative_seven_minus_seven : (-7) - 7 = -14 := by
  sorry

end NUMINAMATH_CALUDE_negative_seven_minus_seven_l859_85980


namespace NUMINAMATH_CALUDE_star_value_zero_l859_85985

-- Define the star operation
def star (a b c : ℤ) : ℤ := (a + b + c)^2

-- Theorem statement
theorem star_value_zero : star 3 (-5) 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_star_value_zero_l859_85985


namespace NUMINAMATH_CALUDE_shift_selection_count_l859_85941

def workers : Nat := 3
def positions : Nat := 2

theorem shift_selection_count : (workers * (workers - 1) = 6) := by
  sorry

end NUMINAMATH_CALUDE_shift_selection_count_l859_85941


namespace NUMINAMATH_CALUDE_pyramid_sphere_theorem_l859_85914

/-- Represents a triangular pyramid with a sphere touching its edges -/
structure PyramidWithSphere where
  -- Base triangle side length
  base_side : ℝ
  -- Height of the pyramid
  height : ℝ
  -- Radius of the inscribed sphere
  sphere_radius : ℝ

/-- Properties of the pyramid and sphere system -/
def pyramid_sphere_properties (p : PyramidWithSphere) : Prop :=
  -- Base is an equilateral triangle
  p.base_side = 8 ∧
  -- Height of the pyramid
  p.height = 15 ∧
  -- Sphere touches edges of the pyramid
  ∃ (aa₁ : ℝ) (dist_o_bc : ℝ),
    -- Distance from vertex A to point of contact A₁
    aa₁ = 6 ∧
    -- Distance from sphere center O to edge BC
    dist_o_bc = 18 / 5 ∧
    -- Radius of the sphere
    p.sphere_radius = 4 * Real.sqrt 39 / 5

/-- Theorem stating the properties of the pyramid and sphere system -/
theorem pyramid_sphere_theorem (p : PyramidWithSphere) :
  pyramid_sphere_properties p → 
  ∃ (aa₁ : ℝ) (dist_o_bc : ℝ),
    aa₁ = 6 ∧
    dist_o_bc = 18 / 5 ∧
    p.sphere_radius = 4 * Real.sqrt 39 / 5 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_sphere_theorem_l859_85914


namespace NUMINAMATH_CALUDE_alvin_wood_gathering_l859_85906

theorem alvin_wood_gathering (total_needed wood_from_friend wood_from_brother : ℕ) 
  (h1 : total_needed = 376)
  (h2 : wood_from_friend = 123)
  (h3 : wood_from_brother = 136) :
  total_needed - (wood_from_friend + wood_from_brother) = 117 := by
  sorry

end NUMINAMATH_CALUDE_alvin_wood_gathering_l859_85906


namespace NUMINAMATH_CALUDE_max_value_and_minimum_l859_85940

noncomputable def f (x a b c : ℝ) : ℝ := |x + a| - |x - b| + c

theorem max_value_and_minimum (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hmax : ∀ x, f x a b c ≤ 10) 
  (hmax_exists : ∃ x, f x a b c = 10) : 
  (a + b + c = 10) ∧ 
  (∀ a' b' c', a' > 0 → b' > 0 → c' > 0 → a' + b' + c' = 10 → 
    1/4 * (a - 1)^2 + (b - 2)^2 + (c - 3)^2 ≤ 1/4 * (a' - 1)^2 + (b' - 2)^2 + (c' - 3)^2) ∧
  (1/4 * (a - 1)^2 + (b - 2)^2 + (c - 3)^2 = 8/3) ∧
  (a = 11/3 ∧ b = 8/3 ∧ c = 11/3) :=
by sorry

end NUMINAMATH_CALUDE_max_value_and_minimum_l859_85940


namespace NUMINAMATH_CALUDE_equation_solution_l859_85974

def equation (x : ℝ) : Prop :=
  (45 * x)^2 = (0.45 * 1200) * 80 / (12 + 4 * 3)

theorem equation_solution :
  ∃ x : ℝ, equation x ∧ abs (x - 0.942808153803174) < 1e-10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l859_85974


namespace NUMINAMATH_CALUDE_division_remainder_l859_85966

theorem division_remainder :
  ∀ (dividend divisor quotient remainder : ℕ),
    dividend = 136 →
    divisor = 15 →
    quotient = 9 →
    dividend = divisor * quotient + remainder →
    remainder = 1 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l859_85966


namespace NUMINAMATH_CALUDE_train_length_is_300_l859_85952

/-- The length of the train in meters -/
def train_length : ℝ := 300

/-- The time (in seconds) it takes for the train to cross the platform -/
def platform_crossing_time : ℝ := 39

/-- The time (in seconds) it takes for the train to cross a signal pole -/
def pole_crossing_time : ℝ := 12

/-- The length of the platform in meters -/
def platform_length : ℝ := 675

/-- Theorem stating that the train length is 300 meters given the conditions -/
theorem train_length_is_300 :
  train_length = 300 ∧
  train_length + platform_length = (train_length / pole_crossing_time) * platform_crossing_time :=
by sorry

end NUMINAMATH_CALUDE_train_length_is_300_l859_85952


namespace NUMINAMATH_CALUDE_base_conversion_sum_l859_85903

def base_11_to_10 (n : ℕ) : ℕ := 3224

def base_5_to_10 (n : ℕ) : ℕ := 36

def base_7_to_10 (n : ℕ) : ℕ := 1362

def base_8_to_10 (n : ℕ) : ℕ := 3008

theorem base_conversion_sum :
  (base_11_to_10 2471 / base_5_to_10 121) - base_7_to_10 3654 + base_8_to_10 5680 = 1736 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_sum_l859_85903


namespace NUMINAMATH_CALUDE_unique_intersection_point_l859_85928

-- Define the function g
def g (x : ℝ) : ℝ := x^3 + 3*x^2 + 9*x + 15

-- Theorem statement
theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, p.1 = g p.2 ∧ p.2 = g p.1 ∧ p = (-3, -3) :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_point_l859_85928


namespace NUMINAMATH_CALUDE_book_page_numbering_l859_85904

def total_digits (n : ℕ) : ℕ :=
  let d1 := min n 9
  let d2 := min (n - 9) 90
  let d3 := min (n - 99) 900
  let d4 := max (n - 999) 0
  d1 + 2 * d2 + 3 * d3 + 4 * d4

theorem book_page_numbering :
  total_digits 5000 = 18893 := by
sorry

end NUMINAMATH_CALUDE_book_page_numbering_l859_85904


namespace NUMINAMATH_CALUDE_square_unbounded_l859_85975

theorem square_unbounded : ∀ (M : ℝ), M > 0 → ∃ (N : ℝ), ∀ (x : ℝ), x > N → x^2 > M := by
  sorry

end NUMINAMATH_CALUDE_square_unbounded_l859_85975


namespace NUMINAMATH_CALUDE_sin_90_degrees_l859_85969

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_90_degrees_l859_85969


namespace NUMINAMATH_CALUDE_circle_equation_correct_l859_85999

/-- The equation of a circle with center (h, k) and radius r -/
def circle_equation (x y h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- The specific circle we're considering -/
def specific_circle (x y : ℝ) : Prop :=
  (x - 3)^2 + (y + 1)^2 = 16

theorem circle_equation_correct :
  ∀ x y : ℝ, specific_circle x y ↔ circle_equation x y 3 (-1) 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_correct_l859_85999


namespace NUMINAMATH_CALUDE_sue_total_items_l859_85993

def initial_books : ℕ := 15
def initial_movies : ℕ := 6
def returned_books : ℕ := 8
def checked_out_books : ℕ := 9

def remaining_books : ℕ := initial_books - returned_books + checked_out_books
def remaining_movies : ℕ := initial_movies - (initial_movies / 3)

theorem sue_total_items : remaining_books + remaining_movies = 20 := by
  sorry

end NUMINAMATH_CALUDE_sue_total_items_l859_85993


namespace NUMINAMATH_CALUDE_sum_of_max_min_a_is_zero_l859_85916

-- Define the quadratic function
def f (a x : ℝ) : ℝ := x^2 - a*x - 20*a^2

-- Define the condition that the difference between any two solutions does not exceed 9
def solution_difference_condition (a : ℝ) : Prop :=
  ∀ x y : ℝ, f a x < 0 → f a y < 0 → |x - y| ≤ 9

-- Define the set of valid 'a' values
def valid_a_set : Set ℝ :=
  {a : ℝ | solution_difference_condition a}

-- State the theorem
theorem sum_of_max_min_a_is_zero :
  ∃ (a_min a_max : ℝ), 
    a_min ∈ valid_a_set ∧ 
    a_max ∈ valid_a_set ∧ 
    (∀ a ∈ valid_a_set, a_min ≤ a ∧ a ≤ a_max) ∧
    a_min + a_max = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_min_a_is_zero_l859_85916


namespace NUMINAMATH_CALUDE_difference_of_squares_301_297_l859_85938

theorem difference_of_squares_301_297 : 301^2 - 297^2 = 2392 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_301_297_l859_85938


namespace NUMINAMATH_CALUDE_pell_like_equation_solution_l859_85933

theorem pell_like_equation_solution (n : ℤ) :
  let x := (1/4) * ((1+Real.sqrt 2)^(2*n+1) + (1-Real.sqrt 2)^(2*n+1) - 2)
  let y := (1/(2*Real.sqrt 2)) * ((1+Real.sqrt 2)^(2*n+1) - (1-Real.sqrt 2)^(2*n+1))
  (x^2 + (x+1)^2 = y^2) ∧
  (∀ (a b : ℝ), a^2 + (a+1)^2 = b^2 → ∃ (m : ℤ), 
    a = (1/4) * ((1+Real.sqrt 2)^(2*m+1) + (1-Real.sqrt 2)^(2*m+1) - 2) ∧
    b = (1/(2*Real.sqrt 2)) * ((1+Real.sqrt 2)^(2*m+1) - (1-Real.sqrt 2)^(2*m+1)))
  := by sorry

end NUMINAMATH_CALUDE_pell_like_equation_solution_l859_85933


namespace NUMINAMATH_CALUDE_problem_solution_l859_85986

theorem problem_solution (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) 
  (h₁ : x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ + 36*x₆ + 49*x₇ = 1)
  (h₂ : 4*x₁ + 9*x₂ + 16*x₃ + 25*x₄ + 36*x₅ + 49*x₆ + 64*x₇ = 12)
  (h₃ : 9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ + 64*x₆ + 81*x₇ = 123) :
  16*x₁ + 25*x₂ + 36*x₃ + 49*x₄ + 64*x₅ + 81*x₆ + 100*x₇ = 334 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l859_85986


namespace NUMINAMATH_CALUDE_shane_minimum_score_l859_85915

def exam_count : ℕ := 5
def max_score : ℕ := 100
def goal_average : ℕ := 86
def first_three_scores : List ℕ := [81, 72, 93]

theorem shane_minimum_score :
  let total_needed : ℕ := goal_average * exam_count
  let scored_so_far : ℕ := first_three_scores.sum
  let remaining_needed : ℕ := total_needed - scored_so_far
  remaining_needed - max_score = 84 :=
by sorry

end NUMINAMATH_CALUDE_shane_minimum_score_l859_85915


namespace NUMINAMATH_CALUDE_substitution_result_l859_85968

theorem substitution_result (x y : ℝ) :
  (4 * x + 5 * y = 7) ∧ (y = 2 * x - 1) →
  4 * x + 10 * x - 5 = 7 := by
sorry

end NUMINAMATH_CALUDE_substitution_result_l859_85968


namespace NUMINAMATH_CALUDE_wally_bears_count_l859_85988

def bear_price (n : ℕ) : ℚ :=
  4 - (n - 1) * (1/2)

def total_cost (num_bears : ℕ) : ℚ :=
  (num_bears : ℚ) / 2 * (2 * 4 + (num_bears - 1) * (-1/2))

theorem wally_bears_count : 
  ∃ (n : ℕ), n > 0 ∧ total_cost n = 354 :=
sorry

end NUMINAMATH_CALUDE_wally_bears_count_l859_85988


namespace NUMINAMATH_CALUDE_eleven_points_form_120_triangles_l859_85902

/-- The number of triangles formed by 11 points on two segments -/
def numTriangles (n m : ℕ) : ℕ :=
  n * m * (m - 1) / 2 + m * n * (n - 1) / 2 + (n * (n - 1) * (n - 2)) / 6

/-- Theorem stating that 11 points on two segments (7 on one, 4 on another) form 120 triangles -/
theorem eleven_points_form_120_triangles :
  numTriangles 7 4 = 120 := by
  sorry

end NUMINAMATH_CALUDE_eleven_points_form_120_triangles_l859_85902


namespace NUMINAMATH_CALUDE_non_matching_pairings_eq_twenty_l859_85990

/-- The number of colors available for bowls and glasses -/
def num_colors : ℕ := 5

/-- The number of non-matching pairings between bowls and glasses -/
def non_matching_pairings : ℕ := num_colors * (num_colors - 1)

/-- Theorem stating that the number of non-matching pairings is 20 -/
theorem non_matching_pairings_eq_twenty : non_matching_pairings = 20 := by
  sorry

end NUMINAMATH_CALUDE_non_matching_pairings_eq_twenty_l859_85990


namespace NUMINAMATH_CALUDE_lock_problem_l859_85917

/-- The number of buttons on the lock -/
def num_buttons : ℕ := 10

/-- The number of buttons that need to be pressed simultaneously -/
def buttons_to_press : ℕ := 3

/-- The time taken for each attempt in seconds -/
def time_per_attempt : ℕ := 2

/-- The total number of possible combinations -/
def total_combinations : ℕ := (num_buttons.choose buttons_to_press)

/-- The maximum time needed to try all combinations in seconds -/
def max_time : ℕ := total_combinations * time_per_attempt

/-- The average number of attempts needed -/
def avg_attempts : ℚ := (1 + total_combinations) / 2

/-- The average time needed to open the door in seconds -/
def avg_time : ℚ := avg_attempts * time_per_attempt

/-- The maximum number of attempts possible in 60 seconds -/
def max_attempts_in_minute : ℕ := 60 / time_per_attempt

/-- The probability of opening the door in less than 60 seconds -/
def prob_less_than_minute : ℚ := (max_attempts_in_minute - 1) / total_combinations

theorem lock_problem :
  (max_time = 240) ∧
  (avg_time = 121) ∧
  (prob_less_than_minute = 29 / 120) := by
  sorry

end NUMINAMATH_CALUDE_lock_problem_l859_85917


namespace NUMINAMATH_CALUDE_absolute_value_of_negative_l859_85987

theorem absolute_value_of_negative (a : ℝ) : a < 0 → |a| = -a := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_negative_l859_85987


namespace NUMINAMATH_CALUDE_commodity_price_difference_l859_85945

/-- The year when commodity X costs 70 cents more than commodity Y -/
def yearWhenXCostsMoreThanY : ℕ := by sorry

theorem commodity_price_difference : yearWhenXCostsMoreThanY = 2010 := by
  have h1 : ∀ n : ℕ, priceX n = 420 + 30 * n := by sorry
  have h2 : ∀ n : ℕ, priceY n = 440 + 20 * n := by sorry
  have h3 : priceX 0 = 420 := by sorry
  have h4 : priceY 0 = 440 := by sorry
  have h5 : priceX yearWhenXCostsMoreThanY = priceY yearWhenXCostsMoreThanY + 70 := by sorry
  sorry

where
  priceX (n : ℕ) : ℕ := 420 + 30 * n
  priceY (n : ℕ) : ℕ := 440 + 20 * n

end NUMINAMATH_CALUDE_commodity_price_difference_l859_85945


namespace NUMINAMATH_CALUDE_compare_values_l859_85971

theorem compare_values : 
  let a := (4 : ℝ) ^ (1/4 : ℝ)
  let b := (27 : ℝ) ^ (1/3 : ℝ)
  let c := (16 : ℝ) ^ (1/8 : ℝ)
  let d := (81 : ℝ) ^ (1/2 : ℝ)
  (d > a ∧ d > b ∧ d > c) ∧ 
  (b > a ∧ b > c) :=
by sorry

end NUMINAMATH_CALUDE_compare_values_l859_85971


namespace NUMINAMATH_CALUDE_smallest_fourth_number_l859_85972

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem smallest_fourth_number (a b c d : ℕ) 
  (ha : is_two_digit a) (hb : is_two_digit b) (hc : is_two_digit c) (hd : is_two_digit d)
  (h1 : a = 45) (h2 : b = 26) (h3 : c = 63)
  (h4 : sum_of_digits a + sum_of_digits b + sum_of_digits c + sum_of_digits d = (a + b + c + d) / 3)
  (h5 : (a + b + c + d) % 7 = 0) :
  d ≥ 37 := by
sorry

end NUMINAMATH_CALUDE_smallest_fourth_number_l859_85972


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_64_l859_85995

/-- The sum of the coefficients of the terms in the expansion of (x+y+3)^3 that do not contain y -/
def sum_of_coefficients (x y : ℝ) : ℝ := (x + 3)^3

/-- Theorem: The sum of the coefficients of the terms in the expansion of (x+y+3)^3 that do not contain y is 64 -/
theorem sum_of_coefficients_is_64 :
  ∀ x y : ℝ, sum_of_coefficients x y = 64 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_64_l859_85995


namespace NUMINAMATH_CALUDE_human_habitable_area_l859_85905

/-- The fraction of Earth's surface that is not covered by water -/
def land_fraction : ℚ := 1/3

/-- The fraction of land that is inhabitable for humans -/
def inhabitable_land_fraction : ℚ := 2/3

/-- The fraction of Earth's surface that humans can live on -/
def human_habitable_fraction : ℚ := land_fraction * inhabitable_land_fraction

theorem human_habitable_area :
  human_habitable_fraction = 2/9 := by sorry

end NUMINAMATH_CALUDE_human_habitable_area_l859_85905


namespace NUMINAMATH_CALUDE_problem_solution_l859_85929

theorem problem_solution (x y z : ℝ) (h1 : x + y + z = 25) (h2 : y + z = 14) : x = 11 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l859_85929


namespace NUMINAMATH_CALUDE_quadratic_inequality_specific_case_l859_85922

theorem quadratic_inequality (k : ℝ) :
  (∀ x : ℝ, x^2 - k*x + 4 > 0) ↔ -4 < k ∧ k < 4 :=
by sorry

theorem specific_case :
  ∀ x : ℝ, x^2 - 5*x + 4 > 0 ↔ x < 1 ∨ x > 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_specific_case_l859_85922


namespace NUMINAMATH_CALUDE_fractional_inequality_solution_set_l859_85912

def solution_set : Set ℝ := Set.union (Set.Icc (-1/2) 1) (Set.Ico 1 3)

theorem fractional_inequality_solution_set :
  {x : ℝ | (x + 5) / ((x - 1)^2) ≥ 2 ∧ x ≠ 1} = solution_set := by
  sorry

end NUMINAMATH_CALUDE_fractional_inequality_solution_set_l859_85912


namespace NUMINAMATH_CALUDE_f_lower_bound_m_range_l859_85970

def f (x m : ℝ) : ℝ := |x + m^2| + |x - 2*m - 3|

theorem f_lower_bound (x m : ℝ) : f x m ≥ 2 := by sorry

theorem m_range : 
  {m : ℝ | ∀ x : ℝ, f 2 m ≤ 16} = Set.Icc (-3) (Real.sqrt 14 - 1) := by sorry

end NUMINAMATH_CALUDE_f_lower_bound_m_range_l859_85970


namespace NUMINAMATH_CALUDE_mn_value_l859_85925

theorem mn_value (m n : ℤ) (h : |3*m - 6| + (n + 4)^2 = 0) : m * n = -8 := by
  sorry

end NUMINAMATH_CALUDE_mn_value_l859_85925


namespace NUMINAMATH_CALUDE_equation_graph_is_two_lines_l859_85965

-- Define the set of points satisfying the original equation
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - p.2)^2 = 3 * p.1^2 + p.2^2}

-- Define the two lines
def L1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 0}
def L2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = -p.1}

-- State the theorem
theorem equation_graph_is_two_lines : S = L1 ∪ L2 := by
  sorry

end NUMINAMATH_CALUDE_equation_graph_is_two_lines_l859_85965


namespace NUMINAMATH_CALUDE_crayons_per_child_l859_85956

/-- Given that there are 6 children and a total of 18 crayons,
    prove that each child has 3 crayons. -/
theorem crayons_per_child :
  ∀ (total_crayons : ℕ) (num_children : ℕ),
    total_crayons = 18 →
    num_children = 6 →
    total_crayons / num_children = 3 :=
by sorry

end NUMINAMATH_CALUDE_crayons_per_child_l859_85956


namespace NUMINAMATH_CALUDE_viewership_difference_l859_85931

/-- The number of viewers for each game this week -/
structure ViewersThisWeek where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- The total number of viewers last week -/
def viewersLastWeek : ℕ := 350

/-- The conditions for this week's viewership -/
def viewershipConditions (v : ViewersThisWeek) : Prop :=
  v.second = 80 ∧
  v.first = v.second - 20 ∧
  v.third = v.second + 15 ∧
  v.fourth = v.third + (v.third / 10)

/-- The theorem to prove -/
theorem viewership_difference (v : ViewersThisWeek) 
  (h : viewershipConditions v) : 
  v.first + v.second + v.third + v.fourth = viewersLastWeek - 10 := by
  sorry

end NUMINAMATH_CALUDE_viewership_difference_l859_85931


namespace NUMINAMATH_CALUDE_point_transformation_l859_85989

def rotate_270_clockwise (x y h k : ℝ) : ℝ × ℝ :=
  (h - (y - k), k + (x - h))

def reflect_about_y_eq_x (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation (a b : ℝ) :
  let (x₁, y₁) := rotate_270_clockwise a b 2 3
  let (x₂, y₂) := reflect_about_y_eq_x x₁ y₁
  (x₂ = 4 ∧ y₂ = -7) → b - a = -7 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l859_85989


namespace NUMINAMATH_CALUDE_crepe_myrtle_count_l859_85954

theorem crepe_myrtle_count (total : ℕ) (pink : ℕ) (red : ℕ) (white : ℕ) : 
  total = 42 →
  pink = total / 3 →
  red = 2 →
  white = total - (pink + red) →
  white = 26 := by
sorry

end NUMINAMATH_CALUDE_crepe_myrtle_count_l859_85954


namespace NUMINAMATH_CALUDE_mika_birthday_stickers_l859_85960

/-- The number of stickers Mika received for her birthday -/
def birthday_stickers (initial : ℕ) (bought : ℕ) (given_away : ℕ) (used : ℕ) (left : ℕ) : ℕ :=
  (left + given_away + used) - (initial + bought)

/-- Theorem stating that Mika received 20 stickers for her birthday -/
theorem mika_birthday_stickers : 
  birthday_stickers 20 26 6 58 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_mika_birthday_stickers_l859_85960


namespace NUMINAMATH_CALUDE_elizabeth_pencil_purchase_l859_85978

/-- The amount of additional cents Elizabeth needs to purchase a pencil -/
def additional_cents_needed (elizabeth_dollars : ℕ) (borrowed_cents : ℕ) (pencil_dollars : ℕ) : ℕ :=
  pencil_dollars * 100 - (elizabeth_dollars * 100 + borrowed_cents)

theorem elizabeth_pencil_purchase :
  additional_cents_needed 5 53 6 = 47 := by
  sorry

end NUMINAMATH_CALUDE_elizabeth_pencil_purchase_l859_85978


namespace NUMINAMATH_CALUDE_four_integers_sum_problem_l859_85919

theorem four_integers_sum_problem :
  ∀ a b c d : ℕ,
    0 < a ∧ a < b ∧ b < c ∧ c < d →
    a + b + c = 6 →
    a + b + d = 7 →
    a + c + d = 8 →
    b + c + d = 9 →
    a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 4 :=
by sorry

end NUMINAMATH_CALUDE_four_integers_sum_problem_l859_85919


namespace NUMINAMATH_CALUDE_circle_ratio_l859_85907

theorem circle_ratio (r R : ℝ) (h : r > 0) (H : R > 0) 
  (area_condition : π * R^2 - π * r^2 = 4 * (π * r^2)) : 
  r / R = 1 / Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_circle_ratio_l859_85907


namespace NUMINAMATH_CALUDE_angle_function_value_l859_85994

theorem angle_function_value (α : Real) : 
  ((-4 : Real), (3 : Real)) ∈ {(x, y) | x = r * Real.cos α ∧ y = r * Real.sin α ∧ r > 0} →
  (Real.cos (π/2 + α) * Real.sin (3*π/2 - α)) / Real.tan (-π + α) = 16/25 := by
sorry

end NUMINAMATH_CALUDE_angle_function_value_l859_85994


namespace NUMINAMATH_CALUDE_cookies_given_to_friend_l859_85958

theorem cookies_given_to_friend (total : ℕ) (given_to_friend : ℕ) (given_to_family : ℕ) (eaten : ℕ) (left : ℕ) :
  total = 19 →
  given_to_family = (total - given_to_friend) / 2 →
  eaten = 2 →
  left = 5 →
  left = total - given_to_friend - given_to_family - eaten →
  given_to_friend = 5 := by
  sorry

end NUMINAMATH_CALUDE_cookies_given_to_friend_l859_85958


namespace NUMINAMATH_CALUDE_probability_ratio_l859_85932

def total_slips : ℕ := 50
def numbers_range : ℕ := 10
def slips_per_number : ℕ := 5
def drawn_slips : ℕ := 5

def probability_same_number (total : ℕ) (range : ℕ) (per_num : ℕ) (drawn : ℕ) : ℚ :=
  (range : ℚ) / Nat.choose total drawn

def probability_three_and_two (total : ℕ) (range : ℕ) (per_num : ℕ) (drawn : ℕ) : ℚ :=
  (Nat.choose range 2 * Nat.choose per_num 3 * Nat.choose per_num 2 : ℚ) / Nat.choose total drawn

theorem probability_ratio :
  (probability_three_and_two total_slips numbers_range slips_per_number drawn_slips) /
  (probability_same_number total_slips numbers_range slips_per_number drawn_slips) = 450 := by
  sorry

end NUMINAMATH_CALUDE_probability_ratio_l859_85932


namespace NUMINAMATH_CALUDE_pyarelal_loss_pyarelal_loss_is_1500_l859_85900

/-- Represents the investment and loss calculation for a business venture --/
structure BusinessVenture where
  /-- Pyarelal's investment amount --/
  pyarelal_investment : ℝ
  /-- Total loss amount in rupees --/
  total_loss : ℝ
  /-- Condition: Total loss is 2000 rupees --/
  total_loss_is_2000 : total_loss = 2000

/-- Theorem stating Pyarelal's individual loss in the business venture --/
theorem pyarelal_loss (bv : BusinessVenture) : ℝ :=
  let ashok_investment := bv.pyarelal_investment / 9
  let ramesh_investment := 2 * ashok_investment
  let total_investment := ashok_investment + bv.pyarelal_investment + ramesh_investment
  let pyarelal_share := bv.pyarelal_investment / total_investment
  pyarelal_share * bv.total_loss

/-- Theorem stating that Pyarelal's loss is 1500 rupees --/
theorem pyarelal_loss_is_1500 (bv : BusinessVenture) : pyarelal_loss bv = 1500 := by
  sorry

#check pyarelal_loss_is_1500

end NUMINAMATH_CALUDE_pyarelal_loss_pyarelal_loss_is_1500_l859_85900


namespace NUMINAMATH_CALUDE_parabola_tangent_ellipse_l859_85981

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define point A on the parabola
def point_A : ℝ × ℝ := (2, 4)

-- Define the tangent line
def tangent_line (x : ℝ) : ℝ := 4*x - 4

-- Define the ellipse
def ellipse (a b x y : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

-- State the theorem
theorem parabola_tangent_ellipse :
  ∀ a b : ℝ,
  a > b ∧ b > 0 →
  parabola (point_A.1) = point_A.2 →
  tangent_line 1 = 0 →
  tangent_line 0 = -4 →
  ellipse a b 1 0 →
  ellipse a b 0 (-4) →
  ellipse (Real.sqrt 17) 4 1 0 ∧
  ellipse (Real.sqrt 17) 4 0 (-4) :=
sorry

end NUMINAMATH_CALUDE_parabola_tangent_ellipse_l859_85981


namespace NUMINAMATH_CALUDE_smallest_student_count_l859_85918

/-- Represents the number of students in each grade --/
structure GradeCount where
  ninth : ℕ
  tenth : ℕ
  eleventh : ℕ
  twelfth : ℕ

/-- The ratios between 9th grade and other grades --/
def ratios : GradeCount → Prop
  | ⟨n, t, e, w⟩ => 3 * t = 2 * n ∧ 5 * e = 4 * n ∧ 7 * w = 6 * n

/-- The total number of students --/
def total_students (g : GradeCount) : ℕ :=
  g.ninth + g.tenth + g.eleventh + g.twelfth

/-- The theorem stating the smallest possible number of students --/
theorem smallest_student_count :
  ∃ (g : GradeCount), ratios g ∧ total_students g = 349 ∧
  (∀ (h : GradeCount), ratios h → total_students h ≥ 349) :=
sorry

end NUMINAMATH_CALUDE_smallest_student_count_l859_85918


namespace NUMINAMATH_CALUDE_inequality_proof_l859_85924

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 1/a + 1/b = 1) :
  ∀ n : ℕ, (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l859_85924


namespace NUMINAMATH_CALUDE_area_of_R2_l859_85976

/-- Rectangle R1 -/
structure Rectangle1 where
  side : ℝ
  area : ℝ

/-- Rectangle R2 -/
structure Rectangle2 where
  diagonal : ℝ

/-- Given conditions -/
def given_conditions : Prop :=
  ∃ (R1 : Rectangle1) (R2 : Rectangle2),
    R1.side = 4 ∧
    R1.area = 32 ∧
    R2.diagonal = 20 ∧
    -- Similarity condition (ratio of sides is the same)
    ∃ (k : ℝ), k > 0 ∧ R2.diagonal = k * (R1.side * (R1.area / R1.side).sqrt)

/-- Theorem: Area of R2 is 160 square inches -/
theorem area_of_R2 : given_conditions → ∃ (R2 : Rectangle2), R2.diagonal = 20 ∧ R2.diagonal^2 / 2 = 160 :=
sorry

end NUMINAMATH_CALUDE_area_of_R2_l859_85976


namespace NUMINAMATH_CALUDE_inequality_proof_l859_85950

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a^2 + b^2 + c^2 + (a + b + c)^2 ≤ 4) :
  (a*b + 1)/(a + b)^2 + (b*c + 1)/(b + c)^2 + (c*a + 1)/(c + a)^2 ≥ 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l859_85950


namespace NUMINAMATH_CALUDE_inequality_proof_l859_85984

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2*b + b^2*c + c^2*a) * (a*b^2 + b*c^2 + c*a^2) ≥ 9*a^2*b^2*c^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l859_85984


namespace NUMINAMATH_CALUDE_correct_observation_value_l859_85901

theorem correct_observation_value (n : ℕ) (original_mean corrected_mean wrong_value : ℚ) 
  (h1 : n = 50)
  (h2 : original_mean = 41)
  (h3 : corrected_mean = 41.5)
  (h4 : wrong_value = 23) :
  let original_sum := n * original_mean
  let correct_sum := n * corrected_mean
  let correct_value := correct_sum - (original_sum - wrong_value)
  correct_value = 48 := by sorry

end NUMINAMATH_CALUDE_correct_observation_value_l859_85901


namespace NUMINAMATH_CALUDE_normal_distribution_symmetry_l859_85991

-- Define a random variable following a normal distribution
def normal_distribution (μ σ : ℝ) : Type := ℝ

-- Define the probability function
def P (ξ : normal_distribution 3 σ) (pred : ℝ → Prop) : ℝ := sorry

-- Theorem statement
theorem normal_distribution_symmetry (σ : ℝ) (c : ℝ) :
  (∀ (ξ : normal_distribution 3 σ), P ξ (λ x => x > c + 1) = P ξ (λ x => x < c - 1)) →
  c = 3 :=
by sorry

end NUMINAMATH_CALUDE_normal_distribution_symmetry_l859_85991


namespace NUMINAMATH_CALUDE_cookie_count_consistency_l859_85913

theorem cookie_count_consistency (total_cookies : ℕ) (eaten_cookies : ℕ) (remaining_cookies : ℕ) 
  (h1 : total_cookies = 32)
  (h2 : eaten_cookies = 9)
  (h3 : remaining_cookies = 23) :
  total_cookies - eaten_cookies = remaining_cookies := by
  sorry

#check cookie_count_consistency

end NUMINAMATH_CALUDE_cookie_count_consistency_l859_85913
