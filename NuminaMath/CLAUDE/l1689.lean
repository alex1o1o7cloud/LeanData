import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_binomial_coeffs_l1689_168960

-- Define the binomial coefficient
def binomial_coeff (n m : ℕ) : ℕ := sorry

-- State the combinatorial identity
axiom combinatorial_identity (n m : ℕ) : 
  binomial_coeff (n + 1) m = binomial_coeff n (m - 1) + binomial_coeff n m

-- State the theorem to be proved
theorem sum_of_binomial_coeffs :
  binomial_coeff 7 4 + binomial_coeff 7 5 + binomial_coeff 8 6 = binomial_coeff 9 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_binomial_coeffs_l1689_168960


namespace NUMINAMATH_CALUDE_exam_percentage_l1689_168910

theorem exam_percentage (total_students : ℕ) (assigned_avg makeup_avg overall_avg : ℚ) 
  (h1 : total_students = 100)
  (h2 : assigned_avg = 55 / 100)
  (h3 : makeup_avg = 95 / 100)
  (h4 : overall_avg = 67 / 100) :
  ∃ (x : ℚ), 
    0 ≤ x ∧ x ≤ 1 ∧
    x * assigned_avg + (1 - x) * makeup_avg = overall_avg ∧
    x = 70 / 100 := by
  sorry

end NUMINAMATH_CALUDE_exam_percentage_l1689_168910


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1689_168987

theorem complex_equation_solution (z : ℂ) : z - 1 = (z + 1) * I → z = I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1689_168987


namespace NUMINAMATH_CALUDE_largest_x_floor_fraction_l1689_168918

theorem largest_x_floor_fraction : 
  (∃ (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) / x = 7 / 8) ∧ 
  (∀ (y : ℝ), y > 0 → (⌊y⌋ : ℝ) / y = 7 / 8 → y ≤ 48 / 7) := by
  sorry

end NUMINAMATH_CALUDE_largest_x_floor_fraction_l1689_168918


namespace NUMINAMATH_CALUDE_induction_principle_l1689_168948

theorem induction_principle (P : ℕ → Prop) :
  (∀ k, P k → P (k + 1)) →
  ¬ P 4 →
  ∀ n, n ≤ 4 → ¬ P n :=
sorry

end NUMINAMATH_CALUDE_induction_principle_l1689_168948


namespace NUMINAMATH_CALUDE_congruence_solution_l1689_168909

theorem congruence_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 17 ∧ 38574 ≡ n [ZMOD 17] ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l1689_168909


namespace NUMINAMATH_CALUDE_triangle_area_special_conditions_l1689_168947

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating the area of the triangle under given conditions -/
theorem triangle_area_special_conditions (t : Triangle) 
  (h1 : (t.a - t.c)^2 = t.b^2 - 3/4 * t.a * t.c)
  (h2 : t.b = Real.sqrt 13)
  (h3 : ∃ (d : ℝ), Real.sin t.A + Real.sin t.C = 2 * Real.sin t.B) :
  (1/2 * t.a * t.c * Real.sin t.B) = (3 * Real.sqrt 39) / 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_special_conditions_l1689_168947


namespace NUMINAMATH_CALUDE_union_of_sets_l1689_168957

def setA : Set ℝ := {x | x + 2 > 0}
def setB : Set ℝ := {y | ∃ x, y = Real.cos x}

theorem union_of_sets : setA ∪ setB = Set.Ioi (-2) := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l1689_168957


namespace NUMINAMATH_CALUDE_correct_hand_in_amount_l1689_168901

/-- Calculates the amount of money Jack will hand in given the number of bills of each denomination and the amount to be left in the till -/
def money_to_hand_in (hundreds twos fifties twenties tens fives ones leave_in_till : ℕ) : ℕ :=
  let total_in_notes := 100 * hundreds + 50 * fifties + 20 * twenties + 10 * tens + 5 * fives + ones
  total_in_notes - leave_in_till

/-- Theorem stating that the amount Jack will hand in is correct given the problem conditions -/
theorem correct_hand_in_amount :
  money_to_hand_in 2 0 1 5 3 7 27 300 = 142 := by
  sorry

#eval money_to_hand_in 2 0 1 5 3 7 27 300

end NUMINAMATH_CALUDE_correct_hand_in_amount_l1689_168901


namespace NUMINAMATH_CALUDE_sector_area_l1689_168946

/-- The area of a sector with central angle 2π/3 and radius √3 is π -/
theorem sector_area (θ : Real) (r : Real) (h1 : θ = 2 * Real.pi / 3) (h2 : r = Real.sqrt 3) :
  1/2 * r^2 * θ = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l1689_168946


namespace NUMINAMATH_CALUDE_game_lives_calculation_l1689_168974

/-- Given an initial number of players, additional players joining, and lives per player,
    calculate the total number of lives for all players. -/
def totalLives (initialPlayers additionalPlayers livesPerPlayer : ℕ) : ℕ :=
  (initialPlayers + additionalPlayers) * livesPerPlayer

/-- Prove that given 25 initial players, 10 additional players, and 15 lives per player,
    the total number of lives for all players is 525. -/
theorem game_lives_calculation :
  totalLives 25 10 15 = 525 := by
  sorry

end NUMINAMATH_CALUDE_game_lives_calculation_l1689_168974


namespace NUMINAMATH_CALUDE_fruit_selling_results_l1689_168944

/-- Represents the farmer's fruit selling scenario -/
structure FruitSelling where
  investment : ℝ
  total_yield : ℝ
  orchard_price : ℝ
  market_price : ℝ
  daily_market_sales : ℝ
  orchard_sales : ℝ

/-- The main theorem about the fruit selling scenario -/
theorem fruit_selling_results (s : FruitSelling)
  (h1 : s.investment = 13500)
  (h2 : s.total_yield = 19000)
  (h3 : s.orchard_price = 4)
  (h4 : s.market_price > 4)
  (h5 : s.daily_market_sales = 1000)
  (h6 : s.orchard_sales = 6000) :
  (s.total_yield / s.daily_market_sales = 19) ∧
  (s.total_yield * s.market_price - s.total_yield * s.orchard_price = 19000 * s.market_price - 76000) ∧
  (s.orchard_sales * s.orchard_price + (s.total_yield - s.orchard_sales) * s.market_price - s.investment = 13000 * s.market_price + 10500) := by
  sorry


end NUMINAMATH_CALUDE_fruit_selling_results_l1689_168944


namespace NUMINAMATH_CALUDE_product_remainder_zero_l1689_168990

theorem product_remainder_zero : 
  (1296 * 1444 * 1700 * 1875) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_zero_l1689_168990


namespace NUMINAMATH_CALUDE_worker_B_time_proof_l1689_168985

/-- The time taken by worker B to complete a task, given that worker A is five times as efficient and takes 15 days less than B. -/
def time_taken_by_B : ℝ := 18.75

/-- The efficiency ratio of worker A to worker B -/
def efficiency_ratio : ℝ := 5

/-- The difference in days between the time taken by B and A to complete the task -/
def time_difference : ℝ := 15

theorem worker_B_time_proof :
  ∀ (rate_B : ℝ) (time_B : ℝ),
    rate_B > 0 →
    time_B > 0 →
    efficiency_ratio * rate_B * (time_B - time_difference) = rate_B * time_B →
    time_B = time_taken_by_B :=
by sorry

end NUMINAMATH_CALUDE_worker_B_time_proof_l1689_168985


namespace NUMINAMATH_CALUDE_num_sequences_equals_binomial_remainder_of_m_mod_1000_l1689_168986

/-- The number of increasing sequences of 10 positive integers satisfying given conditions -/
def num_sequences : ℕ := sorry

/-- The upper bound for each term in the sequence -/
def upper_bound : ℕ := 2007

/-- The length of the sequence -/
def sequence_length : ℕ := 10

/-- Predicate to check if a sequence satisfies the required conditions -/
def valid_sequence (a : Fin sequence_length → ℕ) : Prop :=
  (∀ i j : Fin sequence_length, i ≤ j → a i ≤ a j) ∧
  (∀ i : Fin sequence_length, a i ≤ upper_bound) ∧
  (∀ i : Fin sequence_length, Even (a i - i.val))

theorem num_sequences_equals_binomial :
  num_sequences = Nat.choose 1008 sequence_length :=
sorry

theorem remainder_of_m_mod_1000 :
  1008 % 1000 = 8 :=
sorry

end NUMINAMATH_CALUDE_num_sequences_equals_binomial_remainder_of_m_mod_1000_l1689_168986


namespace NUMINAMATH_CALUDE_cube_root_of_special_sum_l1689_168998

theorem cube_root_of_special_sum (m n : ℚ) 
  (h : m + 2*n + Real.sqrt 2 * (2 - n) = Real.sqrt 2 * (Real.sqrt 2 + 6) + 15) :
  (((m : ℝ).sqrt + n) ^ 100) ^ (1/3 : ℝ) = 1 :=
sorry

end NUMINAMATH_CALUDE_cube_root_of_special_sum_l1689_168998


namespace NUMINAMATH_CALUDE_jeff_peanut_butter_amount_l1689_168950

/-- The amount of peanut butter in ounces for each jar size -/
def jar_sizes : List Nat := [16, 28, 40]

/-- The total number of jars Jeff has -/
def total_jars : Nat := 9

/-- Theorem stating that Jeff has 252 ounces of peanut butter -/
theorem jeff_peanut_butter_amount :
  (total_jars / jar_sizes.length) * (jar_sizes.sum) = 252 := by
  sorry

#check jeff_peanut_butter_amount

end NUMINAMATH_CALUDE_jeff_peanut_butter_amount_l1689_168950


namespace NUMINAMATH_CALUDE_fixed_point_of_parabolas_unique_fixed_point_l1689_168979

/-- The fixed point of a family of parabolas -/
theorem fixed_point_of_parabolas (t : ℝ) :
  let f (x : ℝ) := 5 * x^2 + 4 * t * x - 3 * t
  f (3/4) = 45/16 := by sorry

/-- The uniqueness of the fixed point -/
theorem unique_fixed_point (t₁ t₂ : ℝ) (x : ℝ) :
  let f₁ (x : ℝ) := 5 * x^2 + 4 * t₁ * x - 3 * t₁
  let f₂ (x : ℝ) := 5 * x^2 + 4 * t₂ * x - 3 * t₂
  f₁ x = f₂ x → x = 3/4 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_parabolas_unique_fixed_point_l1689_168979


namespace NUMINAMATH_CALUDE_cube_side_ratio_l1689_168953

theorem cube_side_ratio (s S : ℝ) (h : s > 0) (H : S > 0) :
  (6 * S^2) / (6 * s^2) = 9 → S / s = 3 := by
sorry

end NUMINAMATH_CALUDE_cube_side_ratio_l1689_168953


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l1689_168935

/-- A coloring function for points in ℚ × ℚ -/
def Coloring := ℚ × ℚ → Fin 2

/-- The distance between two points in ℚ × ℚ -/
def distance (p q : ℚ × ℚ) : ℚ :=
  ((p.1 - q.1)^2 + (p.2 - q.2)^2).sqrt

/-- A valid coloring function assigns different colors to points with distance 1 -/
def is_valid_coloring (f : Coloring) : Prop :=
  ∀ p q : ℚ × ℚ, distance p q = 1 → f p ≠ f q

theorem exists_valid_coloring : ∃ f : Coloring, is_valid_coloring f := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l1689_168935


namespace NUMINAMATH_CALUDE_total_cost_after_discounts_l1689_168970

def dozen : ℕ := 12

def red_roses : ℕ := 2 * dozen
def white_roses : ℕ := 1 * dozen
def yellow_roses : ℕ := 2 * dozen

def red_price : ℚ := 6
def white_price : ℚ := 7
def yellow_price : ℚ := 5

def total_roses : ℕ := red_roses + white_roses + yellow_roses

def initial_cost : ℚ := 
  red_roses * red_price + white_roses * white_price + yellow_roses * yellow_price

def first_discount_rate : ℚ := 15 / 100
def second_discount_rate : ℚ := 10 / 100

theorem total_cost_after_discounts :
  total_roses > 30 ∧ total_roses > 50 →
  let cost_after_first_discount := initial_cost * (1 - first_discount_rate)
  let final_cost := cost_after_first_discount * (1 - second_discount_rate)
  final_cost = 266.22 := by sorry

end NUMINAMATH_CALUDE_total_cost_after_discounts_l1689_168970


namespace NUMINAMATH_CALUDE_pet_store_bird_count_l1689_168905

theorem pet_store_bird_count :
  let num_cages : ℕ := 6
  let parrots_per_cage : ℕ := 6
  let parakeets_per_cage : ℕ := 2
  let total_birds : ℕ := num_cages * (parrots_per_cage + parakeets_per_cage)
  total_birds = 48 := by
sorry

end NUMINAMATH_CALUDE_pet_store_bird_count_l1689_168905


namespace NUMINAMATH_CALUDE_westpark_teachers_l1689_168929

/-- The number of students at Westpark High School -/
def total_students : ℕ := 900

/-- The number of classes each student takes per day -/
def classes_per_student : ℕ := 6

/-- The number of classes each teacher teaches -/
def classes_per_teacher : ℕ := 5

/-- The number of students in each class -/
def students_per_class : ℕ := 25

/-- The number of teachers in each class -/
def teachers_per_class : ℕ := 1

/-- Calculate the number of teachers required at Westpark High School -/
def calculate_teachers : ℕ := 
  (total_students * classes_per_student / students_per_class + 
   (if (total_students * classes_per_student) % students_per_class = 0 then 0 else 1)) / 
  classes_per_teacher +
  (if ((total_students * classes_per_student / students_per_class + 
       (if (total_students * classes_per_student) % students_per_class = 0 then 0 else 1)) % 
      classes_per_teacher = 0) 
   then 0 
   else 1)

/-- Theorem stating that the number of teachers at Westpark High School is 44 -/
theorem westpark_teachers : calculate_teachers = 44 := by
  sorry

end NUMINAMATH_CALUDE_westpark_teachers_l1689_168929


namespace NUMINAMATH_CALUDE_prime_difference_values_l1689_168995

theorem prime_difference_values (p q : ℕ) (n : ℕ+) 
  (h_p : Nat.Prime p) (h_q : Nat.Prime q) 
  (h_eq : (p : ℚ) / (p + 1) + (q + 1 : ℚ) / q = (2 * n) / (n + 2)) :
  q - p ∈ ({2, 3, 5} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_prime_difference_values_l1689_168995


namespace NUMINAMATH_CALUDE_triangle_analogous_to_tetrahedron_l1689_168959

/-- Represents geometric objects -/
inductive GeometricObject
  | Quadrilateral
  | Pyramid
  | Triangle
  | Prism
  | Tetrahedron

/-- Defines the concept of analogous objects based on shared properties -/
def are_analogous (a b : GeometricObject) : Prop :=
  ∃ (property : GeometricObject → Prop), property a ∧ property b

/-- Theorem stating that a triangle is analogous to a tetrahedron -/
theorem triangle_analogous_to_tetrahedron :
  are_analogous GeometricObject.Triangle GeometricObject.Tetrahedron :=
sorry

end NUMINAMATH_CALUDE_triangle_analogous_to_tetrahedron_l1689_168959


namespace NUMINAMATH_CALUDE_max_value_3x_4y_l1689_168955

theorem max_value_3x_4y (x y : ℝ) : 
  x^2 + y^2 = 14*x + 6*y + 6 → (3*x + 4*y ≤ 73) := by
  sorry

end NUMINAMATH_CALUDE_max_value_3x_4y_l1689_168955


namespace NUMINAMATH_CALUDE_simplification_and_exponent_sum_l1689_168983

-- Define the expression
def original_expression (x y z : ℝ) : ℝ := (40 * x^5 * y^9 * z^14) ^ (1/3)

-- Define the simplified expression
def simplified_expression (x y z : ℝ) : ℝ := 2 * x * y * z^4 * (10 * x^2 * z^2) ^ (1/3)

-- Theorem statement
theorem simplification_and_exponent_sum :
  ∀ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 →
  (original_expression x y z = simplified_expression x y z) ∧
  (1 + 1 + 4 = 6) := by
  sorry

end NUMINAMATH_CALUDE_simplification_and_exponent_sum_l1689_168983


namespace NUMINAMATH_CALUDE_movie_outing_cost_is_36_l1689_168943

/-- Represents the cost of a movie outing for a family -/
def MovieOutingCost (ticket_price : ℚ) (popcorn_ratio : ℚ) (soda_ratio : ℚ) 
  (num_tickets : ℕ) (num_popcorn : ℕ) (num_soda : ℕ) : ℚ :=
  let popcorn_price := ticket_price * popcorn_ratio
  let soda_price := popcorn_price * soda_ratio
  (ticket_price * num_tickets) + (popcorn_price * num_popcorn) + (soda_price * num_soda)

/-- Theorem stating that the total cost for the family's movie outing is $36 -/
theorem movie_outing_cost_is_36 : 
  MovieOutingCost 5 (80/100) (50/100) 4 2 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_movie_outing_cost_is_36_l1689_168943


namespace NUMINAMATH_CALUDE_adjacent_diff_at_least_16_l1689_168915

/-- Represents a 6x6 grid with integers from 1 to 36 -/
def Grid := Fin 6 → Fin 6 → Fin 36

/-- Checks if two positions in the grid are adjacent -/
def adjacent (p1 p2 : Fin 6 × Fin 6) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p2.2 = p1.2 + 1)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p2.1 = p1.1 + 1))

/-- A valid grid satisfies the problem conditions -/
def valid_grid (g : Grid) : Prop :=
  (∀ i j, g i j ≤ 36) ∧
  (∃ i1 j1 i2 j2 i3 j3 i4 j4,
    g i1 j1 = 1 ∧ g i2 j2 = 2 ∧ g i3 j3 = 3 ∧ g i4 j4 = 4) ∧
  (∀ i j, g i j ≤ 4 → g i j ≥ 1) ∧
  (∀ i j, g i j > 4 → g i j ≤ 36)

/-- The main theorem -/
theorem adjacent_diff_at_least_16 (g : Grid) (h : valid_grid g) :
  ∃ p1 p2 : Fin 6 × Fin 6, adjacent p1 p2 ∧ |g p1.1 p1.2 - g p2.1 p2.2| ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_diff_at_least_16_l1689_168915


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1689_168919

/-- The line y = kx + 2k + 1 always passes through the point (-2, 1) for all real k. -/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), ((-2 : ℝ) * k + 2 * k + 1 = 1) := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1689_168919


namespace NUMINAMATH_CALUDE_distance_on_quadratic_curve_l1689_168951

/-- The distance between two points on a quadratic curve -/
theorem distance_on_quadratic_curve 
  (a b c p r : ℝ) : 
  let q := a * p^2 + b * p + c
  let s := a * r^2 + b * r + c
  Real.sqrt ((r - p)^2 + (s - q)^2) = |r - p| * Real.sqrt (1 + (a * (r + p) + b)^2) :=
by sorry

end NUMINAMATH_CALUDE_distance_on_quadratic_curve_l1689_168951


namespace NUMINAMATH_CALUDE_simplify_expression_l1689_168914

theorem simplify_expression (x : ℝ) : 3 - (2 - (1 + (2 * (1 - (3 - 2*x))))) = 8 - 4*x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1689_168914


namespace NUMINAMATH_CALUDE_square_of_negative_sqrt_two_l1689_168949

theorem square_of_negative_sqrt_two : (-Real.sqrt 2)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_sqrt_two_l1689_168949


namespace NUMINAMATH_CALUDE_sum_of_n_factorial_times_n_l1689_168945

def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem sum_of_n_factorial_times_n (a : ℕ) (h : factorial 1580 = a) :
  (Finset.range 1581).sum (λ n => n * factorial n) = 1581 * a - 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_n_factorial_times_n_l1689_168945


namespace NUMINAMATH_CALUDE_three_greater_than_negative_four_l1689_168902

theorem three_greater_than_negative_four : 3 > -4 := by
  sorry

end NUMINAMATH_CALUDE_three_greater_than_negative_four_l1689_168902


namespace NUMINAMATH_CALUDE_water_addition_theorem_l1689_168907

/-- Represents the amount of water that can be added to the 6-liter bucket --/
def water_to_add (bucket3 bucket5 bucket6 : ℕ) : ℕ :=
  bucket6 - (bucket5 - bucket3)

/-- Theorem stating the amount of water that can be added to the 6-liter bucket --/
theorem water_addition_theorem :
  water_to_add 3 5 6 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_water_addition_theorem_l1689_168907


namespace NUMINAMATH_CALUDE_constant_distance_l1689_168965

-- Define the ellipse E
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line l
def line (x y m : ℝ) : Prop := y = (1/2) * x + m

-- Define the constraint on m
def m_constraint (m : ℝ) : Prop := -Real.sqrt 2 < m ∧ m < Real.sqrt 2

-- Define the intersection points A and C
def intersection_points (xa ya xc yc m : ℝ) : Prop :=
  ellipse xa ya ∧ ellipse xc yc ∧ line xa ya m ∧ line xc yc m

-- Define the square ABCD
def square_ABCD (xa ya xb yb xc yc xd yd : ℝ) : Prop :=
  (xc - xa)^2 + (yc - ya)^2 = (xd - xb)^2 + (yd - yb)^2 ∧
  (xb - xa)^2 + (yb - ya)^2 = (xd - xc)^2 + (yd - yc)^2

-- Define point N
def point_N (xn m : ℝ) : Prop := xn = -2 * m

-- Main theorem
theorem constant_distance
  (m xa ya xb yb xc yc xd yd xn : ℝ)
  (h_m : m_constraint m)
  (h_int : intersection_points xa ya xc yc m)
  (h_square : square_ABCD xa ya xb yb xc yc xd yd)
  (h_N : point_N xn m) :
  (xb - xn)^2 + yb^2 = 5/2 :=
sorry

end NUMINAMATH_CALUDE_constant_distance_l1689_168965


namespace NUMINAMATH_CALUDE_third_height_less_than_30_l1689_168968

/-- Given a triangle with two heights of 12 and 20, prove that the third height is less than 30. -/
theorem third_height_less_than_30 
  (h_a h_b h_c : ℝ) 
  (triangle_exists : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b)
  (height_a : h_a = 12)
  (height_b : h_b = 20) :
  h_c < 30 := by
sorry

end NUMINAMATH_CALUDE_third_height_less_than_30_l1689_168968


namespace NUMINAMATH_CALUDE_gcd_lcm_theorem_l1689_168927

theorem gcd_lcm_theorem : 
  (Nat.gcd 42 63 = 21 ∧ Nat.lcm 42 63 = 126) ∧ 
  (Nat.gcd 8 20 = 4 ∧ Nat.lcm 8 20 = 40) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_theorem_l1689_168927


namespace NUMINAMATH_CALUDE_direction_vector_of_bisecting_line_l1689_168921

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 2) + 2

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y = 0

-- Define what it means for a line to bisect a circle
def bisects (k : ℝ) : Prop := ∃ (x₀ y₀ : ℝ), circle_C x₀ y₀ ∧ line_l k x₀ y₀

-- Theorem statement
theorem direction_vector_of_bisecting_line :
  ∃ (k : ℝ), bisects k → ∃ (t : ℝ), t ≠ 0 ∧ (2 = t * 2 ∧ 2 = t * 2) :=
sorry

end NUMINAMATH_CALUDE_direction_vector_of_bisecting_line_l1689_168921


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l1689_168922

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 2 / 3)
  (hdb : d / b = 1 / 5) : 
  a / c = 75 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l1689_168922


namespace NUMINAMATH_CALUDE_platform_length_l1689_168966

/-- The length of a platform given train parameters -/
theorem platform_length
  (train_length : ℝ)
  (train_speed_kmph : ℝ)
  (time_to_cross : ℝ)
  (h1 : train_length = 450)
  (h2 : train_speed_kmph = 126)
  (h3 : time_to_cross = 20)
  : ∃ (platform_length : ℝ), platform_length = 250 := by
  sorry

#check platform_length

end NUMINAMATH_CALUDE_platform_length_l1689_168966


namespace NUMINAMATH_CALUDE_jujube_sales_theorem_l1689_168933

/-- Represents the daily sales deviation from the planned amount -/
def daily_deviations : List Int := [4, -3, -5, 14, -8, 21, -6]

/-- The planned daily sales amount in pounds -/
def planned_daily_sales : Nat := 100

/-- The selling price per pound in yuan -/
def selling_price : Nat := 8

/-- The freight cost per pound in yuan -/
def freight_cost : Nat := 3

theorem jujube_sales_theorem :
  /- Total amount sold in first three days -/
  (List.take 3 daily_deviations).sum + 3 * planned_daily_sales = 296 ∧
  /- Total earnings for the week -/
  (daily_deviations.sum + 7 * planned_daily_sales) * (selling_price - freight_cost) = 3585 := by
  sorry

end NUMINAMATH_CALUDE_jujube_sales_theorem_l1689_168933


namespace NUMINAMATH_CALUDE_square_comparison_l1689_168926

theorem square_comparison (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a * b * Real.sqrt (a^2 + b^2)) / (a^2 + a * b + b^2) < (a * b) / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_square_comparison_l1689_168926


namespace NUMINAMATH_CALUDE_choir_theorem_l1689_168924

def choir_problem (original_size absent first_fraction second_fraction third_fraction fourth_fraction late_arrivals : ℕ) : Prop :=
  let present := original_size - absent
  let first_verse := present / 2
  let second_verse := (present - first_verse) / 3
  let third_verse := (present - first_verse - second_verse) / 4
  let fourth_verse := (present - first_verse - second_verse - third_verse) / 5
  let total_before_fifth := first_verse + second_verse + third_verse + fourth_verse + late_arrivals
  total_before_fifth + (present - total_before_fifth) = present

theorem choir_theorem :
  choir_problem 70 10 2 3 4 5 5 :=
sorry

end NUMINAMATH_CALUDE_choir_theorem_l1689_168924


namespace NUMINAMATH_CALUDE_suv_distance_theorem_l1689_168994

/-- Represents the maximum distance an SUV can travel on 24 gallons of gas -/
def max_distance (x : ℝ) : ℝ :=
  1.824 * x + 292.8 - 2.928 * x

/-- Theorem stating the maximum distance formula for the SUV -/
theorem suv_distance_theorem (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 100) :
  max_distance x = 7.6 * (x / 100) * 24 + 12.2 * ((100 - x) / 100) * 24 :=
by sorry

end NUMINAMATH_CALUDE_suv_distance_theorem_l1689_168994


namespace NUMINAMATH_CALUDE_total_cost_of_tickets_l1689_168973

def total_tickets : ℕ := 29
def cheap_ticket_price : ℕ := 7
def expensive_ticket_price : ℕ := 9
def expensive_tickets : ℕ := 11

theorem total_cost_of_tickets : 
  cheap_ticket_price * (total_tickets - expensive_tickets) + 
  expensive_ticket_price * expensive_tickets = 225 := by sorry

end NUMINAMATH_CALUDE_total_cost_of_tickets_l1689_168973


namespace NUMINAMATH_CALUDE_intersection_of_S_and_T_l1689_168916

-- Define the sets S and T
def S : Set ℝ := {x : ℝ | x ≥ 2}
def T : Set ℝ := {x : ℝ | x ≤ 5}

-- State the theorem
theorem intersection_of_S_and_T : S ∩ T = Set.Icc 2 5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_S_and_T_l1689_168916


namespace NUMINAMATH_CALUDE_red_yellow_difference_l1689_168992

/-- Represents the number of marbles of each color in a bowl. -/
structure MarbleCount where
  total : ℕ
  yellow : ℕ
  blue : ℕ
  red : ℕ

/-- Represents the ratio of blue to red marbles. -/
structure BlueRedRatio where
  blue : ℕ
  red : ℕ

/-- Given the total number of marbles, the number of yellow marbles, and the ratio of blue to red marbles,
    proves that there are 3 more red marbles than yellow marbles. -/
theorem red_yellow_difference (m : MarbleCount) (ratio : BlueRedRatio) : 
  m.total = 19 → 
  m.yellow = 5 → 
  m.blue + m.red = m.total - m.yellow →
  m.blue * ratio.red = m.red * ratio.blue →
  ratio.blue = 3 →
  ratio.red = 4 →
  m.red - m.yellow = 3 := by
  sorry

end NUMINAMATH_CALUDE_red_yellow_difference_l1689_168992


namespace NUMINAMATH_CALUDE_orchid_rose_difference_is_nine_l1689_168925

/-- Flower quantities and ratios in a vase --/
structure FlowerVase where
  initial_roses : ℕ
  initial_orchids : ℕ
  initial_tulips : ℕ
  final_roses : ℕ
  final_orchids : ℕ
  final_tulips : ℕ
  rose_orchid_ratio : ℚ
  rose_tulip_ratio : ℚ

/-- The difference between orchids and roses after adding new flowers --/
def orchid_rose_difference (v : FlowerVase) : ℕ :=
  v.final_orchids - v.final_roses

/-- Theorem stating the difference between orchids and roses is 9 --/
theorem orchid_rose_difference_is_nine (v : FlowerVase)
  (h1 : v.initial_roses = 7)
  (h2 : v.initial_orchids = 12)
  (h3 : v.initial_tulips = 5)
  (h4 : v.final_roses = 11)
  (h5 : v.final_orchids = 20)
  (h6 : v.final_tulips = 10)
  (h7 : v.rose_orchid_ratio = 2/5)
  (h8 : v.rose_tulip_ratio = 3/5) :
  orchid_rose_difference v = 9 := by
  sorry

#eval orchid_rose_difference {
  initial_roses := 7,
  initial_orchids := 12,
  initial_tulips := 5,
  final_roses := 11,
  final_orchids := 20,
  final_tulips := 10,
  rose_orchid_ratio := 2/5,
  rose_tulip_ratio := 3/5
}

end NUMINAMATH_CALUDE_orchid_rose_difference_is_nine_l1689_168925


namespace NUMINAMATH_CALUDE_rectangle_formations_l1689_168956

def horizontal_lines : ℕ := 5
def vertical_lines : ℕ := 6

theorem rectangle_formations : 
  (Nat.choose horizontal_lines 2) * (Nat.choose vertical_lines 2) = 150 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_formations_l1689_168956


namespace NUMINAMATH_CALUDE_g_minus_one_eq_zero_l1689_168963

/-- The polynomial function g(x) -/
def g (s : ℝ) (x : ℝ) : ℝ := 3 * x^4 + 2 * x^3 - x^2 - 4 * x + s

/-- Theorem stating that g(-1) = 0 when s = -4 -/
theorem g_minus_one_eq_zero : g (-4) (-1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_minus_one_eq_zero_l1689_168963


namespace NUMINAMATH_CALUDE_parabola_vertex_l1689_168942

/-- The parabola defined by y = -x^2 + cx + d -/
noncomputable def parabola (c d : ℝ) (x : ℝ) : ℝ := -x^2 + c*x + d

/-- The set of x values satisfying the inequality -x^2 + cx + d ≤ 0 -/
def inequality_solution (c d : ℝ) : Set ℝ := {x | x ∈ Set.Icc (-7) 3 ∪ Set.Ici 9}

/-- The vertex of a parabola -/
structure Vertex where
  x : ℝ
  y : ℝ

theorem parabola_vertex (c d : ℝ) :
  (inequality_solution c d = {x | x ∈ Set.Icc (-7) 3 ∪ Set.Ici 9}) →
  (∃ (vertex : Vertex), vertex.x = 1 ∧ vertex.y = -62 ∧
    ∀ (x : ℝ), parabola c d x ≤ parabola c d vertex.x) :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1689_168942


namespace NUMINAMATH_CALUDE_f_neg_nine_equals_neg_three_l1689_168904

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_neg_nine_equals_neg_three
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_pos : ∀ x > 0, f x = Real.sqrt x) :
  f (-9) = -3 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_nine_equals_neg_three_l1689_168904


namespace NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l1689_168930

theorem min_value_sum_of_reciprocals (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_6 : a + b + c = 6) : 
  (9 / a) + (16 / b) + (25 / c) ≥ 24 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l1689_168930


namespace NUMINAMATH_CALUDE_initial_number_of_persons_l1689_168908

theorem initial_number_of_persons (N : ℕ) 
  (h1 : ∃ (avg : ℝ), N * (avg + 5) - N * avg = 105 - 65) : N = 8 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_of_persons_l1689_168908


namespace NUMINAMATH_CALUDE_P_on_y_axis_P_in_first_quadrant_with_distance_condition_l1689_168952

-- Define point P
def P (m : ℝ) : ℝ × ℝ := (8 - 2*m, m + 1)

-- Part 1: P lies on y-axis
theorem P_on_y_axis (m : ℝ) : 
  P m = (0, m + 1) → m = 4 := by sorry

-- Part 2: P in first quadrant with distance condition
theorem P_in_first_quadrant_with_distance_condition (m : ℝ) :
  (8 - 2*m > 0 ∧ m + 1 > 0) ∧ (m + 1 = 2*(8 - 2*m)) → P m = (2, 4) := by sorry

end NUMINAMATH_CALUDE_P_on_y_axis_P_in_first_quadrant_with_distance_condition_l1689_168952


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1689_168984

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (x - 2)^9 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
    a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10 + a₁₁*x^11) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = 510 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1689_168984


namespace NUMINAMATH_CALUDE_vocational_students_form_valid_set_l1689_168991

-- Define the universe of discourse
def Student : Type := String

-- Define the properties
def isDefinite (s : Set Student) : Prop := sorry
def isDistinct (s : Set Student) : Prop := sorry
def isUnordered (s : Set Student) : Prop := sorry

-- Define the sets corresponding to each option
def tallStudents : Set Student := sorry
def vocationalStudents : Set Student := sorry
def goodStudents : Set Student := sorry
def lushTrees : Set Student := sorry

-- Define what makes a valid set
def isValidSet (s : Set Student) : Prop :=
  isDefinite s ∧ isDistinct s ∧ isUnordered s

-- Theorem statement
theorem vocational_students_form_valid_set :
  isValidSet vocationalStudents ∧
  ¬isValidSet tallStudents ∧
  ¬isValidSet goodStudents ∧
  ¬isValidSet lushTrees :=
sorry

end NUMINAMATH_CALUDE_vocational_students_form_valid_set_l1689_168991


namespace NUMINAMATH_CALUDE_visited_both_countries_l1689_168969

theorem visited_both_countries (total : ℕ) (iceland : ℕ) (norway : ℕ) (neither : ℕ) : 
  total = 100 → iceland = 55 → norway = 43 → neither = 63 → 
  (total - neither) = (iceland + norway - (iceland + norway - (total - neither))) := by
  sorry

end NUMINAMATH_CALUDE_visited_both_countries_l1689_168969


namespace NUMINAMATH_CALUDE_typing_task_correct_characters_l1689_168932

/-- The total number of characters in the typing task -/
def total_characters : ℕ := 10000

/-- Xiaoyuan's error rate: 1 mistake per 10 characters -/
def xiaoyuan_error_rate : ℚ := 1 / 10

/-- Xiaofang's error rate: 2 mistakes per 10 characters -/
def xiaofang_error_rate : ℚ := 2 / 10

/-- The ratio of correct characters typed by Xiaoyuan to Xiaofang -/
def correct_ratio : ℕ := 2

theorem typing_task_correct_characters :
  ∃ (xiaoyuan_correct xiaofang_correct : ℕ),
    xiaoyuan_correct + xiaofang_correct = 8640 ∧
    xiaoyuan_correct = 2 * xiaofang_correct ∧
    xiaoyuan_correct = total_characters * (1 - xiaoyuan_error_rate) ∧
    xiaofang_correct = total_characters * (1 - xiaofang_error_rate) :=
sorry

end NUMINAMATH_CALUDE_typing_task_correct_characters_l1689_168932


namespace NUMINAMATH_CALUDE_isosceles_triangle_rectangle_equal_area_l1689_168912

/-- Given an isosceles triangle and a rectangle with the same area, 
    prove that the height of the triangle is twice the breadth of the rectangle. -/
theorem isosceles_triangle_rectangle_equal_area 
  (l b h : ℝ) (hl : l > 0) (hb : b > 0) (hlb : l > b) : 
  (1 / 2 : ℝ) * l * h = l * b → h = 2 * b := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_rectangle_equal_area_l1689_168912


namespace NUMINAMATH_CALUDE_game_result_l1689_168981

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 ∧ n % 5 = 0 then 7
  else if n % 3 = 0 then 3
  else 0

def charlie_rolls : List ℕ := [6, 2, 3, 5]
def dana_rolls : List ℕ := [5, 3, 1, 3]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map g |>.sum

theorem game_result : total_points charlie_rolls * total_points dana_rolls = 36 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l1689_168981


namespace NUMINAMATH_CALUDE_subtraction_theorem_l1689_168903

/-- Represents a four-digit number --/
structure FourDigitNumber where
  thousands : Nat
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_digits : thousands < 10 ∧ hundreds < 10 ∧ tens < 10 ∧ ones < 10

/-- The result of subtracting two four-digit numbers --/
structure SubtractionResult where
  thousands : Int
  hundreds : Int
  tens : Int
  ones : Int

def subtract (minuend subtrahend : FourDigitNumber) : SubtractionResult :=
  sorry

theorem subtraction_theorem (a b c d : Nat) 
  (h_digits : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) :
  let minuend : FourDigitNumber := ⟨a, b, c, d, h_digits⟩
  let subtrahend : FourDigitNumber := ⟨d, b, a, c, sorry⟩
  let result := subtract minuend subtrahend
  (result.hundreds = 7 ∧ minuend.thousands ≥ subtrahend.thousands) →
  result.thousands = 9 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_theorem_l1689_168903


namespace NUMINAMATH_CALUDE_basketball_team_enrollment_l1689_168917

theorem basketball_team_enrollment (total_players : ℕ) 
  (physics_enrollment : ℕ) (both_enrollment : ℕ) :
  total_players = 15 →
  physics_enrollment = 9 →
  both_enrollment = 3 →
  physics_enrollment + (total_players - physics_enrollment) ≥ total_players →
  total_players - physics_enrollment + both_enrollment = 9 :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_enrollment_l1689_168917


namespace NUMINAMATH_CALUDE_intersection_subset_l1689_168954

theorem intersection_subset (A B : Set α) : A ∩ B = B → B ⊆ A := by
  sorry

end NUMINAMATH_CALUDE_intersection_subset_l1689_168954


namespace NUMINAMATH_CALUDE_M_greater_than_N_l1689_168997

theorem M_greater_than_N (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) : a * b > a + b - 1 := by
  sorry

end NUMINAMATH_CALUDE_M_greater_than_N_l1689_168997


namespace NUMINAMATH_CALUDE_abs_diff_positive_iff_not_equal_l1689_168976

theorem abs_diff_positive_iff_not_equal (x : ℝ) : x ≠ 3 ↔ |x - 3| > 0 := by sorry

end NUMINAMATH_CALUDE_abs_diff_positive_iff_not_equal_l1689_168976


namespace NUMINAMATH_CALUDE_spring_decrease_percentage_l1689_168911

theorem spring_decrease_percentage 
  (fall_increase : Real) 
  (total_change : Real) 
  (h1 : fall_increase = 0.08) 
  (h2 : total_change = -0.1252) : 
  let initial := 100
  let after_fall := initial * (1 + fall_increase)
  let after_spring := initial * (1 + total_change)
  (after_fall - after_spring) / after_fall = 0.19 := by
sorry

end NUMINAMATH_CALUDE_spring_decrease_percentage_l1689_168911


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_l1689_168999

noncomputable def f (x : ℝ) : ℝ := (x^5 + 1) / (x^4 + 1)

theorem tangent_line_at_x_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := deriv f x₀
  ∀ x y : ℝ, y = m * (x - x₀) + y₀ → y = (1/2) * x + 1/2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_l1689_168999


namespace NUMINAMATH_CALUDE_cde_value_l1689_168971

/-- Represents the digits in the coding system -/
inductive Digit
| A | B | C | D | E | F

/-- Converts a Digit to its corresponding integer value -/
def digit_to_int : Digit → Nat
| Digit.A => 0
| Digit.B => 5
| Digit.C => 0
| Digit.D => 1
| Digit.E => 0
| Digit.F => 5

/-- Represents a number in the coding system -/
structure EncodedNumber :=
  (hundreds : Digit)
  (tens : Digit)
  (ones : Digit)

/-- Converts an EncodedNumber to its base-10 value -/
def to_base_10 (n : EncodedNumber) : Nat :=
  6^2 * (digit_to_int n.hundreds) + 6 * (digit_to_int n.tens) + (digit_to_int n.ones)

/-- States that BCF, BCE, CAA are consecutive integers -/
axiom consecutive_encoding :
  ∃ (n : Nat),
    to_base_10 (EncodedNumber.mk Digit.B Digit.C Digit.F) = n ∧
    to_base_10 (EncodedNumber.mk Digit.B Digit.C Digit.E) = n + 1 ∧
    to_base_10 (EncodedNumber.mk Digit.C Digit.A Digit.A) = n + 2

theorem cde_value :
  to_base_10 (EncodedNumber.mk Digit.C Digit.D Digit.E) = 6 :=
by sorry

end NUMINAMATH_CALUDE_cde_value_l1689_168971


namespace NUMINAMATH_CALUDE_pear_sales_l1689_168900

theorem pear_sales (morning_sales afternoon_sales total_sales : ℕ) : 
  afternoon_sales = 2 * morning_sales →
  total_sales = morning_sales + afternoon_sales →
  total_sales = 390 →
  afternoon_sales = 260 := by
sorry

end NUMINAMATH_CALUDE_pear_sales_l1689_168900


namespace NUMINAMATH_CALUDE_negation_of_all_positive_square_plus_one_l1689_168964

theorem negation_of_all_positive_square_plus_one (q : Prop) : 
  (q ↔ ∀ x : ℝ, x^2 + 1 > 0) →
  (¬q ↔ ∃ x : ℝ, x^2 + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_positive_square_plus_one_l1689_168964


namespace NUMINAMATH_CALUDE_gcd_8251_6105_l1689_168975

theorem gcd_8251_6105 : Int.gcd 8251 6105 = 39 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8251_6105_l1689_168975


namespace NUMINAMATH_CALUDE_halfway_fraction_l1689_168962

theorem halfway_fraction (a b : ℚ) (ha : a = 1/6) (hb : b = 1/4) :
  (a + b) / 2 = 5/24 := by
  sorry

end NUMINAMATH_CALUDE_halfway_fraction_l1689_168962


namespace NUMINAMATH_CALUDE_solution_set_part_i_range_of_a_part_ii_l1689_168941

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Theorem for part (I)
theorem solution_set_part_i :
  let a : ℝ := -2
  let S := {x : ℝ | f a x + f a (2 * x) > 2}
  S = {x : ℝ | x < -2 ∨ x > -2/3} :=
sorry

-- Theorem for part (II)
theorem range_of_a_part_ii :
  ∀ a : ℝ, a < 0 →
  (∃ x : ℝ, f a x + f a (2 * x) < 1/2) →
  -1 < a ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_solution_set_part_i_range_of_a_part_ii_l1689_168941


namespace NUMINAMATH_CALUDE_scalar_multiplication_distributivity_l1689_168972

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem scalar_multiplication_distributivity
  (m n : ℝ) (a : V)
  (hm : m ≠ 0) (hn : n ≠ 0) (ha : a ≠ 0) :
  (m + n) • a = m • a + n • a :=
by sorry

end NUMINAMATH_CALUDE_scalar_multiplication_distributivity_l1689_168972


namespace NUMINAMATH_CALUDE_line_equation_l1689_168989

/-- Circle P: x^2 + y^2 - 4y = 0 -/
def circleP (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*y = 0

/-- Parabola S: y = x^2 / 8 -/
def parabolaS (x y : ℝ) : Prop :=
  y = x^2 / 8

/-- Line l: y = k*x + b -/
def lineL (k b x y : ℝ) : Prop :=
  y = k*x + b

/-- Center of circle P -/
def centerP : ℝ × ℝ := (0, 2)

/-- Line l passes through the center of circle P -/
def lineThroughCenter (k b : ℝ) : Prop :=
  lineL k b (centerP.1) (centerP.2)

/-- Four intersection points of line l with circle P and parabola S -/
structure IntersectionPoints (k b : ℝ) :=
  (A B C D : ℝ × ℝ)
  (intersectCircleP : circleP A.1 A.2 ∧ circleP B.1 B.2 ∧ circleP C.1 C.2 ∧ circleP D.1 D.2)
  (intersectParabolaS : parabolaS A.1 A.2 ∧ parabolaS B.1 B.2 ∧ parabolaS C.1 C.2 ∧ parabolaS D.1 D.2)
  (onLineL : lineL k b A.1 A.2 ∧ lineL k b B.1 B.2 ∧ lineL k b C.1 C.2 ∧ lineL k b D.1 D.2)
  (leftToRight : A.1 < B.1 ∧ B.1 < C.1 ∧ C.1 < D.1)

/-- Lengths of segments AB, BC, CD form an arithmetic sequence -/
def arithmeticSequence (A B C D : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let CD := Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)
  BC - AB = CD - BC

theorem line_equation :
  ∀ k b : ℝ,
  lineThroughCenter k b →
  (∃ pts : IntersectionPoints k b, arithmeticSequence pts.A pts.B pts.C pts.D) →
  (k = -Real.sqrt 2 / 2 ∨ k = Real.sqrt 2 / 2) ∧ b = 2 :=
sorry

end NUMINAMATH_CALUDE_line_equation_l1689_168989


namespace NUMINAMATH_CALUDE_percentage_50_59_is_100_over_9_l1689_168934

/-- Represents the frequency distribution of test scores -/
structure ScoreDistribution :=
  (score_90_100 : ℕ)
  (score_80_89 : ℕ)
  (score_70_79 : ℕ)
  (score_60_69 : ℕ)
  (score_50_59 : ℕ)
  (score_below_50 : ℕ)

/-- The actual score distribution from Ms. Garcia's geometry class -/
def garcia_distribution : ScoreDistribution :=
  { score_90_100 := 5,
    score_80_89 := 7,
    score_70_79 := 9,
    score_60_69 := 8,
    score_50_59 := 4,
    score_below_50 := 3 }

/-- Calculate the total number of students -/
def total_students (d : ScoreDistribution) : ℕ :=
  d.score_90_100 + d.score_80_89 + d.score_70_79 + d.score_60_69 + d.score_50_59 + d.score_below_50

/-- Calculate the percentage of students in the 50%-59% range -/
def percentage_50_59 (d : ScoreDistribution) : ℚ :=
  (d.score_50_59 : ℚ) / (total_students d : ℚ) * 100

/-- Theorem stating that the percentage of students who scored in the 50%-59% range is 100/9 -/
theorem percentage_50_59_is_100_over_9 :
  percentage_50_59 garcia_distribution = 100 / 9 := by
  sorry


end NUMINAMATH_CALUDE_percentage_50_59_is_100_over_9_l1689_168934


namespace NUMINAMATH_CALUDE_difference_of_squares_l1689_168936

theorem difference_of_squares (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1689_168936


namespace NUMINAMATH_CALUDE_P_investment_theorem_l1689_168996

-- Define the investments and profit ratio
def Q_investment : ℕ := 60000
def profit_ratio_P : ℕ := 2
def profit_ratio_Q : ℕ := 3

-- Theorem to prove P's investment
theorem P_investment_theorem :
  ∃ P_investment : ℕ,
    P_investment * profit_ratio_Q = Q_investment * profit_ratio_P ∧
    P_investment = 40000 := by
  sorry

end NUMINAMATH_CALUDE_P_investment_theorem_l1689_168996


namespace NUMINAMATH_CALUDE_largest_n_for_product_2010_l1689_168940

def is_arithmetic_sequence (s : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, s (n + 1) - s n = d

theorem largest_n_for_product_2010 (a b : ℕ → ℤ) 
  (ha : is_arithmetic_sequence a) 
  (hb : is_arithmetic_sequence b)
  (h1 : a 1 = 1 ∧ b 1 = 1)
  (h2 : a 2 ≤ b 2)
  (h3 : ∃ n : ℕ, a n * b n = 2010)
  : (∃ n : ℕ, a n * b n = 2010 ∧ ∀ m : ℕ, a m * b m = 2010 → m ≤ n) ∧
    (∀ n : ℕ, a n * b n = 2010 → n ≤ 8) :=
sorry

end NUMINAMATH_CALUDE_largest_n_for_product_2010_l1689_168940


namespace NUMINAMATH_CALUDE_solve_for_a_l1689_168980

def A (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b = 0}

def B : Set ℝ := {3}

theorem solve_for_a (a b : ℝ) (h : A a b = B) : a = -6 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l1689_168980


namespace NUMINAMATH_CALUDE_quadratic_root_implies_a_l1689_168937

theorem quadratic_root_implies_a (a : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x - a = 0) ∧ ((-1)^2 - 3*(-1) - a = 0) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_a_l1689_168937


namespace NUMINAMATH_CALUDE_no_integer_tangent_length_l1689_168939

/-- A circle with a point P outside it, from which a tangent and a secant are drawn -/
structure CircleWithExternalPoint where
  /-- The circumference of the circle -/
  circumference : ℝ
  /-- The length of one arc created by the secant -/
  m : ℕ
  /-- The length of the tangent from P to the circle -/
  t₁ : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (c : CircleWithExternalPoint) : Prop :=
  c.circumference = 15 * Real.pi ∧
  c.t₁ * c.t₁ = c.m * (c.circumference - c.m)

/-- The theorem stating that no integer values of t₁ satisfy the conditions -/
theorem no_integer_tangent_length :
  ¬∃ c : CircleWithExternalPoint, satisfiesConditions c :=
sorry

end NUMINAMATH_CALUDE_no_integer_tangent_length_l1689_168939


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l1689_168906

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := sorry

-- State the theorem
theorem z_in_second_quadrant :
  (1 + i) * z = -1 →
  z.re < 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l1689_168906


namespace NUMINAMATH_CALUDE_petya_vasya_meeting_l1689_168977

/-- Represents the meeting point of two people walking towards each other along a line of lamps -/
def meeting_point (total_lamps : ℕ) (p_start p_pos : ℕ) (v_start v_pos : ℕ) : ℕ :=
  let p_traveled := p_pos - p_start
  let v_traveled := v_start - v_pos
  let remaining_distance := v_pos - p_pos
  let total_intervals := total_lamps - 1
  let p_speed := p_traveled
  let v_speed := v_traveled
  p_pos + (remaining_distance * p_speed) / (p_speed + v_speed)

/-- Theorem stating that Petya and Vasya meet at lamp 64 -/
theorem petya_vasya_meeting :
  let total_lamps : ℕ := 100
  let petya_start : ℕ := 1
  let vasya_start : ℕ := 100
  let petya_position : ℕ := 22
  let vasya_position : ℕ := 88
  meeting_point total_lamps petya_start petya_position vasya_start vasya_position = 64 := by
  sorry

#eval meeting_point 100 1 22 100 88

end NUMINAMATH_CALUDE_petya_vasya_meeting_l1689_168977


namespace NUMINAMATH_CALUDE_aras_height_is_55_l1689_168931

/-- Calculates Ara's current height given the conditions of the problem -/
def aras_current_height (original_height : ℝ) (sheas_growth_rate : ℝ) (sheas_current_height : ℝ) (aras_growth_fraction : ℝ) : ℝ :=
  let sheas_growth := sheas_current_height - original_height
  let aras_growth := aras_growth_fraction * sheas_growth
  original_height + aras_growth

/-- The theorem stating that Ara's current height is 55 inches -/
theorem aras_height_is_55 :
  let original_height := 50
  let sheas_growth_rate := 0.3
  let sheas_current_height := 65
  let aras_growth_fraction := 1/3
  aras_current_height original_height sheas_growth_rate sheas_current_height aras_growth_fraction = 55 := by
  sorry


end NUMINAMATH_CALUDE_aras_height_is_55_l1689_168931


namespace NUMINAMATH_CALUDE_quartic_root_inequality_l1689_168928

theorem quartic_root_inequality (a b : ℝ) :
  (∃ x : ℝ, x^4 - a*x^3 + 2*x^2 - b*x + 1 = 0) → a^2 + b^2 ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_quartic_root_inequality_l1689_168928


namespace NUMINAMATH_CALUDE_product_divisible_by_twelve_l1689_168993

theorem product_divisible_by_twelve (a b c d : ℤ) : 
  ∃ k : ℤ, (b - a) * (c - a) * (d - a) * (b - c) * (d - c) * (d - b) = 12 * k := by
  sorry

end NUMINAMATH_CALUDE_product_divisible_by_twelve_l1689_168993


namespace NUMINAMATH_CALUDE_gcd_digits_bound_l1689_168982

theorem gcd_digits_bound (a b : ℕ) : 
  (10^6 ≤ a ∧ a < 10^7) →
  (10^6 ≤ b ∧ b < 10^7) →
  (10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) →
  Nat.gcd a b < 10^3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_digits_bound_l1689_168982


namespace NUMINAMATH_CALUDE_value_of_a_l1689_168923

theorem value_of_a (a b : ℚ) (h1 : b/a = 4) (h2 : b = 15 - 4*a) : a = 15/8 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l1689_168923


namespace NUMINAMATH_CALUDE_reptile_insect_consumption_l1689_168961

theorem reptile_insect_consumption :
  let num_geckos : ℕ := 5
  let num_lizards : ℕ := 3
  let num_chameleons : ℕ := 4
  let num_iguanas : ℕ := 2
  let gecko_consumption : ℕ := 6
  let lizard_consumption : ℝ := 2 * gecko_consumption
  let chameleon_consumption : ℝ := 3.5 * gecko_consumption
  let iguana_consumption : ℝ := 0.75 * gecko_consumption
  
  (num_geckos * gecko_consumption : ℝ) +
  (num_lizards : ℝ) * lizard_consumption +
  (num_chameleons : ℝ) * chameleon_consumption +
  (num_iguanas : ℝ) * iguana_consumption = 159
  := by sorry

end NUMINAMATH_CALUDE_reptile_insect_consumption_l1689_168961


namespace NUMINAMATH_CALUDE_neither_sufficient_nor_necessary_l1689_168967

theorem neither_sufficient_nor_necessary (p q : Prop) :
  ¬(((p ∨ q) → ¬(p ∧ q)) ∧ (¬(p ∧ q) → (p ∨ q))) :=
by sorry

end NUMINAMATH_CALUDE_neither_sufficient_nor_necessary_l1689_168967


namespace NUMINAMATH_CALUDE_inequality_condition_l1689_168913

theorem inequality_condition (x y : ℝ) : 
  y - x < Real.sqrt (x^2 + 4*x*y) ↔ 
  ((y < x + Real.sqrt (x^2 + 4*x*y) ∨ y < x - Real.sqrt (x^2 + 4*x*y)) ∧ x*(x + 4*y) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l1689_168913


namespace NUMINAMATH_CALUDE_runs_scored_for_new_average_l1689_168920

/-- Represents a cricket player's statistics -/
structure CricketPlayer where
  matches_played : ℕ
  total_runs : ℕ

/-- Calculate the batting average of a player -/
def batting_average (player : CricketPlayer) : ℚ :=
  player.total_runs / player.matches_played

/-- Calculate the total runs after a new match -/
def total_runs_after_match (player : CricketPlayer) (new_runs : ℕ) : ℕ :=
  player.total_runs + new_runs

/-- Calculate the new batting average after a match -/
def new_batting_average (player : CricketPlayer) (new_runs : ℕ) : ℚ :=
  (total_runs_after_match player new_runs) / (player.matches_played + 1)

theorem runs_scored_for_new_average 
  (player : CricketPlayer) 
  (new_runs : ℕ) :
  player.matches_played = 5 ∧ 
  batting_average player = 51 ∧
  new_batting_average player new_runs = 54 →
  new_runs = 69 := by
sorry

end NUMINAMATH_CALUDE_runs_scored_for_new_average_l1689_168920


namespace NUMINAMATH_CALUDE_find_f_2022_l1689_168978

/-- A function satisfying the given condition -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f ((a + 2 * b) / 3) = (f a + 2 * f b) / 3

/-- The theorem to prove -/
theorem find_f_2022 (f : ℝ → ℝ) (h : special_function f) (h1 : f 1 = 5) (h4 : f 4 = 2) :
  f 2022 = -2016 := by
  sorry


end NUMINAMATH_CALUDE_find_f_2022_l1689_168978


namespace NUMINAMATH_CALUDE_two_medians_not_unique_l1689_168938

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the concept of a median
def Median (t : Triangle) : ℝ → Prop := sorry

-- Define the concept of uniquely determining a triangle's shape
def UniquelyDeterminesShape (data : Set (Triangle → Prop)) : Prop := sorry

-- Define the five sets of data
def TwoSidesIncludedAngle : Set (Triangle → Prop) := sorry
def ThreeSides : Set (Triangle → Prop) := sorry
def TwoMedians : Set (Triangle → Prop) := sorry
def OneAltitudeAndBase : Set (Triangle → Prop) := sorry
def TwoAngles : Set (Triangle → Prop) := sorry

-- Theorem statement
theorem two_medians_not_unique :
  UniquelyDeterminesShape TwoSidesIncludedAngle ∧
  UniquelyDeterminesShape ThreeSides ∧
  ¬UniquelyDeterminesShape TwoMedians ∧
  UniquelyDeterminesShape OneAltitudeAndBase ∧
  UniquelyDeterminesShape TwoAngles :=
sorry

end NUMINAMATH_CALUDE_two_medians_not_unique_l1689_168938


namespace NUMINAMATH_CALUDE_max_power_of_five_equals_three_l1689_168988

/-- The number of divisors of a positive integer -/
noncomputable def num_divisors (n : ℕ+) : ℕ := sorry

/-- The greatest integer j such that 5^j divides n -/
noncomputable def max_power_of_five (n : ℕ+) : ℕ := sorry

theorem max_power_of_five_equals_three (n : ℕ+) 
  (h1 : num_divisors n = 72)
  (h2 : num_divisors (5 * n) = 90) :
  max_power_of_five n = 3 := by sorry

end NUMINAMATH_CALUDE_max_power_of_five_equals_three_l1689_168988


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1689_168958

theorem constant_term_expansion (x : ℝ) : 
  let expansion := (5 * x + 1 / (3 * x)) ^ 8
  ∃ (p q : ℝ → ℝ), expansion = p x + (43750 / 81) + q x ∧ 
    (∀ y, y ≠ 0 → p y + q y = 0) :=
by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1689_168958
