import Mathlib

namespace square_table_capacity_square_table_capacity_proof_l1789_178900

theorem square_table_capacity (rectangular_tables : ℕ) (rectangular_capacity : ℕ) 
  (square_tables : ℕ) (total_pupils : ℕ) : ℕ :=
  let remaining_pupils := total_pupils - rectangular_tables * rectangular_capacity
  remaining_pupils / square_tables

#check square_table_capacity 7 10 5 90 = 4

theorem square_table_capacity_proof 
  (h1 : rectangular_tables = 7)
  (h2 : rectangular_capacity = 10)
  (h3 : square_tables = 5)
  (h4 : total_pupils = 90) :
  square_table_capacity rectangular_tables rectangular_capacity square_tables total_pupils = 4 := by
  sorry

end square_table_capacity_square_table_capacity_proof_l1789_178900


namespace parabola_focus_directrix_distance_l1789_178935

/-- For a parabola with equation y^2 = ax, if the distance from its focus to its directrix is 2, then a = 4. -/
theorem parabola_focus_directrix_distance (a : ℝ) : 
  (∃ y : ℝ → ℝ, ∀ x, (y x)^2 = a * x) →  -- Parabola equation
  (∃ f d : ℝ, abs (f - d) = 2) →        -- Distance between focus and directrix
  a = 4 := by
sorry

end parabola_focus_directrix_distance_l1789_178935


namespace marks_difference_l1789_178967

/-- Given that the average mark in chemistry and mathematics is 55,
    prove that the difference between the total marks in all three subjects
    and the marks in physics is 110. -/
theorem marks_difference (P C M : ℝ) 
    (h1 : (C + M) / 2 = 55) : 
    (P + C + M) - P = 110 := by
  sorry

end marks_difference_l1789_178967


namespace deck_cost_per_square_foot_l1789_178976

/-- Proves the cost per square foot for deck construction given the dimensions, sealant cost, and total cost paid. -/
theorem deck_cost_per_square_foot 
  (length : ℝ) 
  (width : ℝ) 
  (sealant_cost_per_sq_ft : ℝ) 
  (total_cost : ℝ) 
  (h1 : length = 30) 
  (h2 : width = 40) 
  (h3 : sealant_cost_per_sq_ft = 1) 
  (h4 : total_cost = 4800) : 
  ∃ (cost_per_sq_ft : ℝ), 
    cost_per_sq_ft = 3 ∧ 
    total_cost = length * width * (cost_per_sq_ft + sealant_cost_per_sq_ft) :=
by sorry


end deck_cost_per_square_foot_l1789_178976


namespace smallest_valid_fourth_number_l1789_178987

def is_valid_fourth_number (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧
  let sum_of_numbers := 42 + 25 + 56 + n
  let sum_of_digits := (4 + 2 + 2 + 5 + 5 + 6 + (n / 10) + (n % 10))
  4 * sum_of_digits = sum_of_numbers

theorem smallest_valid_fourth_number :
  ∀ n : ℕ, is_valid_fourth_number n → n ≥ 79 :=
sorry

end smallest_valid_fourth_number_l1789_178987


namespace fraction_equality_l1789_178927

theorem fraction_equality : 
  (3 / 7 + 5 / 8) / (5 / 12 + 2 / 15) = 295 / 154 := by sorry

end fraction_equality_l1789_178927


namespace three_percent_difference_l1789_178979

theorem three_percent_difference (x y : ℝ) 
  (hx : 3 = 0.15 * x) 
  (hy : 3 = 0.10 * y) : 
  x - y = -10 := by
sorry

end three_percent_difference_l1789_178979


namespace odd_function_domain_sum_l1789_178981

def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_domain_sum (f : ℝ → ℝ) (a b : ℝ) 
  (h1 : OddFunction f) 
  (h2 : Set.range f = {-1, 2, a, b}) : 
  a + b = -1 := by
  sorry

end odd_function_domain_sum_l1789_178981


namespace madam_arrangements_count_l1789_178938

/-- The number of unique arrangements of the letters in the word MADAM -/
def madam_arrangements : ℕ := 30

/-- The total number of letters in the word MADAM -/
def total_letters : ℕ := 5

/-- The number of times the letter M appears in MADAM -/
def m_count : ℕ := 2

/-- The number of times the letter A appears in MADAM -/
def a_count : ℕ := 2

/-- Theorem stating that the number of unique arrangements of the letters in MADAM is 30 -/
theorem madam_arrangements_count :
  madam_arrangements = Nat.factorial total_letters / (Nat.factorial m_count * Nat.factorial a_count) :=
by sorry

end madam_arrangements_count_l1789_178938


namespace bries_slacks_count_l1789_178940

/-- Proves that Brie has 8 slacks given the conditions of the problem -/
theorem bries_slacks_count :
  ∀ (total_blouses total_skirts total_slacks : ℕ)
    (blouses_in_hamper skirts_in_hamper slacks_in_hamper : ℕ)
    (clothes_to_wash : ℕ),
  total_blouses = 12 →
  total_skirts = 6 →
  blouses_in_hamper = (75 * total_blouses) / 100 →
  skirts_in_hamper = (50 * total_skirts) / 100 →
  slacks_in_hamper = (25 * total_slacks) / 100 →
  clothes_to_wash = 14 →
  clothes_to_wash = blouses_in_hamper + skirts_in_hamper + slacks_in_hamper →
  total_slacks = 8 := by
sorry

end bries_slacks_count_l1789_178940


namespace largest_four_digit_divisible_by_12_l1789_178966

theorem largest_four_digit_divisible_by_12 : ∃ n : ℕ, n = 9996 ∧ 
  n % 12 = 0 ∧ 
  n ≤ 9999 ∧ 
  n ≥ 1000 ∧
  ∀ m : ℕ, m % 12 = 0 ∧ m ≤ 9999 ∧ m ≥ 1000 → m ≤ n := by
  sorry

end largest_four_digit_divisible_by_12_l1789_178966


namespace problem_solution_l1789_178939

theorem problem_solution (a b : ℝ) 
  (h : ∀ x : ℝ, x ∈ Set.Icc 0 1 → |a * x + b - Real.sqrt (1 - x^2)| ≤ (Real.sqrt 2 - 1) / 2) : 
  a = -1 ∧ b = (Real.sqrt 2 + 1) / 2 := by
  sorry

end problem_solution_l1789_178939


namespace z_modulus_l1789_178923

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation for z
def z_equation (z : ℂ) : Prop := z + 2 * i = (3 - i^3) / (1 + i)

-- Theorem statement
theorem z_modulus (z : ℂ) (h : z_equation z) : Complex.abs z = Real.sqrt 13 := by
  sorry

end z_modulus_l1789_178923


namespace sqrt_equation_implies_sum_and_reciprocal_l1789_178977

theorem sqrt_equation_implies_sum_and_reciprocal (x : ℝ) (h : x > 0) :
  Real.sqrt x - 1 / Real.sqrt x = 2 * Real.sqrt 3 → x + 1 / x = 14 := by
  sorry

end sqrt_equation_implies_sum_and_reciprocal_l1789_178977


namespace sum_square_of_sum_and_diff_l1789_178990

theorem sum_square_of_sum_and_diff (x y : ℝ) 
  (sum_eq : x + y = 60) 
  (diff_eq : x - y = 10) : 
  (x + y)^2 = 3600 := by
  sorry

end sum_square_of_sum_and_diff_l1789_178990


namespace point_movement_theorem_l1789_178968

/-- Represents the final position of a point on a number line after a series of movements -/
def final_position (initial : Int) (right_move : Int) (left_move : Int) : Int :=
  initial + right_move - left_move

/-- Theorem stating that given the specific movements in the problem, 
    the final position is -5 -/
theorem point_movement_theorem :
  final_position (-3) 5 7 = -5 := by
  sorry

end point_movement_theorem_l1789_178968


namespace quadratic_inequality_solution_l1789_178954

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, a * x^2 - 3*x + b > 4 ↔ x < 1 ∨ x > 2) →
  (a = 1 ∧ b = 6) ∧
  (∀ c : ℝ,
    (c > 2 → ∀ x, x^2 - (c+2)*x + 2*c < 0 ↔ 2 < x ∧ x < c) ∧
    (c = 2 → ∀ x, ¬(x^2 - (c+2)*x + 2*c < 0)) ∧
    (c < 2 → ∀ x, x^2 - (c+2)*x + 2*c < 0 ↔ c < x ∧ x < 2)) :=
by sorry

end quadratic_inequality_solution_l1789_178954


namespace rectangles_cover_interior_l1789_178995

-- Define the basic structures
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

structure Rectangle where
  corners : Fin 4 → ℝ × ℝ

-- Define the given line
def given_line : Set (ℝ × ℝ) := sorry

-- Define the property of covering the sides of a triangle
def covers_sides (rectangles : Fin 3 → Rectangle) (triangle : Triangle) : Prop := sorry

-- Define the property of having a side parallel to the given line
def has_parallel_side (rectangle : Rectangle) : Prop := sorry

-- Define the property of covering the interior of a triangle
def covers_interior (rectangles : Fin 3 → Rectangle) (triangle : Triangle) : Prop := sorry

-- The main theorem
theorem rectangles_cover_interior 
  (triangle : Triangle) 
  (rectangles : Fin 3 → Rectangle) 
  (h1 : covers_sides rectangles triangle)
  (h2 : ∀ i : Fin 3, has_parallel_side (rectangles i)) :
  covers_interior rectangles triangle := by sorry

end rectangles_cover_interior_l1789_178995


namespace hyperbola_eccentricity_l1789_178913

/-- Given a hyperbola with standard equation (x²/a² - y²/b² = 1) where a > 0 and b > 0,
    if one of its asymptotes has equation y = 3x, then its eccentricity is √10. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = 3) : 
  Real.sqrt (1 + (b / a)^2) = Real.sqrt 10 := by
  sorry

end hyperbola_eccentricity_l1789_178913


namespace binomial_60_3_l1789_178999

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by sorry

end binomial_60_3_l1789_178999


namespace union_of_A_and_B_l1789_178989

def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {x | 1 < x ∧ x < 7}

theorem union_of_A_and_B : A ∪ B = {x | x < 7} := by
  sorry

end union_of_A_and_B_l1789_178989


namespace optimal_triangle_sides_l1789_178964

noncomputable def minTriangleSides (S : ℝ) (x : ℝ) : ℝ × ℝ × ℝ :=
  let BC := 2 * Real.sqrt (S * Real.tan (x / 2))
  let AB := Real.sqrt (S / (Real.sin (x / 2) * Real.cos (x / 2)))
  (BC, AB, AB)

theorem optimal_triangle_sides (S : ℝ) (x : ℝ) (h1 : 0 < S) (h2 : 0 < x) (h3 : x < π) :
  let (BC, AB, AC) := minTriangleSides S x
  BC = 2 * Real.sqrt (S * Real.tan (x / 2)) ∧
  AB = AC ∧
  AB = Real.sqrt (S / (Real.sin (x / 2) * Real.cos (x / 2))) ∧
  ∀ (BC' AB' AC' : ℝ), 
    (BC' * AB' * Real.sin x) / 2 = S → 
    BC' ≥ BC :=
by sorry

end optimal_triangle_sides_l1789_178964


namespace cups_in_first_stack_l1789_178975

theorem cups_in_first_stack (s : Fin 5 → ℕ) 
  (h1 : s 1 = 21)
  (h2 : s 2 = 25)
  (h3 : s 3 = 29)
  (h4 : s 4 = 33)
  (h_arithmetic : ∃ d : ℕ, ∀ i : Fin 4, s (i + 1) = s i + d) :
  s 0 = 17 := by
sorry

end cups_in_first_stack_l1789_178975


namespace cos_negative_23pi_over_4_l1789_178986

theorem cos_negative_23pi_over_4 : Real.cos (-23 * Real.pi / 4) = Real.sqrt 2 / 2 := by
  sorry

end cos_negative_23pi_over_4_l1789_178986


namespace sum_is_composite_l1789_178988

theorem sum_is_composite (a b : ℕ) (h : 34 * a = 43 * b) : 
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ a + b = x * y :=
by sorry

end sum_is_composite_l1789_178988


namespace tennis_game_wins_l1789_178994

theorem tennis_game_wins (total_games : ℕ) (player_a_wins player_b_wins player_c_wins : ℕ) :
  total_games = 6 →
  player_a_wins = 5 →
  player_b_wins = 2 →
  player_c_wins = 1 →
  ∃ player_d_wins : ℕ, player_d_wins = 4 ∧ player_a_wins + player_b_wins + player_c_wins + player_d_wins = 2 * total_games :=
by sorry

end tennis_game_wins_l1789_178994


namespace computer_table_cost_price_l1789_178926

/-- Proves that the cost price of a computer table is 2500 when the selling price is 3000 
    and the markup is 20% -/
theorem computer_table_cost_price 
  (markup_percentage : ℝ) 
  (selling_price : ℝ) : 
  markup_percentage = 20 →
  selling_price = 3000 →
  (100 + markup_percentage) / 100 * (selling_price / (1 + markup_percentage / 100)) = 2500 := by
  sorry

end computer_table_cost_price_l1789_178926


namespace cyclist_average_speed_l1789_178907

/-- Calculates the average speed of a cyclist given two trips with different distances and speeds -/
theorem cyclist_average_speed (d1 d2 v1 v2 : ℝ) (h1 : d1 = 9) (h2 : d2 = 12) (h3 : v1 = 12) (h4 : v2 = 9) :
  let t1 := d1 / v1
  let t2 := d2 / v2
  let total_distance := d1 + d2
  let total_time := t1 + t2
  let average_speed := total_distance / total_time
  ∃ ε > 0, |average_speed - 10.1| < ε :=
by sorry

end cyclist_average_speed_l1789_178907


namespace hyperbola_m_value_l1789_178917

def hyperbola_equation (m : ℝ) (x y : ℝ) : Prop :=
  m * x^2 + y^2 = 1

def imaginary_axis_twice_real_axis (m : ℝ) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2*a = b ∧
  ∀ x y : ℝ, hyperbola_equation m x y ↔ (x/a)^2 - (y/b)^2 = 1

theorem hyperbola_m_value :
  ∀ m : ℝ, imaginary_axis_twice_real_axis m → m = -1/4 :=
by sorry

end hyperbola_m_value_l1789_178917


namespace representation_of_real_number_l1789_178952

theorem representation_of_real_number (x : ℝ) (hx : 0 < x ∧ x ≤ 1) :
  ∃ (n : ℕ → ℕ), 
    (∀ k, n (k + 1) / n k ∈ ({2, 3, 4} : Set ℕ)) ∧ 
    (∑' k, (1 : ℝ) / n k) = x :=
sorry

end representation_of_real_number_l1789_178952


namespace percent_relation_l1789_178919

theorem percent_relation (a b : ℝ) (h : a = 2 * b) : 5 * b = (5/2) * a := by sorry

end percent_relation_l1789_178919


namespace bisected_tangents_iff_parabola_l1789_178931

/-- A curve in the xy-plane -/
structure Curve where
  -- The equation of the curve
  equation : ℝ → ℝ → Prop

/-- Property that any tangent line segment between the point of tangency and the x-axis is bisected by the y-axis -/
def has_bisected_tangents (c : Curve) : Prop :=
  ∀ (x y : ℝ), c.equation x y →
    ∃ (slope : ℝ), 
      -- The tangent line at (x, y) intersects the x-axis at (-x, 0)
      y = slope * (x - (-x))

/-- A parabola of the form y^2 = Cx -/
def is_parabola (c : Curve) : Prop :=
  ∃ (C : ℝ), ∀ (x y : ℝ), c.equation x y ↔ y^2 = C * x

/-- Theorem stating the equivalence between the bisected tangents property and being a parabola -/
theorem bisected_tangents_iff_parabola (c : Curve) :
  has_bisected_tangents c ↔ is_parabola c :=
sorry

end bisected_tangents_iff_parabola_l1789_178931


namespace triangle_inequality_l1789_178949

/-- The length of the shortest altitude of a triangle, or 0 if the points are collinear -/
noncomputable def m (P Q R : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

/-- For any four points on a plane, the inequality m(ABC) ≤ m(ABX) + m(AXC) + m(XBC) holds -/
theorem triangle_inequality (A B C X : EuclideanSpace ℝ (Fin 2)) :
  m A B C ≤ m A B X + m A X C + m X B C := by sorry

end triangle_inequality_l1789_178949


namespace sara_savings_l1789_178950

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The number of quarters Sara has -/
def sara_quarters : ℕ := 11

/-- Theorem: Sara's total savings in cents -/
theorem sara_savings : quarter_value * sara_quarters = 275 := by
  sorry

end sara_savings_l1789_178950


namespace badminton_equipment_purchase_l1789_178915

/-- Represents the cost of purchasing badminton equipment from Store A -/
def cost_store_a (x : ℝ) : ℝ := 1760 + 40 * x

/-- Represents the cost of purchasing badminton equipment from Store B -/
def cost_store_b (x : ℝ) : ℝ := 1920 + 32 * x

theorem badminton_equipment_purchase (x : ℝ) (h : x > 16) :
  (x > 20 → cost_store_b x < cost_store_a x) ∧
  (x < 20 → cost_store_a x < cost_store_b x) := by
  sorry

#check badminton_equipment_purchase

end badminton_equipment_purchase_l1789_178915


namespace derivative_x_squared_sin_x_l1789_178992

theorem derivative_x_squared_sin_x :
  ∀ x : ℝ, deriv (λ x => x^2 * Real.sin x) x = 2 * x * Real.sin x + x^2 * Real.cos x :=
by sorry

end derivative_x_squared_sin_x_l1789_178992


namespace lines_skew_iff_b_ne_18_l1789_178924

/-- Two lines in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Determines if two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop := sorry

/-- The first line -/
def line1 (b : ℝ) : Line3D :=
  { point := (3, 2, b),
    direction := (2, 3, 4) }

/-- The second line -/
def line2 : Line3D :=
  { point := (4, 1, 0),
    direction := (3, 4, 2) }

/-- Main theorem: The lines are skew if and only if b ≠ 18 -/
theorem lines_skew_iff_b_ne_18 (b : ℝ) :
  are_skew (line1 b) line2 ↔ b ≠ 18 := by sorry

end lines_skew_iff_b_ne_18_l1789_178924


namespace max_sum_of_factors_of_24_l1789_178957

theorem max_sum_of_factors_of_24 : 
  ∀ (a b : ℕ), a * b = 24 → a + b ≤ 25 :=
by
  sorry

end max_sum_of_factors_of_24_l1789_178957


namespace cookies_left_for_sonny_l1789_178920

theorem cookies_left_for_sonny (total : ℕ) (brother sister cousin : ℕ) 
  (h1 : total = 45)
  (h2 : brother = 12)
  (h3 : sister = 9)
  (h4 : cousin = 7) :
  total - (brother + sister + cousin) = 17 := by
  sorry

end cookies_left_for_sonny_l1789_178920


namespace razor_blade_profit_equation_l1789_178906

theorem razor_blade_profit_equation (x : ℝ) :
  (x ≥ 0) →                          -- number of razors sold is non-negative
  (30 : ℝ) * x +                     -- profit from razors
  (-0.5 : ℝ) * (2 * x) =             -- loss from blades (twice the number of razors)
  (5800 : ℝ)                         -- total profit
  := by sorry

end razor_blade_profit_equation_l1789_178906


namespace third_month_sale_l1789_178901

def sales_data : List ℕ := [8435, 8927, 9230, 8562, 6991]
def average_sale : ℕ := 8500
def num_months : ℕ := 6

theorem third_month_sale :
  ∃ x : ℕ, 
    (List.sum sales_data + x) / num_months = average_sale ∧
    x = 8855 := by
  sorry

end third_month_sale_l1789_178901


namespace pascals_triangle_row20_l1789_178933

def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem pascals_triangle_row20 : 
  (binomial 20 6 = 38760) ∧ 
  (binomial 20 6 / binomial 20 2 = 204) := by
sorry

end pascals_triangle_row20_l1789_178933


namespace prob_three_spades_two_hearts_correct_l1789_178984

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Fin 52)

/-- Represents the suit of a card -/
inductive Suit
| spades | hearts | diamonds | clubs

/-- Represents the rank of a card -/
def Rank := Fin 13

/-- The probability of drawing three spades followed by two hearts from a standard deck -/
def prob_three_spades_two_hearts : ℚ :=
  432 / 6497400

theorem prob_three_spades_two_hearts_correct (d : Deck) :
  prob_three_spades_two_hearts = 
    (13 * 12 * 11 * 13 * 12) / (52 * 51 * 50 * 49 * 48) :=
by sorry

end prob_three_spades_two_hearts_correct_l1789_178984


namespace random_points_probability_l1789_178909

/-- The probability that a randomly chosen point y is greater than another randomly chosen point x
    but less than three times x, where both x and y are chosen uniformly from the interval [0, 1] -/
theorem random_points_probability : Real := by
  sorry

end random_points_probability_l1789_178909


namespace prime_value_problem_l1789_178993

theorem prime_value_problem : ∃ p : ℕ, 
  Prime p ∧ 
  (5 * p) % 4 = 3 ∧ 
  Prime (13 * p + 2) ∧ 
  13 * p + 2 = 41 := by
sorry

end prime_value_problem_l1789_178993


namespace short_hair_dog_count_is_six_l1789_178961

/-- Represents the dog grooming scenario -/
structure DogGrooming where
  shortHairDryTime : ℕ
  fullHairDryTime : ℕ
  fullHairDogCount : ℕ
  totalDryTime : ℕ

/-- The number of short-haired dogs in the grooming scenario -/
def shortHairDogCount (dg : DogGrooming) : ℕ :=
  (dg.totalDryTime - dg.fullHairDogCount * dg.fullHairDryTime) / dg.shortHairDryTime

/-- Theorem stating the number of short-haired dogs in the given scenario -/
theorem short_hair_dog_count_is_six :
  let dg : DogGrooming := {
    shortHairDryTime := 10,
    fullHairDryTime := 20,
    fullHairDogCount := 9,
    totalDryTime := 240
  }
  shortHairDogCount dg = 6 := by sorry

end short_hair_dog_count_is_six_l1789_178961


namespace total_boys_in_three_sections_l1789_178916

theorem total_boys_in_three_sections (section1_total : ℕ) (section2_total : ℕ) (section3_total : ℕ)
  (section1_girls_ratio : ℚ) (section2_boys_ratio : ℚ) (section3_boys_ratio : ℚ) :
  section1_total = 160 →
  section2_total = 200 →
  section3_total = 240 →
  section1_girls_ratio = 1/4 →
  section2_boys_ratio = 3/5 →
  section3_boys_ratio = 7/12 →
  (section1_total - section1_total * section1_girls_ratio) +
  (section2_total * section2_boys_ratio) +
  (section3_total * section3_boys_ratio) = 380 := by
sorry

end total_boys_in_three_sections_l1789_178916


namespace negation_false_l1789_178928

theorem negation_false : ¬∃ a : ℝ, a ≤ 2 ∧ a^2 ≥ 4 := by
  sorry

end negation_false_l1789_178928


namespace roundness_of_250000_l1789_178955

/-- The roundness of a positive integer is the sum of the exponents in its prime factorization. -/
def roundness (n : ℕ+) : ℕ := sorry

/-- The roundness of 250,000 is 10. -/
theorem roundness_of_250000 : roundness 250000 = 10 := by sorry

end roundness_of_250000_l1789_178955


namespace probability_of_sum_5_is_one_thirty_sixth_l1789_178941

/-- A fair 6-sided die with distinct numbers 1 through 6 -/
def FairDie : Type := Fin 6

/-- The number of dice rolled -/
def numDice : ℕ := 3

/-- The target sum we're aiming for -/
def targetSum : ℕ := 5

/-- The set of all possible outcomes when rolling three fair 6-sided dice -/
def allOutcomes : Finset (FairDie × FairDie × FairDie) := sorry

/-- The set of favorable outcomes (those that sum to targetSum) -/
def favorableOutcomes : Finset (FairDie × FairDie × FairDie) := sorry

/-- The probability of rolling a total of 5 with three fair 6-sided dice -/
def probabilityOfSum5 : ℚ :=
  (Finset.card favorableOutcomes : ℚ) / (Finset.card allOutcomes : ℚ)

/-- Theorem stating that the probability of rolling a sum of 5 with three fair 6-sided dice is 1/36 -/
theorem probability_of_sum_5_is_one_thirty_sixth :
  probabilityOfSum5 = 1 / 36 := by sorry

end probability_of_sum_5_is_one_thirty_sixth_l1789_178941


namespace square_equals_self_only_zero_and_one_l1789_178956

theorem square_equals_self_only_zero_and_one :
  ∀ x : ℝ, x^2 = x ↔ x = 0 ∨ x = 1 := by sorry

end square_equals_self_only_zero_and_one_l1789_178956


namespace quadratic_inequality_l1789_178914

theorem quadratic_inequality (a b c : ℝ) (ha : a ≠ 0)
  (hroot : ∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ 2 * a * x^2 + b * x + c = 0) :
  min c (a + c + 1) ≤ max (|b - a + 1|) (|b + a - 1|) := by
  sorry

end quadratic_inequality_l1789_178914


namespace max_consecutive_indivisible_l1789_178965

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def is_indivisible (n : ℕ) : Prop :=
  ∀ a b : ℕ, (100 ≤ a ∧ a ≤ 999) → (100 ≤ b ∧ b ≤ 999) → n ≠ a * b

theorem max_consecutive_indivisible :
  ∀ start : ℕ, is_five_digit start →
    ∃ k : ℕ, k ≤ 99 ∧ ¬(is_indivisible (start + k + 1)) :=
by sorry

end max_consecutive_indivisible_l1789_178965


namespace can_cut_one_more_square_l1789_178930

/-- Represents a grid of squares -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a square that can be cut from the grid -/
structure Square :=
  (size : ℕ)

/-- Function to calculate the number of cells in a grid -/
def grid_cells (g : Grid) : ℕ := g.rows * g.cols

/-- Function to calculate the number of cells in a square -/
def square_cells (s : Square) : ℕ := s.size * s.size

/-- Function to calculate the number of cells remaining after cutting squares -/
def remaining_cells (g : Grid) (s : Square) (n : ℕ) : ℕ :=
  grid_cells g - n * square_cells s

/-- Theorem stating that after cutting 99 2x2 squares from a 29x29 grid, 
    at least one more 2x2 square can be cut -/
theorem can_cut_one_more_square (g : Grid) (s : Square) :
  g.rows = 29 → g.cols = 29 → s.size = 2 →
  ∃ (m : ℕ), m > 99 ∧ remaining_cells g s m ≥ square_cells s :=
by sorry

end can_cut_one_more_square_l1789_178930


namespace circle_and_line_properties_l1789_178934

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Define the line l₁ passing through A(1,0)
def line_l1 (x y : ℝ) : Prop := ∃ k : ℝ, y = k * (x - 1)

-- Define tangent line condition
def is_tangent (line : (ℝ → ℝ → Prop)) (circle : (ℝ → ℝ → Prop)) : Prop :=
  ∃ x y : ℝ, line x y ∧ circle x y ∧
  ∀ x' y' : ℝ, line x' y' → circle x' y' → (x', y') = (x, y)

-- Define the slope angle of π/4
def slope_angle_pi_4 (line : (ℝ → ℝ → Prop)) : Prop :=
  ∃ k : ℝ, (∀ x y : ℝ, line x y ↔ y = k * (x - 1)) ∧ k = 1

-- Main theorem
theorem circle_and_line_properties :
  (is_tangent line_l1 circle_C →
    (∀ x y : ℝ, line_l1 x y ↔ (x = 1 ∨ 3*x - 4*y - 3 = 0))) ∧
  (slope_angle_pi_4 line_l1 →
    ∃ P Q : ℝ × ℝ,
      circle_C P.1 P.2 ∧ circle_C Q.1 Q.2 ∧
      line_l1 P.1 P.2 ∧ line_l1 Q.1 Q.2 ∧
      ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) = (4, 3)) :=
sorry

end circle_and_line_properties_l1789_178934


namespace equation_implies_a_equals_four_l1789_178960

theorem equation_implies_a_equals_four (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x + 4 = (x + 2)^2) → a = 4 := by
  sorry

end equation_implies_a_equals_four_l1789_178960


namespace quadratic_max_min_difference_l1789_178921

def f (x : ℝ) := x^2 - 4*x - 6

theorem quadratic_max_min_difference :
  let x_min := -3
  let x_max := 4
  ∃ (x_min_value x_max_value : ℝ),
    (∀ x, x_min ≤ x ∧ x ≤ x_max → f x ≥ x_min_value) ∧
    (∃ x, x_min ≤ x ∧ x ≤ x_max ∧ f x = x_min_value) ∧
    (∀ x, x_min ≤ x ∧ x ≤ x_max → f x ≤ x_max_value) ∧
    (∃ x, x_min ≤ x ∧ x ≤ x_max ∧ f x = x_max_value) ∧
    x_max_value - x_min_value = 25 :=
by sorry

end quadratic_max_min_difference_l1789_178921


namespace x_range_l1789_178911

theorem x_range (x : ℝ) (h : |2*x + 1| + |2*x - 5| = 6) : 
  x ∈ Set.Icc (-1/2 : ℝ) (5/2 : ℝ) := by
  sorry

end x_range_l1789_178911


namespace parabola_sum_l1789_178947

/-- A parabola with equation y = px^2 + qx + r, vertex (3, 7), vertical axis of symmetry, and containing the point (0, 10) has p + q + r = 8 1/3 -/
theorem parabola_sum (p q r : ℝ) : 
  (∀ x y : ℝ, y = p * x^2 + q * x + r) →  -- Equation of the parabola
  (∃ a : ℝ, ∀ x y : ℝ, y = a * (x - 3)^2 + 7) →  -- Vertex form with (3, 7)
  (10 : ℝ) = p * 0^2 + q * 0 + r →  -- Point (0, 10) on the parabola
  p + q + r = 8 + 1/3 := by
sorry

end parabola_sum_l1789_178947


namespace absolute_value_inequality_l1789_178908

theorem absolute_value_inequality (x : ℝ) :
  (4 ≤ |x + 2| ∧ |x + 2| ≤ 8) ↔ ((-10 ≤ x ∧ x ≤ -6) ∨ (2 ≤ x ∧ x ≤ 6)) :=
by sorry

end absolute_value_inequality_l1789_178908


namespace symmetric_points_sum_l1789_178948

/-- Two points are symmetric about a line if their midpoint lies on that line -/
def symmetric_points (x₁ y₁ x₂ y₂ k b : ℝ) : Prop :=
  let mx := (x₁ + x₂) / 2
  let my := (y₁ + y₂) / 2
  k * mx - my + b = 0

/-- The theorem statement -/
theorem symmetric_points_sum (m k : ℝ) :
  symmetric_points 1 2 (-1) m k 3 → m + k = 5 := by
  sorry

end symmetric_points_sum_l1789_178948


namespace fernanda_savings_after_payments_l1789_178929

/-- Calculates the total amount in Fernanda's savings account after receiving payments from debtors -/
theorem fernanda_savings_after_payments (aryan_debt kyro_debt : ℚ) 
  (h1 : aryan_debt = 1200)
  (h2 : aryan_debt = 2 * kyro_debt)
  (h3 : aryan_payment = 0.6 * aryan_debt)
  (h4 : kyro_payment = 0.8 * kyro_debt)
  (h5 : initial_savings = 300) :
  initial_savings + aryan_payment + kyro_payment = 1500 := by
  sorry

end fernanda_savings_after_payments_l1789_178929


namespace smallest_square_ending_2016_l1789_178998

theorem smallest_square_ending_2016 : ∃ (n : ℕ), n = 996 ∧ 
  (∀ (m : ℕ), m < n → m^2 % 10000 ≠ 2016) ∧ n^2 % 10000 = 2016 := by
  sorry

end smallest_square_ending_2016_l1789_178998


namespace olivia_total_time_l1789_178996

/-- The total time Olivia spent on her math problems -/
def total_time (
  num_problems : ℕ)
  (time_first_three : ℕ)
  (time_next_three : ℕ)
  (time_last : ℕ)
  (break_time : ℕ)
  (checking_time : ℕ) : ℕ :=
  3 * time_first_three + 3 * time_next_three + time_last + break_time + checking_time

/-- Theorem stating that Olivia spent 43 minutes in total on her math problems -/
theorem olivia_total_time :
  total_time 7 4 6 8 2 3 = 43 :=
by sorry

end olivia_total_time_l1789_178996


namespace x_value_proof_l1789_178904

theorem x_value_proof (w y z x : ℤ) 
  (hw : w = 90)
  (hz : z = w + 25)
  (hy : y = z + 12)
  (hx : x = y + 7) : 
  x = 134 := by
  sorry

end x_value_proof_l1789_178904


namespace functional_equation_solution_l1789_178985

-- Define the property that a function f must satisfy
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y + f (x + y) = x * y

-- State the theorem
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesEquation f ↔ (∀ x, f x = x - 1) ∨ (∀ x, f x = -x - 1) :=
by sorry

end functional_equation_solution_l1789_178985


namespace abc_equality_l1789_178905

theorem abc_equality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a * (1 - b) = 1/4) (h2 : b * (1 - c) = 1/4) (h3 : c * (1 - a) = 1/4) :
  a = b ∧ b = c := by
  sorry

end abc_equality_l1789_178905


namespace scientific_notation_239000000_l1789_178951

theorem scientific_notation_239000000 :
  239000000 = 2.39 * (10 ^ 8) := by
  sorry

end scientific_notation_239000000_l1789_178951


namespace factorization_x8_minus_81_l1789_178942

theorem factorization_x8_minus_81 (x : ℝ) : 
  x^8 - 81 = (x^2 - 3) * (x^2 + 3) * (x^4 + 9) := by
  sorry

end factorization_x8_minus_81_l1789_178942


namespace limit_equivalence_l1789_178980

def has_limit (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε : ℝ, ∃ N : ℕ, ∀ n : ℕ, (ε > 0 ∧ n ≥ N) → |L - u n| ≤ ε

def alt_def1 (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε : ℝ, ε ≤ 0 ∨ (∃ N : ℕ, ∀ n : ℕ, |L - u n| ≤ ε ∨ n < N)

def alt_def2 (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε : ℝ, ∀ n : ℕ, ∃ N : ℕ, (ε > 0 ∧ n ≥ N) → |L - u n| ≤ ε

def alt_def3 (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε : ℝ, ∃ N : ℕ, ∀ n : ℕ, (ε > 0 ∧ n > N) → |L - u n| < ε

def alt_def4 (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∃ N : ℕ, ∀ ε : ℝ, ∀ n : ℕ, (ε > 0 ∧ n ≥ N) → |L - u n| ≤ ε

theorem limit_equivalence (u : ℕ → ℝ) (L : ℝ) :
  (has_limit u L ↔ alt_def1 u L) ∧
  (has_limit u L ↔ alt_def3 u L) ∧
  ¬(has_limit u L ↔ alt_def2 u L) ∧
  ¬(has_limit u L ↔ alt_def4 u L) := by
  sorry

end limit_equivalence_l1789_178980


namespace trigonometric_identity_l1789_178978

theorem trigonometric_identity : 
  Real.sin (347 * π / 180) * Real.cos (148 * π / 180) + 
  Real.sin (77 * π / 180) * Real.cos (58 * π / 180) = 
  Real.sqrt 2 / 2 := by
sorry

end trigonometric_identity_l1789_178978


namespace inequality_system_solution_l1789_178973

theorem inequality_system_solution (x : ℝ) :
  (x - 1 < 2*x + 1) ∧ ((2*x - 5) / 3 ≤ 1) → -2 < x ∧ x ≤ 4 := by
  sorry

end inequality_system_solution_l1789_178973


namespace f_g_5_l1789_178972

def g (x : ℝ) : ℝ := 4 * x - 5

def f (x : ℝ) : ℝ := 6 * x + 11

theorem f_g_5 : f (g 5) = 101 := by
  sorry

end f_g_5_l1789_178972


namespace operations_to_equality_l1789_178937

theorem operations_to_equality (a b : ℕ) (h : a = 515 ∧ b = 53) : 
  ∃ n : ℕ, n = 21 ∧ a - 11 * n = b + 11 * n :=
by sorry

end operations_to_equality_l1789_178937


namespace tangent_length_circle_l1789_178925

/-- The length of the tangent line from a point on a circle to the circle itself -/
theorem tangent_length_circle (x y : ℝ) : 
  x^2 + (y - 2)^2 = 4 → 
  x = 2 → 
  y = 2 → 
  Real.sqrt ((x - 0)^2 + (y - 2)^2 - 4) = 2 := by
sorry

end tangent_length_circle_l1789_178925


namespace combined_girls_average_l1789_178902

/-- Represents a high school with given average scores -/
structure School where
  boys_avg : ℝ
  girls_avg : ℝ
  combined_avg : ℝ

/-- Represents the combined data for two high schools -/
structure CombinedSchools where
  cedar : School
  delta : School
  boys_combined_avg : ℝ

/-- Theorem stating that the combined girls' average is 86 -/
theorem combined_girls_average (schools : CombinedSchools) 
  (h1 : schools.cedar.boys_avg = 85)
  (h2 : schools.cedar.girls_avg = 80)
  (h3 : schools.cedar.combined_avg = 83)
  (h4 : schools.delta.boys_avg = 76)
  (h5 : schools.delta.girls_avg = 95)
  (h6 : schools.delta.combined_avg = 87)
  (h7 : schools.boys_combined_avg = 73) :
  ∃ (cedar_boys cedar_girls delta_boys delta_girls : ℝ),
    cedar_boys > 0 ∧ cedar_girls > 0 ∧ delta_boys > 0 ∧ delta_girls > 0 ∧
    (cedar_boys * 85 + cedar_girls * 80) / (cedar_boys + cedar_girls) = 83 ∧
    (delta_boys * 76 + delta_girls * 95) / (delta_boys + delta_girls) = 87 ∧
    (cedar_boys * 85 + delta_boys * 76) / (cedar_boys + delta_boys) = 73 ∧
    (cedar_girls * 80 + delta_girls * 95) / (cedar_girls + delta_girls) = 86 :=
by sorry


end combined_girls_average_l1789_178902


namespace ellipse_properties_l1789_178922

/-- An ellipse E with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0

/-- Points on the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The ellipse problem setup -/
structure EllipseProblem where
  E : Ellipse
  O : Point
  A : Point
  B : Point
  C : Point
  M : Point
  N : Point
  h_O : O.x = 0 ∧ O.y = 0
  h_A : A.x = E.a ∧ A.y = 0
  h_B : B.x = 0 ∧ B.y = E.b
  h_C : C.x = -E.a ∧ C.y = 0
  h_M_on_AB : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ M.x = (1 - t) * A.x + t * B.x ∧ M.y = (1 - t) * A.y + t * B.y
  h_BM_AM : (B.x - M.x)^2 + (B.y - M.y)^2 = 4 * ((M.x - A.x)^2 + (M.y - A.y)^2)
  h_OM_slope : M.y / M.x = Real.sqrt 5 / 10
  h_N_midpoint : N.x = (B.x + C.x) / 2 ∧ N.y = (B.y + C.y) / 2
  h_N_reflection : ∃ S : Point, (S.x - N.x) * E.b = (S.y - N.y) * E.a ∧ S.y = 13/2

/-- The main theorem stating the properties of the ellipse -/
theorem ellipse_properties (prob : EllipseProblem) :
  (prob.E.a^2 - prob.E.b^2) / prob.E.a^2 = 4/5 ∧
  prob.E.a = 3 * Real.sqrt 5 ∧ prob.E.b = 3 :=
sorry

end ellipse_properties_l1789_178922


namespace bernardo_wins_l1789_178910

def game_winner (N : ℕ) : Prop :=
  N ≤ 999 ∧
  2 * N < 1000 ∧
  2 * N + 75 < 1000 ∧
  4 * N + 150 < 1000 ∧
  4 * N + 225 < 1000 ∧
  8 * N + 450 < 1000 ∧
  8 * N + 525 < 1000 ∧
  16 * N + 1050 ≥ 1000

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem bernardo_wins :
  (∃ N : ℕ, game_winner N ∧ 
    (∀ M : ℕ, M < N → ¬game_winner M) ∧
    N = 56 ∧
    sum_of_digits N = 11) :=
  sorry

end bernardo_wins_l1789_178910


namespace integer_fraction_equality_l1789_178983

theorem integer_fraction_equality (d : ℤ) : ∃ m n : ℤ, d * (m^2 - n) = n - 2*m + 1 := by
  sorry

end integer_fraction_equality_l1789_178983


namespace closest_to_fraction_l1789_178991

def options : List ℝ := [50, 500, 1500, 1600, 2000]

theorem closest_to_fraction (options : List ℝ) :
  let fraction : ℝ := 351 / 0.22
  let differences := options.map (λ x => |x - fraction|)
  let min_diff := differences.minimum?
  let closest := options.find? (λ x => |x - fraction| = min_diff.get!)
  closest = some 1600 := by
  sorry

end closest_to_fraction_l1789_178991


namespace no_solution_in_naturals_l1789_178903

theorem no_solution_in_naturals :
  ∀ (x y z t : ℕ), (15^x + 29^y + 43^z) % 7 ≠ (t^2) % 7 := by
  sorry

end no_solution_in_naturals_l1789_178903


namespace correct_number_of_bills_l1789_178974

/-- The total amount of money in dollars -/
def total_amount : ℕ := 10000

/-- The denomination of each bill in dollars -/
def bill_denomination : ℕ := 50

/-- The number of bills -/
def number_of_bills : ℕ := total_amount / bill_denomination

theorem correct_number_of_bills : number_of_bills = 200 := by
  sorry

end correct_number_of_bills_l1789_178974


namespace ellipse_properties_l1789_178918

/-- An ellipse with center at the origin, foci on the x-axis, left focus at (-2,0), and passing through (2,√2) -/
structure Ellipse :=
  (equation : ℝ → ℝ → Prop)
  (center_origin : equation 0 0)
  (foci_on_x_axis : ∀ y, y ≠ 0 → ¬ equation (-2) y ∧ ¬ equation 2 y)
  (left_focus : equation (-2) 0)
  (passes_through : equation 2 (Real.sqrt 2))

/-- The intersection points of a line y=kx with the ellipse -/
def intersect (C : Ellipse) (k : ℝ) : Set (ℝ × ℝ) :=
  {p | C.equation p.1 p.2 ∧ p.2 = k * p.1}

/-- The y-intercepts of lines from A to intersection points -/
def y_intercepts (C : Ellipse) (k : ℝ) : Set ℝ :=
  {y | ∃ p ∈ intersect C k, y = (p.2 / (p.1 + 2*Real.sqrt 2)) * (2*Real.sqrt 2)}

/-- The theorem to be proved -/
theorem ellipse_properties (C : Ellipse) :
  (∀ x y, C.equation x y ↔ x^2/8 + y^2/4 = 1) ∧
  (∀ k ≠ 0, ∀ y ∈ y_intercepts C k,
    (0^2 + y^2 + 2*Real.sqrt 2/k*y = 4) ∧
    (2^2 + 0^2 + 2*Real.sqrt 2/k*0 = 4) ∧
    ((-2)^2 + 0^2 + 2*Real.sqrt 2/k*0 = 4)) :=
sorry

end ellipse_properties_l1789_178918


namespace shape_area_l1789_178969

-- Define the shape
structure Shape where
  sides_equal : Bool
  right_angles : Bool
  num_squares : Nat
  small_square_side : Real

-- Define the theorem
theorem shape_area (s : Shape) 
  (h1 : s.sides_equal = true) 
  (h2 : s.right_angles = true) 
  (h3 : s.num_squares = 8) 
  (h4 : s.small_square_side = 2) : 
  s.num_squares * (s.small_square_side * s.small_square_side) = 32 := by
  sorry

end shape_area_l1789_178969


namespace min_value_expression_l1789_178962

theorem min_value_expression (n : ℕ+) : 
  (n : ℝ) / 2 + 24 / (n : ℝ) ≥ 7 ∧ ∃ m : ℕ+, (m : ℝ) / 2 + 24 / (m : ℝ) = 7 := by
  sorry

end min_value_expression_l1789_178962


namespace purple_flowers_killed_is_40_l1789_178936

/-- Represents the florist's bouquet problem -/
structure BouquetProblem where
  flowers_per_bouquet : ℕ
  initial_seeds_per_color : ℕ
  num_colors : ℕ
  red_killed : ℕ
  yellow_killed : ℕ
  orange_killed : ℕ
  bouquets_made : ℕ

/-- Calculates the number of purple flowers killed by the fungus -/
def purple_flowers_killed (problem : BouquetProblem) : ℕ :=
  let total_initial := problem.initial_seeds_per_color * problem.num_colors
  let red_left := problem.initial_seeds_per_color - problem.red_killed
  let yellow_left := problem.initial_seeds_per_color - problem.yellow_killed
  let orange_left := problem.initial_seeds_per_color - problem.orange_killed
  let total_needed := problem.flowers_per_bouquet * problem.bouquets_made
  let non_purple_left := red_left + yellow_left + orange_left
  problem.initial_seeds_per_color - (total_needed - non_purple_left)

/-- Theorem stating that the number of purple flowers killed is 40 -/
theorem purple_flowers_killed_is_40 (problem : BouquetProblem) 
    (h1 : problem.flowers_per_bouquet = 9)
    (h2 : problem.initial_seeds_per_color = 125)
    (h3 : problem.num_colors = 4)
    (h4 : problem.red_killed = 45)
    (h5 : problem.yellow_killed = 61)
    (h6 : problem.orange_killed = 30)
    (h7 : problem.bouquets_made = 36) :
    purple_flowers_killed problem = 40 := by
  sorry

end purple_flowers_killed_is_40_l1789_178936


namespace det_B_squared_minus_3B_l1789_178912

def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 3, 2]

theorem det_B_squared_minus_3B : 
  Matrix.det ((B ^ 2) - 3 • B) = 88 := by sorry

end det_B_squared_minus_3B_l1789_178912


namespace cubic_root_sum_l1789_178953

theorem cubic_root_sum (u v w : ℝ) : 
  (u^3 - 6*u^2 + 11*u - 6 = 0) →
  (v^3 - 6*v^2 + 11*v - 6 = 0) →
  (w^3 - 6*w^2 + 11*w - 6 = 0) →
  u*v/w + v*w/u + w*u/v = 49/6 := by
sorry

end cubic_root_sum_l1789_178953


namespace parking_garage_spaces_l1789_178971

theorem parking_garage_spaces (level1 level2 level3 level4 : ℕ) : 
  level1 = 90 →
  level3 = level2 + 12 →
  level4 = level3 - 9 →
  level1 + level2 + level3 + level4 = 399 →
  level2 = level1 + 8 :=
by
  sorry

end parking_garage_spaces_l1789_178971


namespace book_reading_time_l1789_178963

/-- Calculates the number of weeks needed to read a book given the total pages and pages read per week. -/
def weeks_to_read (total_pages : ℕ) (pages_per_day : ℕ) (reading_days_per_week : ℕ) : ℕ :=
  total_pages / (pages_per_day * reading_days_per_week)

/-- Proves that it takes 7 weeks to read a 2100-page book when reading 100 pages on 3 days per week. -/
theorem book_reading_time : weeks_to_read 2100 100 3 = 7 := by
  sorry

end book_reading_time_l1789_178963


namespace students_not_enrolled_l1789_178944

theorem students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ)
  (h_total : total = 78)
  (h_french : french = 41)
  (h_german : german = 22)
  (h_both : both = 9) :
  total - (french + german - both) = 24 := by
  sorry

end students_not_enrolled_l1789_178944


namespace janice_earnings_this_week_l1789_178932

/-- Calculates Janice's earnings after deductions for a week -/
def janice_earnings (regular_daily_rate : ℚ) (days_worked : ℕ) 
  (weekday_overtime_rate : ℚ) (weekend_overtime_rate : ℚ)
  (weekday_overtime_shifts : ℕ) (weekend_overtime_shifts : ℕ)
  (tips : ℚ) (tax_rate : ℚ) : ℚ :=
  let regular_earnings := regular_daily_rate * days_worked
  let weekday_overtime := weekday_overtime_rate * weekday_overtime_shifts
  let weekend_overtime := weekend_overtime_rate * weekend_overtime_shifts
  let total_before_tax := regular_earnings + weekday_overtime + weekend_overtime + tips
  let tax := tax_rate * total_before_tax
  total_before_tax - tax

/-- Theorem stating Janice's earnings after deductions -/
theorem janice_earnings_this_week : 
  janice_earnings 30 6 15 20 2 1 10 (1/10) = 216 := by
  sorry


end janice_earnings_this_week_l1789_178932


namespace trigonometric_inequality_l1789_178959

theorem trigonometric_inequality (x : ℝ) (h : x ∈ Set.Ioo 0 (3 * π / 8)) :
  (1 / Real.sin (x / 3)) + (1 / Real.sin (8 * x / 3)) > 
  Real.sin (3 * x / 2) / (Real.sin (x / 2) * Real.sin (2 * x)) := by
  sorry

end trigonometric_inequality_l1789_178959


namespace square_root_of_8_factorial_over_70_l1789_178997

theorem square_root_of_8_factorial_over_70 : 
  let factorial_8 := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
  Real.sqrt (factorial_8 / 70) = 24 := by
  sorry

end square_root_of_8_factorial_over_70_l1789_178997


namespace problem_statement_l1789_178970

theorem problem_statement (x y : ℕ) (hx : x = 4) (hy : y = 3) : 5 * x + 2 * y * 3 = 38 := by
  sorry

end problem_statement_l1789_178970


namespace box_volume_l1789_178982

/-- The volume of a rectangular box with given dimensions -/
theorem box_volume (height length width : ℝ) 
  (h_height : height = 12)
  (h_length : length = 3 * height)
  (h_width : width = length / 4) :
  height * length * width = 3888 :=
by sorry

end box_volume_l1789_178982


namespace y₁_gt_y₂_l1789_178943

/-- A linear function f(x) = -2x + 1 -/
def f (x : ℝ) : ℝ := -2 * x + 1

/-- Point A on the graph of f -/
def A : ℝ × ℝ := (1, f 1)

/-- Point B on the graph of f -/
def B : ℝ × ℝ := (3, f 3)

/-- y₁ coordinate of point A -/
def y₁ : ℝ := (A.2)

/-- y₂ coordinate of point B -/
def y₂ : ℝ := (B.2)

theorem y₁_gt_y₂ : y₁ > y₂ := by
  sorry

end y₁_gt_y₂_l1789_178943


namespace solve_run_problem_l1789_178958

def run_problem (speed2 : ℝ) : Prop :=
  let time1 : ℝ := 0.5
  let speed1 : ℝ := 10
  let time2 : ℝ := 0.5
  let time3 : ℝ := 0.25
  let speed3 : ℝ := 8
  let total_distance : ℝ := 17
  (speed1 * time1) + (speed2 * time2) + (speed3 * time3) = total_distance

theorem solve_run_problem : 
  ∃ (speed2 : ℝ), run_problem speed2 ∧ speed2 = 20 := by
  sorry

end solve_run_problem_l1789_178958


namespace antifreeze_concentration_proof_l1789_178946

-- Define the constants
def total_volume : ℝ := 55
def pure_antifreeze_volume : ℝ := 6.11
def other_mixture_concentration : ℝ := 0.1

-- Define the theorem
theorem antifreeze_concentration_proof :
  let other_mixture_volume : ℝ := total_volume - pure_antifreeze_volume
  let total_pure_antifreeze : ℝ := pure_antifreeze_volume + other_mixture_concentration * other_mixture_volume
  let final_concentration : ℝ := total_pure_antifreeze / total_volume
  ∃ ε > 0, |final_concentration - 0.2| < ε := by
  sorry

end antifreeze_concentration_proof_l1789_178946


namespace robot_center_movement_l1789_178945

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circular robot -/
structure CircularRobot where
  center : Point
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a point remains on a line -/
def remainsOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Predicate to check if a point is on the boundary of a circular robot -/
def isOnBoundary (p : Point) (r : CircularRobot) : Prop :=
  (p.x - r.center.x)^2 + (p.y - r.center.y)^2 = r.radius^2

/-- The main theorem -/
theorem robot_center_movement
  (r : CircularRobot)
  (h : ∀ (p : Point), isOnBoundary p r → ∃ (l : Line), ∀ (t : ℝ), remainsOnLine p l) :
  ¬ (∀ (t : ℝ), ∃ (l : Line), remainsOnLine r.center l) :=
sorry

end robot_center_movement_l1789_178945
