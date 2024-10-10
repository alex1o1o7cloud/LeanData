import Mathlib

namespace opposite_pairs_l2652_265286

theorem opposite_pairs : 
  (-((-2)^3) ≠ -|((-2)^3)|) ∧ 
  ((-2)^3 ≠ -(2^3)) ∧ 
  (-2^2 = -(((-2)^2))) ∧ 
  (-(-2) ≠ -|(-2)|) := by
  sorry

end opposite_pairs_l2652_265286


namespace missed_number_sum_l2652_265229

theorem missed_number_sum (n : ℕ) (missing : ℕ) : 
  n = 63 → 
  missing ≤ n →
  (n * (n + 1)) / 2 - missing = 1991 →
  missing = 25 := by
sorry

end missed_number_sum_l2652_265229


namespace standard_deviation_transform_l2652_265223

/-- Given a sample of 10 data points, this function represents their standard deviation. -/
def standard_deviation (x : Fin 10 → ℝ) : ℝ := sorry

/-- This function represents the transformation applied to each data point. -/
def transform (x : ℝ) : ℝ := 3 * x - 1

theorem standard_deviation_transform (x : Fin 10 → ℝ) :
  standard_deviation x = 8 →
  standard_deviation (λ i => transform (x i)) = 24 := by
  sorry

end standard_deviation_transform_l2652_265223


namespace negation_of_implication_l2652_265218

theorem negation_of_implication (a b c : ℝ) :
  ¬(a > b → a + c > b + c) ↔ (a ≤ b → a + c ≤ b + c) := by sorry

end negation_of_implication_l2652_265218


namespace bottle_cap_count_l2652_265283

/-- Represents the number of bottle caps in one ounce -/
def caps_per_ounce : ℕ := 7

/-- Represents the weight of the bottle cap collection in pounds -/
def collection_weight_pounds : ℕ := 18

/-- Represents the number of ounces in one pound -/
def ounces_per_pound : ℕ := 16

/-- Calculates the total number of bottle caps in the collection -/
def total_caps : ℕ := collection_weight_pounds * ounces_per_pound * caps_per_ounce

theorem bottle_cap_count : total_caps = 2016 := by
  sorry

end bottle_cap_count_l2652_265283


namespace triangle_side_lengths_l2652_265242

theorem triangle_side_lengths 
  (α : Real) 
  (r R : Real) 
  (hr : r > 0) 
  (hR : R > 0) 
  (ha : ∃ a, a = Real.sqrt (r * R)) :
  ∃ b c : Real,
    b^2 - (Real.sqrt (r * R) * (5 + 4 * Real.cos α)) * b + 4 * r * R * (3 + 2 * Real.cos α) = 0 ∧
    c^2 - (Real.sqrt (r * R) * (5 + 4 * Real.cos α)) * c + 4 * r * R * (3 + 2 * Real.cos α) = 0 ∧
    b ≠ c :=
by sorry

end triangle_side_lengths_l2652_265242


namespace sin_145_cos_35_l2652_265245

theorem sin_145_cos_35 :
  Real.sin (145 * π / 180) * Real.cos (35 * π / 180) = (1/2) * Real.sin (70 * π / 180) :=
by sorry

end sin_145_cos_35_l2652_265245


namespace number_of_persimmons_l2652_265262

/-- Given that there are 18 apples and the sum of apples and persimmons is 33,
    prove that the number of persimmons is 15. -/
theorem number_of_persimmons (apples : ℕ) (total : ℕ) (persimmons : ℕ) 
    (h1 : apples = 18)
    (h2 : apples + persimmons = total)
    (h3 : total = 33) :
    persimmons = 15 := by
  sorry

end number_of_persimmons_l2652_265262


namespace second_largest_of_five_consecutive_sum_90_l2652_265212

theorem second_largest_of_five_consecutive_sum_90 (a b c d e : ℕ) : 
  (a + b + c + d + e = 90) → 
  (b = a + 1) → 
  (c = b + 1) → 
  (d = c + 1) → 
  (e = d + 1) → 
  d = 19 := by
sorry

end second_largest_of_five_consecutive_sum_90_l2652_265212


namespace factorization_equality_l2652_265270

theorem factorization_equality (x : ℝ) : 84 * x^7 - 297 * x^13 = 3 * x^7 * (28 - 99 * x^6) := by
  sorry

end factorization_equality_l2652_265270


namespace probability_two_green_marbles_l2652_265255

/-- The probability of drawing two green marbles without replacement from a jar containing 5 red, 3 green, and 7 white marbles is 1/35. -/
theorem probability_two_green_marbles (red green white : ℕ) 
  (h_red : red = 5) 
  (h_green : green = 3) 
  (h_white : white = 7) : 
  (green / (red + green + white)) * ((green - 1) / (red + green + white - 1)) = 1 / 35 := by
  sorry

end probability_two_green_marbles_l2652_265255


namespace smallest_number_l2652_265272

theorem smallest_number (a b c d : ℤ) 
  (ha : a = 2023) 
  (hb : b = 2022) 
  (hc : c = -2023) 
  (hd : d = -2022) : 
  c ≤ a ∧ c ≤ b ∧ c ≤ d := by
sorry

end smallest_number_l2652_265272


namespace derivative_at_one_l2652_265202

/-- Given a function f(x) = 2k*ln(x) - x, where k is a constant, prove that f'(1) = 1 -/
theorem derivative_at_one (k : ℝ) : 
  let f := fun (x : ℝ) => 2 * k * Real.log x - x
  deriv f 1 = 1 := by
sorry

end derivative_at_one_l2652_265202


namespace cookie_sale_total_l2652_265222

theorem cookie_sale_total (raisin_cookies : ℕ) (ratio : ℚ) : 
  raisin_cookies = 42 → ratio = 6/1 → raisin_cookies + (raisin_cookies / ratio.num) = 49 :=
by sorry

end cookie_sale_total_l2652_265222


namespace brian_pencils_given_to_friend_l2652_265295

/-- 
Given that Brian initially had 39 pencils, bought 22 more, and ended up with 43 pencils,
this theorem proves that Brian gave 18 pencils to his friend.
-/
theorem brian_pencils_given_to_friend : 
  ∀ (initial_pencils bought_pencils final_pencils pencils_given : ℕ),
    initial_pencils = 39 →
    bought_pencils = 22 →
    final_pencils = 43 →
    final_pencils = initial_pencils - pencils_given + bought_pencils →
    pencils_given = 18 := by
  sorry

end brian_pencils_given_to_friend_l2652_265295


namespace sams_books_l2652_265233

theorem sams_books (joan_books : ℕ) (total_books : ℕ) (h1 : joan_books = 102) (h2 : total_books = 212) :
  total_books - joan_books = 110 := by
  sorry

end sams_books_l2652_265233


namespace two_digit_number_property_l2652_265252

theorem two_digit_number_property (a b k : ℕ) : 
  (a ≥ 1 ∧ a ≤ 9) →  -- a is a single digit (tens place)
  (b ≥ 0 ∧ b ≤ 9) →  -- b is a single digit (ones place)
  (10 * a + b = k * (a + b)) →  -- original number condition
  (10 * b + a = (13 - k) * (a + b)) →  -- interchanged digits condition
  k = 11 / 2 := by
sorry

end two_digit_number_property_l2652_265252


namespace ellipse_hyperbola_foci_l2652_265211

/-- Given an ellipse and a hyperbola with coinciding foci, prove that d^2 = 215/16 -/
theorem ellipse_hyperbola_foci (d : ℝ) : 
  (∀ x y : ℝ, x^2/25 + y^2/d^2 = 1 ↔ x^2/169 - y^2/64 = 1/16) →
  d^2 = 215/16 := by
  sorry

end ellipse_hyperbola_foci_l2652_265211


namespace triangle_properties_l2652_265264

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def condition1 (t : Triangle) : Prop :=
  1 + Real.tan t.C / Real.tan t.B = 2 * t.a / t.b

def condition2 (t : Triangle) : Prop :=
  (t.a + t.b)^2 - t.c^2 = 4

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : condition1 t) (h2 : condition2 t) : 
  t.C = Real.pi / 3 ∧ 
  ∃ (min : ℝ), min = -4 ∧ ∀ (x : ℝ), x ≥ min → 1 / t.b^2 - 3 * t.a ≥ x :=
sorry

end triangle_properties_l2652_265264


namespace intersection_of_A_and_B_l2652_265216

def A : Set ℝ := {x | ∃ k : ℤ, x = 2 * k + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 5}

theorem intersection_of_A_and_B : A ∩ B = {1, 3} := by sorry

end intersection_of_A_and_B_l2652_265216


namespace parabola_directrix_tangent_to_circle_l2652_265263

/-- The value of p for a parabola y^2 = 2px (p > 0) whose directrix is tangent to the circle (x-3)^2 + y^2 = 16 -/
theorem parabola_directrix_tangent_to_circle : 
  ∃ (p : ℝ), p > 0 ∧ 
  (∃ (x y : ℝ), y^2 = 2*p*x) ∧ 
  (∃ (x y : ℝ), (x-3)^2 + y^2 = 16) ∧
  (∃ (x : ℝ), x = -p/2 ∧ (x-3)^2 = 16) →
  p = 2 :=
by sorry

end parabola_directrix_tangent_to_circle_l2652_265263


namespace problem_solution_l2652_265274

theorem problem_solution (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (3 / x) * (2 / y) = 1 / 3) : x * y = 18 := by
  sorry

end problem_solution_l2652_265274


namespace sum_abc_equals_109610_l2652_265204

/-- Proves that given the conditions, the sum of a, b, and c is 109610 rupees -/
theorem sum_abc_equals_109610 (a b c : ℕ) : 
  (0.5 / 100 : ℚ) * a = 95 / 100 →  -- 0.5% of a equals 95 paise
  b = 3 * a - 50 →                  -- b is three times the amount of a minus 50
  c = (a - b) ^ 2 →                 -- c is the difference between a and b squared
  a > 0 →                           -- a is a positive integer
  c > 0 →                           -- c is a positive integer
  a + b + c = 109610 := by           -- The sum of a, b, and c is 109610 rupees
sorry

end sum_abc_equals_109610_l2652_265204


namespace correct_initial_distribution_l2652_265287

/-- Represents the initial and final coin counts for each person -/
structure CoinCounts where
  initial_gold : ℕ
  initial_silver : ℕ
  final_gold : ℕ
  final_silver : ℕ

/-- Represents the treasure distribution problem -/
def treasure_distribution (k : CoinCounts) (v : CoinCounts) : Prop :=
  -- Křemílek loses half of his gold coins
  k.initial_gold / 2 = k.final_gold - v.initial_gold / 3 ∧
  -- Vochomůrka loses half of his silver coins
  v.initial_silver / 2 = v.final_silver - k.initial_silver / 4 ∧
  -- Vochomůrka gives one-third of his remaining gold coins to Křemílek
  v.initial_gold * 2 / 3 = v.final_gold ∧
  -- Křemílek gives one-quarter of his silver coins to Vochomůrka
  k.initial_silver * 3 / 4 = k.final_silver ∧
  -- After exchanges, each has exactly 12 gold coins and 18 silver coins
  k.final_gold = 12 ∧ k.final_silver = 18 ∧
  v.final_gold = 12 ∧ v.final_silver = 18

/-- Theorem stating the correct initial distribution of coins -/
theorem correct_initial_distribution :
  ∃ (k v : CoinCounts),
    treasure_distribution k v ∧
    k.initial_gold = 12 ∧ k.initial_silver = 24 ∧
    v.initial_gold = 18 ∧ v.initial_silver = 24 :=
  sorry

end correct_initial_distribution_l2652_265287


namespace pencils_per_row_l2652_265281

theorem pencils_per_row (total_pencils : ℕ) (num_rows : ℕ) (pencils_per_row : ℕ) 
  (h1 : total_pencils = 32)
  (h2 : num_rows = 4)
  (h3 : total_pencils = num_rows * pencils_per_row) :
  pencils_per_row = 8 := by
  sorry

end pencils_per_row_l2652_265281


namespace fourth_degree_polynomial_abs_value_l2652_265230

/-- A fourth-degree polynomial with real coefficients -/
def fourth_degree_polynomial (f : ℝ → ℝ) : Prop :=
  ∃ a b c d e : ℝ, ∀ x, f x = a * x^4 + b * x^3 + c * x^2 + d * x + e

/-- The absolute value of f at specific points is 16 -/
def abs_value_16 (f : ℝ → ℝ) : Prop :=
  |f 1| = 16 ∧ |f 3| = 16 ∧ |f 4| = 16 ∧ |f 5| = 16 ∧ |f 7| = 16

theorem fourth_degree_polynomial_abs_value (f : ℝ → ℝ) :
  fourth_degree_polynomial f → abs_value_16 f → |f 0| = 436 := by
  sorry

end fourth_degree_polynomial_abs_value_l2652_265230


namespace expression_equality_l2652_265227

theorem expression_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 = 1 / y^2) :
  (x^2 - 1/x^2) * (y^2 + 1/y^2) = x^4 - y^4 := by
  sorry

end expression_equality_l2652_265227


namespace binary_multiplication_correct_l2652_265299

/-- Represents a binary number as a list of booleans, where true represents 1 and false represents 0. 
    The least significant bit is at the head of the list. -/
def BinaryNumber := List Bool

/-- Converts a binary number to its decimal representation -/
def binary_to_decimal (b : BinaryNumber) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

/-- Multiplies two binary numbers -/
def binary_multiply (a b : BinaryNumber) : BinaryNumber :=
  sorry -- Implementation details omitted

theorem binary_multiplication_correct :
  let a : BinaryNumber := [true, true, false, true]  -- 1011₂
  let b : BinaryNumber := [true, false, true]        -- 101₂
  let result : BinaryNumber := [true, true, true, false, true, true]  -- 110111₂
  binary_multiply a b = result ∧ 
  binary_to_decimal (binary_multiply a b) = binary_to_decimal a * binary_to_decimal b :=
by sorry

end binary_multiplication_correct_l2652_265299


namespace jordans_sister_jars_l2652_265241

def total_plums : ℕ := 240
def ripe_ratio : ℚ := 1/4
def unripe_ratio : ℚ := 3/4
def kept_unripe : ℕ := 46
def plums_per_mango : ℕ := 7
def mangoes_per_jar : ℕ := 5

theorem jordans_sister_jars : 
  ⌊(total_plums * unripe_ratio - kept_unripe + total_plums * ripe_ratio) / plums_per_mango / mangoes_per_jar⌋ = 5 := by
  sorry

end jordans_sister_jars_l2652_265241


namespace decimal_to_base5_l2652_265244

theorem decimal_to_base5 : 
  (3 * 5^2 + 2 * 5^1 + 3 * 5^0 : ℕ) = 88 := by
  sorry

end decimal_to_base5_l2652_265244


namespace vector_problem_l2652_265282

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (2, -3)

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem vector_problem (c : ℝ × ℝ) :
  perpendicular c (a.1 + b.1, a.2 + b.2) ∧
  parallel b (a.1 - c.1, a.2 - c.2) →
  c = (7/9, 7/3) :=
by sorry

end vector_problem_l2652_265282


namespace two_hundred_thousand_squared_l2652_265265

theorem two_hundred_thousand_squared : 200000 * 200000 = 40000000000 := by
  sorry

end two_hundred_thousand_squared_l2652_265265


namespace total_texts_sent_l2652_265208

/-- The number of texts Sydney sent to Allison, Brittney, and Carol over three days -/
theorem total_texts_sent (
  monday_allison monday_brittney monday_carol : ℕ)
  (tuesday_allison tuesday_brittney tuesday_carol : ℕ)
  (wednesday_allison wednesday_brittney wednesday_carol : ℕ)
  (h1 : monday_allison = 5 ∧ monday_brittney = 5 ∧ monday_carol = 5)
  (h2 : tuesday_allison = 15 ∧ tuesday_brittney = 10 ∧ tuesday_carol = 12)
  (h3 : wednesday_allison = 20 ∧ wednesday_brittney = 18 ∧ wednesday_carol = 7) :
  monday_allison + monday_brittney + monday_carol +
  tuesday_allison + tuesday_brittney + tuesday_carol +
  wednesday_allison + wednesday_brittney + wednesday_carol = 97 :=
by sorry

end total_texts_sent_l2652_265208


namespace influenza_test_probability_l2652_265247

theorem influenza_test_probability 
  (P : Set Ω → ℝ) 
  (A C : Set Ω) 
  (h1 : P (A ∩ C) / P C = 0.9)
  (h2 : P ((Cᶜ) ∩ (Aᶜ)) / P (Cᶜ) = 0.9)
  (h3 : P C = 0.005)
  : P (C ∩ A) / P A = 9 / 208 := by
  sorry

end influenza_test_probability_l2652_265247


namespace prime_simultaneous_l2652_265237

theorem prime_simultaneous (p : ℕ) : 
  Nat.Prime p ∧ Nat.Prime (8 * p^2 + 1) → p = 3 := by
  sorry

end prime_simultaneous_l2652_265237


namespace train_car_count_l2652_265259

/-- Calculates the number of cars in a train given the observed data -/
def train_cars (cars_observed : ℕ) (observation_time : ℕ) (total_time : ℕ) : ℕ :=
  (cars_observed * total_time) / observation_time

/-- Theorem stating the number of cars in the train -/
theorem train_car_count :
  let cars_observed : ℕ := 8
  let observation_time : ℕ := 12  -- in seconds
  let total_time : ℕ := 3 * 60    -- 3 minutes converted to seconds
  train_cars cars_observed observation_time total_time = 120 := by
  sorry

#eval train_cars 8 12 (3 * 60)

end train_car_count_l2652_265259


namespace basketball_price_proof_l2652_265275

/-- The price of a basketball in yuan -/
def basketball_price : ℕ := 124

/-- The price of a soccer ball in yuan -/
def soccer_ball_price : ℕ := 62

/-- The total cost of a basketball and a soccer ball in yuan -/
def total_cost : ℕ := 186

theorem basketball_price_proof :
  (basketball_price = 124) ∧
  (basketball_price + soccer_ball_price = total_cost) ∧
  (basketball_price = 2 * soccer_ball_price) :=
sorry

end basketball_price_proof_l2652_265275


namespace solve_frog_pond_l2652_265243

def frog_pond_problem (initial_frogs : ℕ) : Prop :=
  let tadpoles := 3 * initial_frogs
  let surviving_tadpoles := (2 * tadpoles) / 3
  let total_frogs := initial_frogs + surviving_tadpoles
  (total_frogs = 8) ∧ (total_frogs - 7 = 1)

theorem solve_frog_pond : ∃ (n : ℕ), frog_pond_problem n ∧ n = 2 := by
  sorry

end solve_frog_pond_l2652_265243


namespace arithmetic_sequence_formula_l2652_265203

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_formula (a : ℕ → ℤ) :
  arithmetic_sequence a → a 1 = -1 → a 2 = 1 →
  ∀ n : ℕ, a n = 2 * n - 3 := by sorry

end arithmetic_sequence_formula_l2652_265203


namespace inscribed_rectangle_circle_circumference_l2652_265200

theorem inscribed_rectangle_circle_circumference :
  ∀ (rectangle_width rectangle_height : ℝ) (circle_circumference : ℝ),
    rectangle_width = 9 →
    rectangle_height = 12 →
    (rectangle_width ^ 2 + rectangle_height ^ 2).sqrt * π = circle_circumference →
    circle_circumference = 15 * π :=
by sorry

end inscribed_rectangle_circle_circumference_l2652_265200


namespace smallest_voltage_l2652_265201

theorem smallest_voltage (a b c : ℕ) : 
  a ≤ 10 ∧ b ≤ 10 ∧ c ≤ 10 ∧
  (a + b + c : ℚ) / 3 = 4 ∧
  (a + b + c) % 5 = 0 →
  min a (min b c) = 0 :=
by sorry

end smallest_voltage_l2652_265201


namespace club_size_l2652_265269

/-- The number of committees in the club -/
def num_committees : ℕ := 5

/-- A member of the club -/
structure Member where
  committees : Finset (Fin num_committees)
  mem_two_committees : committees.card = 2

/-- The club -/
structure Club where
  members : Finset Member
  unique_pair_member : ∀ (c1 c2 : Fin num_committees), c1 ≠ c2 → 
    (members.filter (λ m => c1 ∈ m.committees ∧ c2 ∈ m.committees)).card = 1

theorem club_size (c : Club) : c.members.card = 10 := by
  sorry

end club_size_l2652_265269


namespace triangle_segment_inequality_l2652_265213

/-- Represents a configuration of points in space -/
structure PointConfiguration where
  n : ℕ
  K : ℕ
  T : ℕ
  h_n_ge_2 : n ≥ 2
  h_K_gt_1 : K > 1
  h_no_four_coplanar : True  -- This is a placeholder for the condition

/-- The main theorem -/
theorem triangle_segment_inequality (config : PointConfiguration) :
  9 * (config.T ^ 2) < 2 * (config.K ^ 3) := by
  sorry

end triangle_segment_inequality_l2652_265213


namespace additional_rows_l2652_265220

theorem additional_rows (initial_rows : ℕ) (initial_trees_per_row : ℕ) (new_trees_per_row : ℕ) :
  initial_rows = 24 →
  initial_trees_per_row = 42 →
  new_trees_per_row = 28 →
  (initial_rows * initial_trees_per_row) / new_trees_per_row - initial_rows = 12 :=
by sorry

end additional_rows_l2652_265220


namespace mixed_number_comparison_l2652_265256

theorem mixed_number_comparison : (-2 - 1/3 : ℚ) < -2.3 := by
  sorry

end mixed_number_comparison_l2652_265256


namespace tangent_product_simplification_l2652_265235

theorem tangent_product_simplification :
  (1 + Real.tan (15 * π / 180)) * (1 + Real.tan (30 * π / 180)) = 2 := by
  sorry

end tangent_product_simplification_l2652_265235


namespace condo_cats_count_l2652_265217

theorem condo_cats_count :
  ∀ (x y z : ℕ),
    x + y + z = 29 →
    x = z →
    87 = x * 1 + y * 3 + z * 5 :=
by
  sorry

end condo_cats_count_l2652_265217


namespace exists_coverable_parallelepiped_l2652_265248

/-- Represents a parallelepiped with integer dimensions -/
structure Parallelepiped where
  length : ℕ+
  width : ℕ+
  height : ℕ+

/-- Represents a square with integer side length -/
structure Square where
  side : ℕ+

/-- Checks if three squares can cover a parallelepiped with shared edges -/
def can_cover_with_shared_edges (p : Parallelepiped) (s1 s2 s3 : Square) : Prop :=
  -- The squares cover the surface area of the parallelepiped
  2 * (p.length * p.width + p.length * p.height + p.width * p.height) =
    s1.side * s1.side + s2.side * s2.side + s3.side * s3.side ∧
  -- Each pair of squares shares an edge
  (s1.side = p.length ∨ s1.side = p.width ∨ s1.side = p.height) ∧
  (s2.side = p.length ∨ s2.side = p.width ∨ s2.side = p.height) ∧
  (s3.side = p.length ∨ s3.side = p.width ∨ s3.side = p.height)

/-- Theorem stating the existence of a parallelepiped coverable by three squares with shared edges -/
theorem exists_coverable_parallelepiped :
  ∃ (p : Parallelepiped) (s1 s2 s3 : Square),
    can_cover_with_shared_edges p s1 s2 s3 :=
  sorry

end exists_coverable_parallelepiped_l2652_265248


namespace password_count_correct_l2652_265234

/-- The number of letters in the English alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of letters in the password -/
def num_password_letters : ℕ := 2

/-- The number of digits in the password -/
def num_password_digits : ℕ := 2

/-- The total number of possible passwords -/
def num_possible_passwords : ℕ := (num_letters * (num_letters - 1)) * (num_digits * (num_digits - 1))

theorem password_count_correct :
  num_possible_passwords = num_letters * (num_letters - 1) * num_digits * (num_digits - 1) := by
  sorry

end password_count_correct_l2652_265234


namespace inequality_solution_implies_a_real_l2652_265285

theorem inequality_solution_implies_a_real : 
  (∃ x : ℝ, x^2 - a*x + a ≤ 1) → a ∈ Set.univ := by sorry

end inequality_solution_implies_a_real_l2652_265285


namespace solve_probability_problem_l2652_265228

/-- Given three independent events A, B, and C with their respective probabilities -/
def probability_problem (P_A P_B P_C : ℝ) : Prop :=
  0 ≤ P_A ∧ P_A ≤ 1 ∧
  0 ≤ P_B ∧ P_B ≤ 1 ∧
  0 ≤ P_C ∧ P_C ≤ 1 →
  -- All three events occur simultaneously
  P_A * P_B * P_C = 0.612 ∧
  -- At least two events do not occur
  (1 - P_A) * (1 - P_B) * P_C +
  (1 - P_A) * P_B * (1 - P_C) +
  P_A * (1 - P_B) * (1 - P_C) +
  (1 - P_A) * (1 - P_B) * (1 - P_C) = 0.059

/-- The theorem stating the solution to the probability problem -/
theorem solve_probability_problem :
  probability_problem 0.9 0.8 0.85 := by
  sorry


end solve_probability_problem_l2652_265228


namespace dividend_calculation_l2652_265206

theorem dividend_calculation (divisor quotient remainder dividend : ℤ) :
  divisor = 800 →
  quotient = 594 →
  remainder = -968 →
  dividend = divisor * quotient + remainder →
  dividend = 474232 := by
sorry

end dividend_calculation_l2652_265206


namespace complement_of_A_in_U_l2652_265214

def U : Set ℕ := {x | 1 < x ∧ x < 5}

def A : Set ℕ := {2, 3}

theorem complement_of_A_in_U : 
  (U \ A) = {4} := by sorry

end complement_of_A_in_U_l2652_265214


namespace min_speed_to_arrive_first_l2652_265225

/-- Proves the minimum speed required for Person B to arrive before Person A --/
theorem min_speed_to_arrive_first (distance : ℝ) (speed_a : ℝ) (delay : ℝ) 
  (h1 : distance = 220)
  (h2 : speed_a = 40)
  (h3 : delay = 0.5)
  (h4 : speed_a > 0) :
  ∃ (min_speed : ℝ), 
    (∀ (speed_b : ℝ), speed_b > min_speed → 
      distance / speed_b + delay < distance / speed_a) ∧
    min_speed = 44 := by
  sorry

end min_speed_to_arrive_first_l2652_265225


namespace girls_percentage_after_boy_added_l2652_265254

theorem girls_percentage_after_boy_added (initial_boys initial_girls added_boys : ℕ) 
  (h1 : initial_boys = 11)
  (h2 : initial_girls = 13)
  (h3 : added_boys = 1) :
  (initial_girls : ℚ) / ((initial_boys + added_boys + initial_girls) : ℚ) = 52 / 100 := by
sorry

end girls_percentage_after_boy_added_l2652_265254


namespace attraction_visit_orders_l2652_265291

theorem attraction_visit_orders (n : ℕ) (h : n = 5) : 
  (n! / 2 : ℕ) = 60 := by
  sorry

end attraction_visit_orders_l2652_265291


namespace john_works_five_days_week_l2652_265273

/-- Represents John's work schedule and patient count --/
structure DoctorSchedule where
  patients_hospital1 : ℕ
  patients_hospital2 : ℕ
  total_patients_year : ℕ
  weeks_per_year : ℕ

/-- Calculates the number of days John works per week --/
def days_per_week (s : DoctorSchedule) : ℚ :=
  s.total_patients_year / (s.weeks_per_year * (s.patients_hospital1 + s.patients_hospital2))

/-- Theorem stating that John works 5 days a week --/
theorem john_works_five_days_week (s : DoctorSchedule)
  (h1 : s.patients_hospital1 = 20)
  (h2 : s.patients_hospital2 = 24)
  (h3 : s.total_patients_year = 11000)
  (h4 : s.weeks_per_year = 50) :
  days_per_week s = 5 := by
  sorry

#eval days_per_week { patients_hospital1 := 20, patients_hospital2 := 24, total_patients_year := 11000, weeks_per_year := 50 }

end john_works_five_days_week_l2652_265273


namespace min_value_expression_l2652_265236

theorem min_value_expression (x y : ℝ) :
  Real.sqrt (x^2 + y^2 - 2*x - 2*y + 2) + 
  Real.sqrt (x^2 + y^2 - 2*x + 4*y + 2*Real.sqrt 3*y + 8 + 4*Real.sqrt 3) + 
  Real.sqrt (x^2 + y^2 + 8*x + 4*Real.sqrt 3*x - 4*y + 32 + 16*Real.sqrt 3) ≥ 
  3 * Real.sqrt 6 + 4 * Real.sqrt 2 :=
by sorry

end min_value_expression_l2652_265236


namespace polynomial_equality_implies_sum_of_squares_l2652_265276

theorem polynomial_equality_implies_sum_of_squares (a b c d e f : ℤ) :
  (∀ x : ℝ, 8 * x^3 + 125 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 767 := by
  sorry

end polynomial_equality_implies_sum_of_squares_l2652_265276


namespace julia_cakes_remaining_l2652_265288

/-- 
Given:
- Julia bakes one less than 5 cakes per day
- Julia bakes for 6 days
- Clifford eats one cake every other day

Prove that Julia has 21 cakes remaining after 6 days
-/
theorem julia_cakes_remaining (cakes_per_day : ℕ) (days : ℕ) (clifford_eats : ℕ) : 
  cakes_per_day = 5 - 1 → 
  days = 6 → 
  clifford_eats = days / 2 → 
  cakes_per_day * days - clifford_eats = 21 := by
sorry

end julia_cakes_remaining_l2652_265288


namespace simplify_expression_l2652_265257

theorem simplify_expression (x : ℝ) : 1 - (2 + (1 - (1 + (2 - x)))) = 1 - x := by
  sorry

end simplify_expression_l2652_265257


namespace school_paper_usage_theorem_l2652_265280

/-- The number of sheets of paper used by a school in a week -/
def school_paper_usage (sheets_per_class_per_day : ℕ) (school_days_per_week : ℕ) (num_classes : ℕ) : ℕ :=
  sheets_per_class_per_day * school_days_per_week * num_classes

/-- Theorem stating that under given conditions, the school uses 9000 sheets of paper per week -/
theorem school_paper_usage_theorem :
  school_paper_usage 200 5 9 = 9000 := by
  sorry

end school_paper_usage_theorem_l2652_265280


namespace andreas_living_room_area_l2652_265271

/-- The area of Andrea's living room floor, given that 20% is covered by a 4ft by 9ft carpet -/
theorem andreas_living_room_area : 
  ∀ (carpet_length carpet_width carpet_area total_area : ℝ),
  carpet_length = 4 →
  carpet_width = 9 →
  carpet_area = carpet_length * carpet_width →
  carpet_area / total_area = 1/5 →
  total_area = 180 := by
sorry

end andreas_living_room_area_l2652_265271


namespace certain_number_proof_l2652_265232

theorem certain_number_proof (x y C : ℝ) : 
  (2 * x - y = C) → (6 * x - 3 * y = 12) → C = 4 := by sorry

end certain_number_proof_l2652_265232


namespace pentagon_area_sum_l2652_265239

theorem pentagon_area_sum (u v : ℤ) : 
  0 < v → v < u → (u^2 + 3*u*v = 150) → u + v = 15 := by
  sorry

#check pentagon_area_sum

end pentagon_area_sum_l2652_265239


namespace D_72_l2652_265261

/-- D(n) is the number of ways to write n as a product of factors greater than 1,
    considering the order of factors, and allowing any number of factors (at least one). -/
def D (n : ℕ) : ℕ := sorry

/-- Theorem stating that D(72) equals 93 -/
theorem D_72 : D 72 = 93 := by sorry

end D_72_l2652_265261


namespace plane_properties_l2652_265238

def plane_equation (x y z : ℝ) : ℝ := 4*x - 3*y - z - 7

def point_M : ℝ × ℝ × ℝ := (2, -1, 4)
def point_N : ℝ × ℝ × ℝ := (3, 2, -1)

def given_plane_normal : ℝ × ℝ × ℝ := (1, 1, 1)

theorem plane_properties :
  (plane_equation point_M.1 point_M.2.1 point_M.2.2 = 0) ∧
  (plane_equation point_N.1 point_N.2.1 point_N.2.2 = 0) ∧
  (4 * given_plane_normal.1 + (-3) * given_plane_normal.2.1 + (-1) * given_plane_normal.2.2 = 0) :=
by sorry

end plane_properties_l2652_265238


namespace consecutive_integers_square_difference_l2652_265246

theorem consecutive_integers_square_difference (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 2720 → (n + 1)^2 - n^2 = 103 := by
  sorry

end consecutive_integers_square_difference_l2652_265246


namespace ratio_equality_l2652_265289

theorem ratio_equality (a b : ℝ) (h1 : 4 * a = 5 * b) (h2 : a * b ≠ 0) :
  (a / 5) / (b / 4) = 1 := by
  sorry

end ratio_equality_l2652_265289


namespace cube_sum_over_product_l2652_265266

theorem cube_sum_over_product (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h_sum : x + y + z = 3) :
  (x^3 + y^3 + z^3) / (x * y * z) = 3 := by
  sorry

end cube_sum_over_product_l2652_265266


namespace isosceles_triangle_vertex_angle_l2652_265296

theorem isosceles_triangle_vertex_angle (α : ℝ) :
  α > 0 ∧ α < 180 →  -- Angle is positive and less than 180°
  50 > 0 ∧ 50 < 180 →  -- 50° is a valid angle
  α + 50 + 50 = 180 →  -- Sum of angles in a triangle is 180°
  α = 80 := by
sorry

end isosceles_triangle_vertex_angle_l2652_265296


namespace fraction_sum_simplification_l2652_265205

theorem fraction_sum_simplification :
  1 / 462 + 17 / 42 = 94 / 231 := by
sorry

end fraction_sum_simplification_l2652_265205


namespace fourth_root_equivalence_l2652_265294

theorem fourth_root_equivalence (x : ℝ) (hx : x > 0) : (x^3 * x^(1/2))^(1/4) = x^(7/8) := by
  sorry

end fourth_root_equivalence_l2652_265294


namespace triangle_cosine_sum_max_l2652_265278

theorem triangle_cosine_sum_max (A B C : ℝ) (h : Real.sin C = 2 * Real.cos A * Real.cos B) :
  ∃ (max : ℝ), max = (Real.sqrt 2 + 1) / 2 ∧ 
    ∀ (A' B' C' : ℝ), Real.sin C' = 2 * Real.cos A' * Real.cos B' →
      Real.cos A' ^ 2 + Real.cos B' ^ 2 ≤ max :=
sorry

end triangle_cosine_sum_max_l2652_265278


namespace parallel_segments_between_parallel_planes_are_equal_l2652_265293

/-- Two planes are parallel if they do not intersect -/
def ParallelPlanes (p q : Plane) : Prop := sorry

/-- A line segment between two planes -/
def LineSegmentBetweenPlanes (p q : Plane) (s : Segment) : Prop := sorry

/-- Two line segments are parallel -/
def ParallelSegments (s₁ s₂ : Segment) : Prop := sorry

/-- Two line segments are equal (have the same length) -/
def EqualSegments (s₁ s₂ : Segment) : Prop := sorry

/-- Theorem: Parallel line segments between two parallel planes are equal -/
theorem parallel_segments_between_parallel_planes_are_equal 
  (p q : Plane) (s₁ s₂ : Segment) :
  ParallelPlanes p q →
  LineSegmentBetweenPlanes p q s₁ →
  LineSegmentBetweenPlanes p q s₂ →
  ParallelSegments s₁ s₂ →
  EqualSegments s₁ s₂ := by
  sorry

end parallel_segments_between_parallel_planes_are_equal_l2652_265293


namespace hannahs_peppers_total_l2652_265253

theorem hannahs_peppers_total :
  let green_peppers : ℝ := 0.3333333333333333
  let red_peppers : ℝ := 0.4444444444444444
  let yellow_peppers : ℝ := 0.2222222222222222
  let orange_peppers : ℝ := 0.7777777777777778
  green_peppers + red_peppers + yellow_peppers + orange_peppers = 1.7777777777777777 := by
  sorry

end hannahs_peppers_total_l2652_265253


namespace pyramid_triangular_faces_area_l2652_265210

/-- The area of triangular faces of a right square-based pyramid -/
theorem pyramid_triangular_faces_area
  (base_edge : ℝ)
  (lateral_edge : ℝ)
  (h_base : base_edge = 8)
  (h_lateral : lateral_edge = 7) :
  4 * (1/2 * base_edge * Real.sqrt (lateral_edge^2 - (base_edge/2)^2)) = 16 * Real.sqrt 33 :=
by sorry

end pyramid_triangular_faces_area_l2652_265210


namespace characterization_of_M_inequality_for_M_elements_l2652_265231

-- Define the set M
def M : Set ℝ := {x | |2*x - 1| < 1}

-- Theorem 1: Characterization of set M
theorem characterization_of_M : M = {x | 0 < x ∧ x < 1} := by sorry

-- Theorem 2: Inequality for elements in M
theorem inequality_for_M_elements (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) : 
  a * b + 1 > a + b := by sorry

end characterization_of_M_inequality_for_M_elements_l2652_265231


namespace ratio_of_vectors_l2652_265267

/-- Given points O, A, B, C in a Cartesian coordinate system where O is the origin,
    prove that if OC = 2/3 * OA + 1/3 * OB, then |AC| / |AB| = 1/3 -/
theorem ratio_of_vectors (O A B C : ℝ × ℝ × ℝ) (h : C = (2/3 : ℝ) • A + (1/3 : ℝ) • B) :
  ‖C - A‖ / ‖B - A‖ = 1/3 := by sorry

end ratio_of_vectors_l2652_265267


namespace min_reciprocal_sum_l2652_265277

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 12) :
  (1 / x + 1 / y) ≥ 1 / 3 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 12 ∧ 1 / x + 1 / y = 1 / 3 :=
by sorry

end min_reciprocal_sum_l2652_265277


namespace largest_digit_divisible_by_six_l2652_265224

theorem largest_digit_divisible_by_six :
  ∃ (N : ℕ), N ≤ 9 ∧ (7218 * N) % 6 = 0 ∧ ∀ (M : ℕ), M ≤ 9 ∧ (7218 * M) % 6 = 0 → M ≤ N :=
by sorry

end largest_digit_divisible_by_six_l2652_265224


namespace fraction_powers_sum_l2652_265249

theorem fraction_powers_sum : 
  (8/9 : ℚ)^3 * (3/4 : ℚ)^3 + (1/2 : ℚ)^3 = 91/216 := by
  sorry

end fraction_powers_sum_l2652_265249


namespace least_subtraction_for_divisibility_l2652_265258

theorem least_subtraction_for_divisibility (n : ℕ) (primes : List ℕ) 
  (h_n : n = 899830)
  (h_primes : primes = [2, 3, 5, 7, 11]) : 
  ∃ (k : ℕ), 
    k = 2000 ∧ 
    (∀ m : ℕ, m < k → ¬((n - m) % (primes.prod) = 0)) ∧ 
    ((n - k) % (primes.prod) = 0) :=
by sorry

end least_subtraction_for_divisibility_l2652_265258


namespace gecko_eggs_calcification_fraction_l2652_265268

def total_eggs : ℕ := 30
def infertile_percentage : ℚ := 1/5
def hatched_eggs : ℕ := 16

theorem gecko_eggs_calcification_fraction :
  (total_eggs * (1 - infertile_percentage) - hatched_eggs) / (total_eggs * (1 - infertile_percentage)) = 1/3 := by
  sorry

end gecko_eggs_calcification_fraction_l2652_265268


namespace complex_equation_solution_l2652_265219

theorem complex_equation_solution (z : ℂ) (h : z * (1 - Complex.I) = 2) : z = 1 + Complex.I := by
  sorry

end complex_equation_solution_l2652_265219


namespace walter_school_expenses_l2652_265226

/-- Represents Walter's weekly work schedule and earnings --/
structure WalterSchedule where
  job1_weekday_hours : ℝ
  job1_weekend_hours : ℝ
  job1_hourly_rate : ℝ
  job1_weekly_bonus : ℝ
  job1_tax_rate : ℝ
  job2_hours : ℝ
  job2_hourly_rate : ℝ
  job2_tax_rate : ℝ
  job3_hours : ℝ
  job3_hourly_rate : ℝ
  school_allocation_rate : ℝ

/-- Calculates Walter's weekly school expense allocation --/
def calculateSchoolExpenses (schedule : WalterSchedule) : ℝ :=
  let job1_earnings := (schedule.job1_weekday_hours * 5 + schedule.job1_weekend_hours * 2) * schedule.job1_hourly_rate + schedule.job1_weekly_bonus
  let job1_after_tax := job1_earnings * (1 - schedule.job1_tax_rate)
  let job2_earnings := schedule.job2_hours * schedule.job2_hourly_rate
  let job2_after_tax := job2_earnings * (1 - schedule.job2_tax_rate)
  let job3_earnings := schedule.job3_hours * schedule.job3_hourly_rate
  let total_earnings := job1_after_tax + job2_after_tax + job3_earnings
  total_earnings * schedule.school_allocation_rate

/-- Theorem stating that Walter's weekly school expense allocation is approximately $211.69 --/
theorem walter_school_expenses (schedule : WalterSchedule) 
  (h1 : schedule.job1_weekday_hours = 4)
  (h2 : schedule.job1_weekend_hours = 6)
  (h3 : schedule.job1_hourly_rate = 5)
  (h4 : schedule.job1_weekly_bonus = 50)
  (h5 : schedule.job1_tax_rate = 0.1)
  (h6 : schedule.job2_hours = 5)
  (h7 : schedule.job2_hourly_rate = 7)
  (h8 : schedule.job2_tax_rate = 0.05)
  (h9 : schedule.job3_hours = 6)
  (h10 : schedule.job3_hourly_rate = 10)
  (h11 : schedule.school_allocation_rate = 0.75) :
  ∃ ε > 0, |calculateSchoolExpenses schedule - 211.69| < ε := by
  sorry

end walter_school_expenses_l2652_265226


namespace min_value_sum_reciprocals_min_value_sum_reciprocals_equality_condition_l2652_265207

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_9 : a + b + c = 9) : 
  2/a + 2/b + 2/c ≥ 2 := by
  sorry

theorem min_value_sum_reciprocals_equality_condition (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_9 : a + b + c = 9) : 
  2/a + 2/b + 2/c = 2 ↔ a = b ∧ b = c := by
  sorry

end min_value_sum_reciprocals_min_value_sum_reciprocals_equality_condition_l2652_265207


namespace remaining_black_cards_after_removal_l2652_265221

/-- Represents a deck of cards -/
structure Deck :=
  (black_cards : ℕ)

/-- Calculates the number of remaining black cards after removing some -/
def remaining_black_cards (d : Deck) (removed : ℕ) : ℕ :=
  d.black_cards - removed

/-- Theorem stating that removing 5 black cards from a deck with 26 black cards leaves 21 black cards -/
theorem remaining_black_cards_after_removal :
  ∀ (d : Deck), d.black_cards = 26 → remaining_black_cards d 5 = 21 := by
  sorry


end remaining_black_cards_after_removal_l2652_265221


namespace binomial_10_3_l2652_265251

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_l2652_265251


namespace set_equality_implies_a_value_l2652_265292

/-- Given two sets are equal, prove that a must be either 1 or -1 -/
theorem set_equality_implies_a_value (a : ℝ) : 
  ({0, -1, 2*a} : Set ℝ) = ({a-1, -abs a, a+1} : Set ℝ) → 
  a = 1 ∨ a = -1 := by
sorry

end set_equality_implies_a_value_l2652_265292


namespace unclaimed_candy_fraction_verify_actual_taken_l2652_265290

/-- Represents the fraction of candy taken by each person -/
structure CandyFraction where
  al : Rat
  bert : Rat
  carl : Rat

/-- The intended ratio for candy distribution -/
def intended_ratio : CandyFraction :=
  { al := 4/9, bert := 1/3, carl := 2/9 }

/-- The actual amount of candy taken by each person -/
def actual_taken : CandyFraction :=
  { al := 4/9, bert := 5/27, carl := 20/243 }

/-- The theorem stating the fraction of candy that goes unclaimed -/
theorem unclaimed_candy_fraction :
  1 - (actual_taken.al + actual_taken.bert + actual_taken.carl) = 230/243 := by
  sorry

/-- Verify that the actual taken amounts are correct based on the problem description -/
theorem verify_actual_taken :
  actual_taken.al = intended_ratio.al ∧
  actual_taken.bert = intended_ratio.bert * (1 - actual_taken.al) ∧
  actual_taken.carl = intended_ratio.carl * (1 - actual_taken.al - actual_taken.bert) := by
  sorry

end unclaimed_candy_fraction_verify_actual_taken_l2652_265290


namespace pool_filling_solution_l2652_265209

/-- Represents the time taken to fill a pool given two pumps with specific properties -/
def pool_filling_time (pool_volume : ℝ) : Prop :=
  ∃ (t1 t2 : ℝ),
    t1 > 0 ∧ t2 > 0 ∧
    -- First pump fills 8 hours faster than second pump
    t2 - t1 = 8 ∧
    -- Second pump initially runs for twice the time of both pumps together
    2 * (1 / (1/t1 + 1/t2)) * (1/t2) +
    -- Then both pumps run for 1.5 hours
    1.5 * (1/t1 + 1/t2) = 1 ∧
    -- Times for each pump to fill separately
    t1 = 4 ∧ t2 = 12

/-- Theorem stating the existence of a solution for the pool filling problem -/
theorem pool_filling_solution (pool_volume : ℝ) (h : pool_volume > 0) :
  pool_filling_time pool_volume :=
sorry

end pool_filling_solution_l2652_265209


namespace two_ducks_in_garden_l2652_265279

/-- The number of ducks in a garden with dogs and ducks -/
def number_of_ducks (num_dogs : ℕ) (total_feet : ℕ) : ℕ :=
  (total_feet - 4 * num_dogs) / 2

/-- Theorem: There are 2 ducks in the garden -/
theorem two_ducks_in_garden : number_of_ducks 6 28 = 2 := by
  sorry

end two_ducks_in_garden_l2652_265279


namespace seventh_root_ratio_l2652_265284

theorem seventh_root_ratio (x : ℝ) (hx : x > 0) :
  (x ^ (1/2)) / (x ^ (1/4)) = x ^ (1/4) :=
sorry

end seventh_root_ratio_l2652_265284


namespace pure_imaginary_condition_l2652_265260

theorem pure_imaginary_condition (x : ℝ) : 
  (∃ (y : ℝ), y ≠ 0 ∧ (x^2 - 1) + (x - 1)*I = y*I) → x = -1 := by
  sorry

end pure_imaginary_condition_l2652_265260


namespace lauras_garden_tulips_l2652_265298

/-- Represents a garden with tulips and lilies -/
structure Garden where
  tulips : ℕ
  lilies : ℕ

/-- Calculates the number of tulips needed to maintain a 3:4 ratio with the given number of lilies -/
def tulipsForRatio (lilies : ℕ) : ℕ :=
  (3 * lilies) / 4

/-- Represents Laura's garden before and after adding flowers -/
def lauras_garden : Garden × Garden :=
  let initial := Garden.mk (tulipsForRatio 32) 32
  let final := Garden.mk (tulipsForRatio (32 + 24)) (32 + 24)
  (initial, final)

/-- Theorem stating that after adding 24 lilies and maintaining the 3:4 ratio, 
    Laura will have 42 tulips in total -/
theorem lauras_garden_tulips : 
  (lauras_garden.2).tulips = 42 := by sorry

end lauras_garden_tulips_l2652_265298


namespace a_in_M_necessary_not_sufficient_for_a_in_N_l2652_265240

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 2}

-- Statement to prove
theorem a_in_M_necessary_not_sufficient_for_a_in_N :
  (∀ a, a ∈ N → a ∈ M) ∧ (∃ a, a ∈ M ∧ a ∉ N) := by sorry

end a_in_M_necessary_not_sufficient_for_a_in_N_l2652_265240


namespace cube_string_length_l2652_265297

theorem cube_string_length (volume : ℝ) (edge_length : ℝ) (string_length : ℝ) : 
  volume = 3375 → 
  edge_length ^ 3 = volume →
  string_length = 12 * edge_length →
  string_length = 180 := by sorry

end cube_string_length_l2652_265297


namespace area_of_S₃_l2652_265250

/-- Given a square S₁ with area 25, S₂ is constructed by bisecting the sides of S₁,
    and S₃ is constructed by bisecting the sides of S₂. -/
def square_construction (S₁ S₂ S₃ : Real → Real → Prop) : Prop :=
  (∀ x y, S₁ x y ↔ x^2 + y^2 = 25) ∧
  (∀ x y, S₂ x y ↔ ∃ a b, S₁ a b ∧ x = a/2 ∧ y = b/2) ∧
  (∀ x y, S₃ x y ↔ ∃ a b, S₂ a b ∧ x = a/2 ∧ y = b/2)

/-- The area of S₃ is 6.25 -/
theorem area_of_S₃ (S₁ S₂ S₃ : Real → Real → Prop) :
  square_construction S₁ S₂ S₃ →
  (∃ x y, S₃ x y ∧ x^2 + y^2 = 6.25) :=
sorry

end area_of_S₃_l2652_265250


namespace common_ratio_of_geometric_series_l2652_265215

def geometric_series (n : ℕ) : ℚ :=
  match n with
  | 0 => 4/7
  | 1 => 36/49
  | 2 => 324/343
  | _ => 0  -- We only need the first three terms for this problem

theorem common_ratio_of_geometric_series :
  (geometric_series 1) / (geometric_series 0) = 9/7 :=
by sorry

end common_ratio_of_geometric_series_l2652_265215
