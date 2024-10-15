import Mathlib

namespace NUMINAMATH_CALUDE_slope_determines_y_coordinate_l1876_187647

/-- Given two points R and S in a coordinate plane, if the slope of the line through R and S
    is equal to -4/3, then the y-coordinate of S is -8/3. -/
theorem slope_determines_y_coordinate (x_R y_R x_S : ℚ) : 
  let R : ℚ × ℚ := (x_R, y_R)
  let S : ℚ × ℚ := (x_S, y_S)
  x_R = -3 →
  y_R = 8 →
  x_S = 5 →
  (y_S - y_R) / (x_S - x_R) = -4/3 →
  y_S = -8/3 := by
sorry

end NUMINAMATH_CALUDE_slope_determines_y_coordinate_l1876_187647


namespace NUMINAMATH_CALUDE_complex_expression_equality_l1876_187611

theorem complex_expression_equality : 
  (Real.sqrt 3 + 5) * (5 - Real.sqrt 3) - 
  (Real.sqrt 8 + 2 * Real.sqrt (1/2)) / Real.sqrt 2 + 
  Real.sqrt ((Real.sqrt 5 - 3)^2) = 22 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l1876_187611


namespace NUMINAMATH_CALUDE_hyperbola_sum_l1876_187683

/-- Represents a hyperbola with center (h, k), focus (h, f), and vertex (h, v) -/
structure Hyperbola where
  h : ℝ
  k : ℝ
  f : ℝ
  v : ℝ

/-- The equation of the hyperbola is (y - k)²/a² - (x - h)²/b² = 1 -/
def hyperbola_equation (hyp : Hyperbola) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (y - hyp.k)^2 / a^2 - (x - hyp.h)^2 / b^2 = 1

/-- The theorem to be proved -/
theorem hyperbola_sum (hyp : Hyperbola) (a b : ℝ) :
  hyp.h = 1 ∧ hyp.k = 1 ∧ hyp.f = 7 ∧ hyp.v = -2 ∧ 
  hyperbola_equation hyp a b →
  hyp.h + hyp.k + a + b = 5 + 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l1876_187683


namespace NUMINAMATH_CALUDE_line_mb_product_l1876_187606

/-- Given a line passing through points (0, -1) and (2, -6) with equation y = mx + b,
    prove that the product mb equals 5/2. -/
theorem line_mb_product (m b : ℚ) : 
  (∀ x y : ℚ, y = m * x + b) →
  (-1 : ℚ) = b →
  (-6 : ℚ) = m * 2 + b →
  m * b = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_line_mb_product_l1876_187606


namespace NUMINAMATH_CALUDE_triangle_properties_l1876_187651

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a + b = 10 →
  c = 2 * Real.sqrt 7 →
  c * Real.sin B = Real.sqrt 3 * b * Real.cos C →
  C = π / 3 ∧ 
  (1/2) * a * b * Real.sin C = 6 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1876_187651


namespace NUMINAMATH_CALUDE_work_completion_time_l1876_187684

theorem work_completion_time (W : ℝ) (W_p W_q W_r : ℝ) :
  W_p = W_q + W_r →                -- p can do the work in the same time as q and r together
  W_p + W_q = W / 10 →             -- p and q together can complete the work in 10 days
  W_r = W / 35 →                   -- r alone needs 35 days to complete the work
  W_q = W / 28                     -- q alone needs 28 days to complete the work
  := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1876_187684


namespace NUMINAMATH_CALUDE_container_fill_fraction_l1876_187628

theorem container_fill_fraction (initial_percentage : ℝ) (added_water : ℝ) (capacity : ℝ) : 
  initial_percentage = 0.3 →
  added_water = 27 →
  capacity = 60 →
  (initial_percentage * capacity + added_water) / capacity = 0.75 := by
sorry

end NUMINAMATH_CALUDE_container_fill_fraction_l1876_187628


namespace NUMINAMATH_CALUDE_equation_solution_l1876_187656

theorem equation_solution :
  ∃! x : ℝ, (32 : ℝ) ^ (x - 2) / (8 : ℝ) ^ (x - 2) = (512 : ℝ) ^ (3 * x) ∧ x = -4/25 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1876_187656


namespace NUMINAMATH_CALUDE_impossible_tiling_l1876_187699

/-- Represents a chessboard with two opposite corners removed -/
structure ChessboardWithCornersRemoved where
  n : ℕ+  -- n is a positive natural number

/-- Represents a 2 × 1 domino -/
structure Domino

/-- Represents a tiling of the chessboard with dominoes -/
def Tiling (board : ChessboardWithCornersRemoved) := List Domino

theorem impossible_tiling (board : ChessboardWithCornersRemoved) :
  ¬ ∃ (t : Tiling board), t.length = (board.n.val ^ 2 - 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_impossible_tiling_l1876_187699


namespace NUMINAMATH_CALUDE_john_house_planks_l1876_187682

theorem john_house_planks :
  ∀ (total_nails nails_per_plank additional_nails : ℕ),
    total_nails = 11 →
    nails_per_plank = 3 →
    additional_nails = 8 →
    ∃ (num_planks : ℕ),
      num_planks * nails_per_plank + additional_nails = total_nails ∧
      num_planks = 1 := by
sorry

end NUMINAMATH_CALUDE_john_house_planks_l1876_187682


namespace NUMINAMATH_CALUDE_train_crossing_time_l1876_187646

/-- Calculates the time for a train to cross a signal pole given its length and the time it takes to cross a platform of equal length. -/
theorem train_crossing_time (train_length platform_length : ℝ) (platform_crossing_time : ℝ) 
  (h1 : train_length = 300)
  (h2 : platform_length = 300)
  (h3 : platform_crossing_time = 36) :
  let total_distance := train_length + platform_length
  let train_speed := total_distance / platform_crossing_time
  train_length / train_speed = 18 := by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1876_187646


namespace NUMINAMATH_CALUDE_permutation_remainders_l1876_187693

theorem permutation_remainders (a : Fin 11 → Fin 11) (h : Function.Bijective a) :
  ∃ i j : Fin 11, i ≠ j ∧ (i.val + 1) * (a i).val ≡ (j.val + 1) * (a j).val [ZMOD 11] := by
  sorry

end NUMINAMATH_CALUDE_permutation_remainders_l1876_187693


namespace NUMINAMATH_CALUDE_potato_sales_total_weight_l1876_187687

theorem potato_sales_total_weight :
  let morning_sales : ℕ := 29
  let afternoon_sales : ℕ := 17
  let bag_weight : ℕ := 7
  let total_bags : ℕ := morning_sales + afternoon_sales
  let total_weight : ℕ := total_bags * bag_weight
  total_weight = 322 := by sorry

end NUMINAMATH_CALUDE_potato_sales_total_weight_l1876_187687


namespace NUMINAMATH_CALUDE_another_divisor_l1876_187695

theorem another_divisor (smallest_number : ℕ) : 
  smallest_number = 44402 →
  (smallest_number + 2) % 12 = 0 →
  (smallest_number + 2) % 30 = 0 →
  (smallest_number + 2) % 48 = 0 →
  (smallest_number + 2) % 74 = 0 →
  (smallest_number + 2) % 22202 = 0 := by
sorry

end NUMINAMATH_CALUDE_another_divisor_l1876_187695


namespace NUMINAMATH_CALUDE_selling_price_calculation_l1876_187659

theorem selling_price_calculation (cost_price : ℝ) (profit_percentage : ℝ) :
  cost_price = 180 →
  profit_percentage = 15 →
  cost_price + (cost_price * (profit_percentage / 100)) = 207 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l1876_187659


namespace NUMINAMATH_CALUDE_b_is_geometric_sequence_l1876_187612

-- Define the geometric sequence a_n
def a (n : ℕ) (a₁ q : ℝ) : ℝ := a₁ * q^(n - 1)

-- Define the sequence b_n
def b (n : ℕ) (a₁ q : ℝ) : ℝ := a (3*n - 2) a₁ q + a (3*n - 1) a₁ q + a (3*n) a₁ q

-- Theorem statement
theorem b_is_geometric_sequence (a₁ q : ℝ) (hq : q ≠ 1) :
  ∀ n : ℕ, b (n + 1) a₁ q = (b n a₁ q) * q^3 :=
sorry

end NUMINAMATH_CALUDE_b_is_geometric_sequence_l1876_187612


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l1876_187681

theorem sufficient_condition_for_inequality (x : ℝ) :
  1 < x ∧ x < 2 → (x + 1) / (x - 1) > 2 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l1876_187681


namespace NUMINAMATH_CALUDE_order_of_abc_l1876_187665

theorem order_of_abc (a b c : ℝ) (ha : a = 2^(4/3)) (hb : b = 3^(2/3)) (hc : c = 25^(1/3)) :
  b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_order_of_abc_l1876_187665


namespace NUMINAMATH_CALUDE_polynomial_functional_equation_l1876_187645

/-- Given a polynomial P : ℝ × ℝ → ℝ satisfying P(x - 1, y - 2x + 1) = P(x, y) for all x and y,
    there exists a polynomial Φ : ℝ → ℝ such that P(x, y) = Φ(y - x^2) for all x and y. -/
theorem polynomial_functional_equation
  (P : ℝ → ℝ → ℝ)
  (h : ∀ x y : ℝ, P (x - 1) (y - 2*x + 1) = P x y)
  : ∃ Φ : ℝ → ℝ, ∀ x y : ℝ, P x y = Φ (y - x^2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_functional_equation_l1876_187645


namespace NUMINAMATH_CALUDE_cupcakes_per_package_l1876_187609

theorem cupcakes_per_package
  (initial_cupcakes : ℕ)
  (eaten_cupcakes : ℕ)
  (num_packages : ℕ)
  (h1 : initial_cupcakes = 18)
  (h2 : eaten_cupcakes = 8)
  (h3 : num_packages = 5)
  : (initial_cupcakes - eaten_cupcakes) / num_packages = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_cupcakes_per_package_l1876_187609


namespace NUMINAMATH_CALUDE_greatest_abdba_divisible_by_13_l1876_187652

def is_valid_abdba (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧
  ∃ (a b d : ℕ),
    a < 10 ∧ b < 10 ∧ d < 10 ∧
    a ≠ b ∧ b ≠ d ∧ a ≠ d ∧
    n = a * 10000 + b * 1000 + d * 100 + b * 10 + a

theorem greatest_abdba_divisible_by_13 :
  ∀ n : ℕ, is_valid_abdba n → n % 13 = 0 → n ≤ 96769 :=
by sorry

end NUMINAMATH_CALUDE_greatest_abdba_divisible_by_13_l1876_187652


namespace NUMINAMATH_CALUDE_volunteers_distribution_l1876_187619

theorem volunteers_distribution (n : ℕ) (k : ℕ) : 
  n = 6 → k = 4 → (k^n - k * (k-1)^n + (k.choose 2) * (k-2)^n) = 1564 := by
  sorry

end NUMINAMATH_CALUDE_volunteers_distribution_l1876_187619


namespace NUMINAMATH_CALUDE_k_value_in_set_union_l1876_187689

theorem k_value_in_set_union (A B : Set ℕ) (k : ℕ) :
  A = {1, 2, k} →
  B = {1, 2, 3, 5} →
  A ∪ B = {1, 2, 3, 5} →
  k = 3 ∨ k = 5 := by
  sorry

end NUMINAMATH_CALUDE_k_value_in_set_union_l1876_187689


namespace NUMINAMATH_CALUDE_milton_zoology_books_l1876_187632

theorem milton_zoology_books :
  ∀ (z b : ℕ),
  z + b = 80 →
  b = 4 * z →
  z = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_milton_zoology_books_l1876_187632


namespace NUMINAMATH_CALUDE_min_center_value_l1876_187635

def RegularOctagon (vertices : Fin 8 → ℕ) (center : ℕ) :=
  (∀ i j : Fin 8, i ≠ j → vertices i ≠ vertices j) ∧
  (vertices 0 + vertices 1 + vertices 4 + vertices 5 + center =
   vertices 1 + vertices 2 + vertices 5 + vertices 6 + center) ∧
  (vertices 2 + vertices 3 + vertices 6 + vertices 7 + center =
   vertices 3 + vertices 0 + vertices 7 + vertices 4 + center) ∧
  (vertices 0 + vertices 1 + vertices 2 + vertices 3 +
   vertices 4 + vertices 5 + vertices 6 + vertices 7 =
   vertices 0 + vertices 1 + vertices 4 + vertices 5 + center)

theorem min_center_value (vertices : Fin 8 → ℕ) (center : ℕ) 
  (h : RegularOctagon vertices center) :
  center ≥ 14 := by
  sorry

end NUMINAMATH_CALUDE_min_center_value_l1876_187635


namespace NUMINAMATH_CALUDE_sam_eating_period_l1876_187698

def apples_per_sandwich : ℕ := 4
def sandwiches_per_day : ℕ := 10
def total_apples : ℕ := 280

theorem sam_eating_period :
  (total_apples / (apples_per_sandwich * sandwiches_per_day) : ℕ) = 7 :=
sorry

end NUMINAMATH_CALUDE_sam_eating_period_l1876_187698


namespace NUMINAMATH_CALUDE_place_value_ratio_l1876_187675

-- Define the number
def number : ℝ := 58624.0791

-- Define the place value of 6 (thousands)
def place_value_6 : ℝ := 1000

-- Define the place value of 7 (tenths)
def place_value_7 : ℝ := 0.1

-- Theorem statement
theorem place_value_ratio :
  place_value_6 / place_value_7 = 10000 := by
  sorry

end NUMINAMATH_CALUDE_place_value_ratio_l1876_187675


namespace NUMINAMATH_CALUDE_circle_intersection_length_l1876_187679

-- Define the right triangle ABC
structure RightTriangle where
  A : Real
  B : Real
  C : Real
  angle_A : A = 30 * Real.pi / 180
  hypotenuse : C = 2 * A
  right_angle : B = 90 * Real.pi / 180

-- Define the circle and point K
structure CircleAndPoint (t : RightTriangle) where
  K : Real
  on_hypotenuse : K ≤ t.C ∧ K ≥ 0
  diameter : t.A = 2

-- Theorem statement
theorem circle_intersection_length (t : RightTriangle) (c : CircleAndPoint t) :
  let CK := Real.sqrt (t.A * (t.C - c.K))
  CK = 1 := by sorry

end NUMINAMATH_CALUDE_circle_intersection_length_l1876_187679


namespace NUMINAMATH_CALUDE_notebook_dispatch_l1876_187614

theorem notebook_dispatch (x y : ℕ) 
  (h1 : x * (y + 5) = x * y + 1250) 
  (h2 : (x + 7) * y = x * y + 3150) : 
  x + y = 700 := by
  sorry

end NUMINAMATH_CALUDE_notebook_dispatch_l1876_187614


namespace NUMINAMATH_CALUDE_not_all_prime_l1876_187610

theorem not_all_prime (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_div_a : a ∣ b + c + b * c)
  (h_div_b : b ∣ c + a + c * a)
  (h_div_c : c ∣ a + b + a * b) :
  ¬(Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c) := by
  sorry

end NUMINAMATH_CALUDE_not_all_prime_l1876_187610


namespace NUMINAMATH_CALUDE_christopher_karen_difference_l1876_187662

/-- Proves that Christopher has $8.00 more than Karen given their quarter counts -/
theorem christopher_karen_difference :
  let karen_quarters : ℕ := 32
  let christopher_quarters : ℕ := 64
  let quarter_value : ℚ := 1/4
  let karen_money := karen_quarters * quarter_value
  let christopher_money := christopher_quarters * quarter_value
  christopher_money - karen_money = 8 :=
by sorry

end NUMINAMATH_CALUDE_christopher_karen_difference_l1876_187662


namespace NUMINAMATH_CALUDE_multiples_between_200_and_500_l1876_187669

def count_multiples (lower upper lcm : ℕ) : ℕ :=
  (upper / lcm) - ((lower - 1) / lcm)

theorem multiples_between_200_and_500 : count_multiples 200 500 36 = 8 := by
  sorry

end NUMINAMATH_CALUDE_multiples_between_200_and_500_l1876_187669


namespace NUMINAMATH_CALUDE_wider_can_radius_l1876_187616

/-- Given two cylindrical cans with the same volume, where the height of one can is five times
    the height of the other, and the radius of the narrower can is 10 units,
    prove that the radius of the wider can is 10√5 units. -/
theorem wider_can_radius (h : ℝ) (volume : ℝ) : 
  volume = π * 10^2 * (5 * h) → 
  volume = π * ((10 * Real.sqrt 5)^2) * h := by
  sorry

end NUMINAMATH_CALUDE_wider_can_radius_l1876_187616


namespace NUMINAMATH_CALUDE_fraction_less_than_mode_l1876_187631

def data_list : List ℕ := [3, 4, 5, 5, 5, 5, 6, 6, 7, 8, 10, 11, 15, 21, 23, 26, 27]

def mode (l : List ℕ) : ℕ := 
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

def count_less_than_mode (l : List ℕ) : ℕ :=
  l.filter (· < mode l) |>.length

theorem fraction_less_than_mode :
  (count_less_than_mode data_list : ℚ) / data_list.length = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_less_than_mode_l1876_187631


namespace NUMINAMATH_CALUDE_t_range_l1876_187650

theorem t_range (t α β a : ℝ) :
  (t = Real.cos β ^ 3 + (α / 2) * Real.cos β) →
  (a ≤ t) →
  (t ≤ α - 5 * Real.cos β) →
  (-2/3 ≤ t ∧ t ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_t_range_l1876_187650


namespace NUMINAMATH_CALUDE_odd_function_zero_value_necessary_not_sufficient_l1876_187605

-- Define what it means for a function to be odd
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_zero_value_necessary_not_sufficient :
  (∀ f : ℝ → ℝ, IsOdd f → f 0 = 0) ∧
  (∃ g : ℝ → ℝ, g 0 = 0 ∧ ¬IsOdd g) :=
sorry

end NUMINAMATH_CALUDE_odd_function_zero_value_necessary_not_sufficient_l1876_187605


namespace NUMINAMATH_CALUDE_f_has_one_real_root_l1876_187667

-- Define the polynomial
def f (x : ℝ) : ℝ := (x - 4) * (x^2 + 4*x + 5)

-- Theorem statement
theorem f_has_one_real_root : ∃! x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_has_one_real_root_l1876_187667


namespace NUMINAMATH_CALUDE_peter_age_approx_l1876_187621

def cindy_age : ℕ := 5

def jan_age : ℕ := cindy_age + 2

def marcia_age : ℕ := 2 * jan_age

def greg_age : ℕ := marcia_age + 2

def bobby_age : ℕ := (3 * greg_age) / 2

noncomputable def peter_age : ℝ := 2 * Real.sqrt (bobby_age : ℝ)

theorem peter_age_approx : 
  ∀ ε > 0, |peter_age - 10| < ε := by sorry

end NUMINAMATH_CALUDE_peter_age_approx_l1876_187621


namespace NUMINAMATH_CALUDE_smallest_self_descriptive_number_l1876_187691

/-- Represents the value of a letter in the alphabet (A=1, B=2, ..., Z=26) -/
def letter_value (c : Char) : ℕ :=
  match c with
  | 'A' => 1 | 'B' => 2 | 'C' => 3 | 'D' => 4 | 'E' => 5 | 'F' => 6 | 'G' => 7
  | 'H' => 8 | 'I' => 9 | 'J' => 10 | 'K' => 11 | 'L' => 12 | 'M' => 13 | 'N' => 14
  | 'O' => 15 | 'P' => 16 | 'Q' => 17 | 'R' => 18 | 'S' => 19 | 'T' => 20
  | 'U' => 21 | 'V' => 22 | 'W' => 23 | 'X' => 24 | 'Y' => 25 | 'Z' => 26
  | _ => 0

/-- Calculates the sum of letter values in a string -/
def string_value (s : String) : ℕ :=
  s.toList.map letter_value |>.sum

/-- Converts a number to its written-out form in French -/
def number_to_french (n : ℕ) : String :=
  match n with
  | 222 => "DEUXCENTVINGTDEUX"
  | _ => ""  -- We only need to define 222 for this problem

theorem smallest_self_descriptive_number :
  ∀ n : ℕ, n < 222 → string_value (number_to_french n) ≠ n ∧
  string_value (number_to_french 222) = 222 := by
  sorry

#eval string_value (number_to_french 222)  -- Should output 222

end NUMINAMATH_CALUDE_smallest_self_descriptive_number_l1876_187691


namespace NUMINAMATH_CALUDE_distance_AB_l1876_187639

def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (-1, 6)

theorem distance_AB : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_AB_l1876_187639


namespace NUMINAMATH_CALUDE_point_on_number_line_l1876_187686

/-- Given two points A and B on a number line where A represents -3 and B is 7 units to the right of A, 
    prove that B represents 4. -/
theorem point_on_number_line (A B : ℝ) : A = -3 ∧ B = A + 7 → B = 4 := by
  sorry

end NUMINAMATH_CALUDE_point_on_number_line_l1876_187686


namespace NUMINAMATH_CALUDE_sets_equivalence_l1876_187653

-- Define the sets
def A : Set ℝ := {1}
def B : Set ℝ := {y : ℝ | (y - 1)^2 = 0}
def D : Set ℝ := {x : ℝ | x - 1 = 0}

-- C is not defined as a set because it's not a valid set notation

theorem sets_equivalence :
  (A = B) ∧ (A = D) ∧ (B = D) :=
sorry

-- Note: We can't include C in the theorem because it's not a valid set

end NUMINAMATH_CALUDE_sets_equivalence_l1876_187653


namespace NUMINAMATH_CALUDE_seven_valid_configurations_l1876_187615

/-- Represents a square piece --/
structure Square :=
  (label : Char)

/-- Represents the T-shaped figure --/
structure TShape

/-- Represents a configuration of squares added to the T-shape --/
structure Configuration :=
  (square1 : Square)
  (square2 : Square)

/-- Checks if a configuration can be folded into a closed cubical box --/
def can_fold_into_cube (config : Configuration) : Prop :=
  sorry

/-- The set of all possible configurations --/
def all_configurations (squares : Finset Square) : Finset Configuration :=
  sorry

/-- The set of valid configurations that can be folded into a cube --/
def valid_configurations (squares : Finset Square) : Finset Configuration :=
  sorry

theorem seven_valid_configurations :
  ∀ (t : TShape) (squares : Finset Square),
    (Finset.card squares = 8) →
    (Finset.card (valid_configurations squares) = 7) :=
  sorry

end NUMINAMATH_CALUDE_seven_valid_configurations_l1876_187615


namespace NUMINAMATH_CALUDE_dark_tile_fraction_is_five_sixteenths_l1876_187604

/-- Represents a tiled floor with a repeating pattern of dark tiles -/
structure TiledFloor :=
  (size : ℕ)  -- Size of the square floor (number of tiles per side)
  (dark_tiles_per_section : ℕ)  -- Number of dark tiles in each 4x4 section
  (total_tiles_per_section : ℕ)  -- Total number of tiles in each 4x4 section

/-- The fraction of dark tiles on the floor -/
def dark_tile_fraction (floor : TiledFloor) : ℚ :=
  (floor.dark_tiles_per_section : ℚ) / (floor.total_tiles_per_section : ℚ)

/-- Theorem stating that the fraction of dark tiles is 5/16 -/
theorem dark_tile_fraction_is_five_sixteenths (floor : TiledFloor) 
  (h1 : floor.size > 0)
  (h2 : floor.dark_tiles_per_section = 5)
  (h3 : floor.total_tiles_per_section = 16) : 
  dark_tile_fraction floor = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_dark_tile_fraction_is_five_sixteenths_l1876_187604


namespace NUMINAMATH_CALUDE_three_Z_five_l1876_187697

/-- The operation Z defined on real numbers -/
def Z (a b : ℝ) : ℝ := b + 7*a - 3*a^2

/-- Theorem stating that 3 Z 5 = -1 -/
theorem three_Z_five : Z 3 5 = -1 := by
  sorry

end NUMINAMATH_CALUDE_three_Z_five_l1876_187697


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l1876_187618

def M : Set ℤ := {-1, 0, 2, 4}
def N : Set ℤ := {0, 2, 3, 4}

theorem union_of_M_and_N : M ∪ N = {-1, 0, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l1876_187618


namespace NUMINAMATH_CALUDE_point_on_line_l1876_187608

/-- Given a line passing through points (0,4) and (-6,1), prove that s = 6 
    is the unique solution such that (s,7) lies on this line. -/
theorem point_on_line (s : ℝ) : 
  (∃! x : ℝ, (x - 0) / (-6 - 0) = (7 - 4) / (x - 0) ∧ x = s) → s = 6 :=
by sorry

end NUMINAMATH_CALUDE_point_on_line_l1876_187608


namespace NUMINAMATH_CALUDE_average_multiples_of_6_up_to_100_l1876_187641

def multiples_of_6 (n : ℕ) : Finset ℕ :=
  Finset.filter (fun x => x % 6 = 0) (Finset.range (n + 1))

theorem average_multiples_of_6_up_to_100 :
  let S := multiples_of_6 100
  (S.sum id) / S.card = 51 := by
  sorry

end NUMINAMATH_CALUDE_average_multiples_of_6_up_to_100_l1876_187641


namespace NUMINAMATH_CALUDE_circle_area_equality_l1876_187625

theorem circle_area_equality (r₁ r₂ r₃ : ℝ) : 
  r₁ = 13 → r₂ = 23 → π * r₃^2 = π * (r₂^2 - r₁^2) → r₃ = 6 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_circle_area_equality_l1876_187625


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1876_187668

-- Define the universe set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | x ≤ 2}

-- Define set N
def N : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

-- State the theorem
theorem intersection_complement_equality :
  M ∩ (U \ N) = {x : ℝ | x < -1 ∨ (1 < x ∧ x ≤ 2)} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1876_187668


namespace NUMINAMATH_CALUDE_tangent_points_distance_l1876_187677

theorem tangent_points_distance (r : ℝ) (d : ℝ) (h1 : r = 7) (h2 : d = 25) :
  let tangent_length := Real.sqrt (d^2 - r^2)
  2 * tangent_length = 48 :=
sorry

end NUMINAMATH_CALUDE_tangent_points_distance_l1876_187677


namespace NUMINAMATH_CALUDE_watermelons_with_seeds_l1876_187629

theorem watermelons_with_seeds (ripe : ℕ) (unripe : ℕ) (seedless : ℕ) : 
  ripe = 11 → unripe = 13 → seedless = 15 → 
  ripe + unripe - seedless = 9 := by
  sorry

end NUMINAMATH_CALUDE_watermelons_with_seeds_l1876_187629


namespace NUMINAMATH_CALUDE_valid_bases_for_346_l1876_187685

def is_valid_base (b : ℕ) : Prop :=
  b > 1 ∧ b^3 ≤ 346 ∧ 346 < b^4 ∧
  ∃ (d₃ d₂ d₁ d₀ : ℕ), 
    d₃ * b^3 + d₂ * b^2 + d₁ * b^1 + d₀ * b^0 = 346 ∧
    d₃ ≠ 0 ∧ d₀ % 2 = 0

theorem valid_bases_for_346 :
  ∀ b : ℕ, is_valid_base b ↔ (b = 6 ∨ b = 7) :=
sorry

end NUMINAMATH_CALUDE_valid_bases_for_346_l1876_187685


namespace NUMINAMATH_CALUDE_kishore_savings_theorem_l1876_187666

/-- Represents Mr. Kishore's financial situation --/
structure KishoreFinances where
  rent : ℕ
  milk : ℕ
  groceries : ℕ
  education : ℕ
  petrol : ℕ
  miscellaneous : ℕ
  savings_percentage : ℚ

/-- Calculates the total expenses --/
def total_expenses (k : KishoreFinances) : ℕ :=
  k.rent + k.milk + k.groceries + k.education + k.petrol + k.miscellaneous

/-- Calculates the monthly salary --/
def monthly_salary (k : KishoreFinances) : ℚ :=
  (total_expenses k : ℚ) / (1 - k.savings_percentage)

/-- Calculates the savings amount --/
def savings_amount (k : KishoreFinances) : ℚ :=
  k.savings_percentage * monthly_salary k

/-- Theorem: Mr. Kishore's savings are approximately 2683.33 Rs. --/
theorem kishore_savings_theorem (k : KishoreFinances) 
  (h1 : k.rent = 5000)
  (h2 : k.milk = 1500)
  (h3 : k.groceries = 4500)
  (h4 : k.education = 2500)
  (h5 : k.petrol = 2000)
  (h6 : k.miscellaneous = 5650)
  (h7 : k.savings_percentage = 1/10) :
  ∃ (ε : ℚ), abs (savings_amount k - 2683.33) < ε ∧ ε < 1/100 := by
  sorry

end NUMINAMATH_CALUDE_kishore_savings_theorem_l1876_187666


namespace NUMINAMATH_CALUDE_curve_is_circle_l1876_187602

-- Define the curve
def curve (x y : ℝ) : Prop := x^2 + y^2 = 16

-- Define what a circle is
def is_circle (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    S = {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Theorem statement
theorem curve_is_circle :
  is_circle {p : ℝ × ℝ | curve p.1 p.2} :=
sorry

end NUMINAMATH_CALUDE_curve_is_circle_l1876_187602


namespace NUMINAMATH_CALUDE_dow_decrease_l1876_187688

def initial_dow : ℝ := 8900
def percentage_decrease : ℝ := 0.02

theorem dow_decrease (initial : ℝ) (decrease : ℝ) :
  initial = initial_dow →
  decrease = percentage_decrease →
  initial * (1 - decrease) = 8722 :=
by sorry

end NUMINAMATH_CALUDE_dow_decrease_l1876_187688


namespace NUMINAMATH_CALUDE_cuboid_lateral_surface_area_l1876_187644

/-- The lateral surface area of a cuboid with given dimensions -/
def lateralSurfaceArea (length width height : ℝ) : ℝ :=
  2 * (length * height + width * height)

/-- Theorem: The lateral surface area of a cuboid with length 10 m, width 14 m, and height 18 m is 864 m² -/
theorem cuboid_lateral_surface_area :
  lateralSurfaceArea 10 14 18 = 864 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_lateral_surface_area_l1876_187644


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1876_187633

/-- The eccentricity of a hyperbola with asymptote tangent to a specific circle is 2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1) →
  (∃ (x y : ℝ), (x - Real.sqrt 3)^2 + (y - 1)^2 = 1) →
  (∃ (m c : ℝ), ∀ (x y : ℝ), y = m*x + c → 
    ((x - Real.sqrt 3)^2 + (y - 1)^2 = 1 ∧ 
     (∃ (t : ℝ), x = a*t ∧ y = b*t))) →
  a / Real.sqrt (a^2 - b^2) = 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1876_187633


namespace NUMINAMATH_CALUDE_sqrt_3_minus_pi_squared_l1876_187617

theorem sqrt_3_minus_pi_squared (π : ℝ) (h : π > 3) : 
  Real.sqrt ((3 - π)^2) = π - 3 := by
sorry

end NUMINAMATH_CALUDE_sqrt_3_minus_pi_squared_l1876_187617


namespace NUMINAMATH_CALUDE_move_fulcrum_towards_wood_l1876_187658

/-- Represents the material of a sphere -/
inductive Material
| CastIron
| Wood

/-- Properties of a sphere -/
structure Sphere where
  material : Material
  density : ℝ
  volume : ℝ
  mass : ℝ

/-- The setup of the balance problem -/
structure BalanceSetup where
  airDensity : ℝ
  castIronSphere : Sphere
  woodenSphere : Sphere
  fulcrumPosition : ℝ  -- 0 means middle, negative means towards cast iron, positive means towards wood

/-- Conditions for the balance problem -/
def validSetup (setup : BalanceSetup) : Prop :=
  setup.castIronSphere.material = Material.CastIron ∧
  setup.woodenSphere.material = Material.Wood ∧
  setup.castIronSphere.density > setup.woodenSphere.density ∧
  setup.castIronSphere.density > setup.airDensity ∧
  setup.woodenSphere.density > setup.airDensity ∧
  setup.castIronSphere.volume < setup.woodenSphere.volume ∧
  setup.castIronSphere.mass < setup.woodenSphere.mass

/-- The balance condition when the fulcrum is in the middle -/
def balanceCondition (setup : BalanceSetup) : Prop :=
  (setup.castIronSphere.density - setup.airDensity) * setup.castIronSphere.volume =
  (setup.woodenSphere.density - setup.airDensity) * setup.woodenSphere.volume

/-- Theorem stating that the fulcrum needs to be moved towards the wooden sphere -/
theorem move_fulcrum_towards_wood (setup : BalanceSetup) :
  validSetup setup → balanceCondition setup → setup.fulcrumPosition > 0 := by
  sorry

end NUMINAMATH_CALUDE_move_fulcrum_towards_wood_l1876_187658


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1876_187674

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ -1 ∧ x^2 - x + 2 ≠ 0 →
  (x^2 + 2*x - 8) / (x^3 - x - 2) = 
  (-9/4) / (x + 1) + (13/4 * x - 7/2) / (x^2 - x + 2) := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1876_187674


namespace NUMINAMATH_CALUDE_wall_building_time_l1876_187620

theorem wall_building_time
  (workers_initial : ℕ)
  (days_initial : ℕ)
  (workers_new : ℕ)
  (h1 : workers_initial = 60)
  (h2 : days_initial = 3)
  (h3 : workers_new = 30)
  (h4 : workers_initial > 0)
  (h5 : workers_new > 0)
  (h6 : days_initial > 0) :
  let days_new := workers_initial * days_initial / workers_new
  days_new = 6 :=
by sorry

end NUMINAMATH_CALUDE_wall_building_time_l1876_187620


namespace NUMINAMATH_CALUDE_circle_line_distance_range_l1876_187676

theorem circle_line_distance_range (b : ℝ) : 
  (∃! (p q : ℝ × ℝ), 
    p.1^2 + p.2^2 = 4 ∧ 
    q.1^2 + q.2^2 = 4 ∧ 
    (p ≠ q) ∧
    (|p.2 - p.1 - b| / Real.sqrt 2 = 1) ∧
    (|q.2 - q.1 - b| / Real.sqrt 2 = 1)) →
  (b < -Real.sqrt 2 ∧ b > -3 * Real.sqrt 2) ∨ 
  (b > Real.sqrt 2 ∧ b < 3 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_circle_line_distance_range_l1876_187676


namespace NUMINAMATH_CALUDE_max_value_of_f_l1876_187664

/-- Given a function f(x) = x^3 - 3ax + 2 where x = 2 is an extremum point,
    prove that the maximum value of f(x) is 18 -/
theorem max_value_of_f (a : ℝ) (f : ℝ → ℝ) (h1 : f = fun x ↦ x^3 - 3*a*x + 2) 
    (h2 : ∃ (ε : ℝ), ε > 0 ∧ ∀ x ∈ Set.Ioo (2 - ε) (2 + ε), f x ≥ f 2 ∨ f x ≤ f 2) :
  (⨆ x, f x) = 18 := by
  sorry


end NUMINAMATH_CALUDE_max_value_of_f_l1876_187664


namespace NUMINAMATH_CALUDE_second_number_proof_l1876_187660

theorem second_number_proof (d : ℕ) (n₁ n₂ x : ℕ) : 
  d = 16 →
  n₁ = 25 →
  n₂ = 105 →
  x = 41 →
  x > n₁ →
  x % d = n₁ % d →
  x % d = n₂ % d →
  ∀ y : ℕ, n₁ < y ∧ y < x → y % d ≠ n₁ % d ∨ y % d ≠ n₂ % d :=
by sorry

end NUMINAMATH_CALUDE_second_number_proof_l1876_187660


namespace NUMINAMATH_CALUDE_f_properties_l1876_187622

noncomputable def f (x : ℝ) : ℝ := x^3 - 1/x^3

theorem f_properties :
  (∀ x > 0, f (-x) = -f x) ∧
  (∀ a b, 0 < a → a < b → f a < f b) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1876_187622


namespace NUMINAMATH_CALUDE_one_sixth_of_twelve_x_plus_six_l1876_187607

theorem one_sixth_of_twelve_x_plus_six (x : ℝ) : (1 / 6) * (12 * x + 6) = 2 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_one_sixth_of_twelve_x_plus_six_l1876_187607


namespace NUMINAMATH_CALUDE_milkshake_hours_l1876_187627

/-- Given that Augustus makes 3 milkshakes per hour and Luna makes 7 milkshakes per hour,
    prove that they have been making milkshakes for 8 hours when they have made 80 milkshakes in total. -/
theorem milkshake_hours (augustus_rate : ℕ) (luna_rate : ℕ) (total_milkshakes : ℕ) (hours : ℕ) :
  augustus_rate = 3 →
  luna_rate = 7 →
  total_milkshakes = 80 →
  augustus_rate * hours + luna_rate * hours = total_milkshakes →
  hours = 8 := by
sorry

end NUMINAMATH_CALUDE_milkshake_hours_l1876_187627


namespace NUMINAMATH_CALUDE_solve_for_y_l1876_187623

theorem solve_for_y (x y : ℝ) (h1 : x^(3*y) = 8) (h2 : x = 2) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1876_187623


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_quadratic_equation_l1876_187626

theorem arithmetic_geometric_mean_quadratic_equation 
  (a b : ℝ) 
  (h_arithmetic_mean : (a + b) / 2 = 8) 
  (h_geometric_mean : Real.sqrt (a * b) = 15) : 
  ∀ x : ℝ, x^2 - 16*x + 225 = 0 ↔ (x = a ∨ x = b) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_quadratic_equation_l1876_187626


namespace NUMINAMATH_CALUDE_min_distance_to_line_l1876_187694

theorem min_distance_to_line (x y : ℝ) (h : x + y - 3 = 0) :
  ∃ (min : ℝ), min = Real.sqrt 2 ∧
  ∀ (x' y' : ℝ), x' + y' - 3 = 0 →
  min ≤ Real.sqrt ((x' - 2)^2 + (y' + 1)^2) :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l1876_187694


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1876_187648

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 - 3*x + 2 < 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - 3*x + 2 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1876_187648


namespace NUMINAMATH_CALUDE_polynomial_relationship_l1876_187613

def x : Fin 5 → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | 3 => 4
  | 4 => 5

def y : Fin 5 → ℕ
  | 0 => 1
  | 1 => 4
  | 2 => 9
  | 3 => 16
  | 4 => 25

theorem polynomial_relationship : ∀ i : Fin 5, y i = (x i) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_relationship_l1876_187613


namespace NUMINAMATH_CALUDE_fixed_point_of_line_l1876_187654

/-- The fixed point through which the line (2k+1)x+(k-1)y+(7-k)=0 passes for all real k -/
theorem fixed_point_of_line (k : ℝ) : 
  ∃! p : ℝ × ℝ, ∀ k : ℝ, (2*k + 1) * p.1 + (k - 1) * p.2 + (7 - k) = 0 :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_of_line_l1876_187654


namespace NUMINAMATH_CALUDE_population_reaches_max_in_180_years_l1876_187624

-- Define the initial conditions
def initial_year : ℕ := 2023
def island_area : ℕ := 31500
def land_per_person : ℕ := 2
def initial_population : ℕ := 250
def doubling_period : ℕ := 30

-- Define the maximum sustainable population
def max_population : ℕ := island_area / land_per_person

-- Define the population growth function
def population_after_years (years : ℕ) : ℕ :=
  initial_population * (2 ^ (years / doubling_period))

-- Theorem statement
theorem population_reaches_max_in_180_years :
  ∃ (years : ℕ), years = 180 ∧ 
  population_after_years years ≥ max_population ∧
  population_after_years (years - doubling_period) < max_population :=
sorry

end NUMINAMATH_CALUDE_population_reaches_max_in_180_years_l1876_187624


namespace NUMINAMATH_CALUDE_next_common_day_l1876_187696

def dance_interval : ℕ := 6
def karate_interval : ℕ := 12
def library_interval : ℕ := 18

theorem next_common_day (dance_interval karate_interval library_interval : ℕ) :
  dance_interval = 6 → karate_interval = 12 → library_interval = 18 →
  Nat.lcm (Nat.lcm dance_interval karate_interval) library_interval = 36 :=
by sorry

end NUMINAMATH_CALUDE_next_common_day_l1876_187696


namespace NUMINAMATH_CALUDE_playground_fundraiser_correct_l1876_187630

def playground_fundraiser (johnson_amount sutton_amount rollin_amount total_amount : ℝ) : Prop :=
  johnson_amount = 2300 ∧
  johnson_amount = 2 * sutton_amount ∧
  rollin_amount = 8 * sutton_amount ∧
  rollin_amount = total_amount / 3 ∧
  total_amount * 0.98 = 27048

theorem playground_fundraiser_correct :
  ∃ johnson_amount sutton_amount rollin_amount total_amount : ℝ,
    playground_fundraiser johnson_amount sutton_amount rollin_amount total_amount :=
by
  sorry

end NUMINAMATH_CALUDE_playground_fundraiser_correct_l1876_187630


namespace NUMINAMATH_CALUDE_bag_slips_problem_l1876_187670

theorem bag_slips_problem (total_slips : ℕ) (num1 num2 : ℕ) (expected_value : ℚ) :
  total_slips = 15 →
  num1 = 3 →
  num2 = 8 →
  expected_value = 5 →
  ∃ (slips_with_num1 : ℕ),
    slips_with_num1 ≤ total_slips ∧
    (slips_with_num1 : ℚ) / total_slips * num1 + 
    ((total_slips - slips_with_num1) : ℚ) / total_slips * num2 = expected_value →
    slips_with_num1 = 9 :=
by sorry

end NUMINAMATH_CALUDE_bag_slips_problem_l1876_187670


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1876_187678

theorem arithmetic_calculation : 1325 + 180 / 60 * 3 - 225 = 1109 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1876_187678


namespace NUMINAMATH_CALUDE_no_polyhedron_with_area_2015_l1876_187600

/-- Represents a polyhedron constructed from unit cubes -/
structure Polyhedron where
  num_cubes : ℕ
  num_glued_faces : ℕ

/-- Calculates the surface area of a polyhedron -/
def surface_area (p : Polyhedron) : ℕ :=
  6 * p.num_cubes - 2 * p.num_glued_faces

/-- Theorem stating the impossibility of constructing a polyhedron with surface area 2015 -/
theorem no_polyhedron_with_area_2015 :
  ∀ p : Polyhedron, surface_area p ≠ 2015 := by
  sorry


end NUMINAMATH_CALUDE_no_polyhedron_with_area_2015_l1876_187600


namespace NUMINAMATH_CALUDE_visible_shaded_area_coefficient_sum_l1876_187640

/-- Represents the visible shaded area of a grid with circles on top. -/
def visibleShadedArea (gridSize : ℕ) (smallCircleCount : ℕ) (smallCircleDiameter : ℝ) 
  (largeCircleCount : ℕ) (largeCircleDiameter : ℝ) : ℝ := by sorry

/-- The sum of coefficients A and B in the expression A - Bπ for the visible shaded area. -/
def coefficientSum (gridSize : ℕ) (smallCircleCount : ℕ) (smallCircleDiameter : ℝ) 
  (largeCircleCount : ℕ) (largeCircleDiameter : ℝ) : ℝ := by sorry

theorem visible_shaded_area_coefficient_sum :
  coefficientSum 6 5 1 1 4 = 41.25 := by sorry

end NUMINAMATH_CALUDE_visible_shaded_area_coefficient_sum_l1876_187640


namespace NUMINAMATH_CALUDE_min_value_theorem_l1876_187690

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1/y) * (x + 1/y - 1000) + (y + 1/x) * (y + 1/x - 1000) ≥ -500000 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1876_187690


namespace NUMINAMATH_CALUDE_pie_eating_contest_l1876_187655

/-- The amount of pie Erik ate -/
def eriks_pie : ℝ := 0.67

/-- The amount of pie Frank ate -/
def franks_pie : ℝ := 0.33

/-- The difference between Erik's and Frank's pie consumption -/
def pie_difference : ℝ := eriks_pie - franks_pie

/-- Theorem stating that the difference between Erik's and Frank's pie consumption is 0.34 -/
theorem pie_eating_contest : pie_difference = 0.34 := by
  sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l1876_187655


namespace NUMINAMATH_CALUDE_parabola_focus_l1876_187642

/-- The focus of the parabola x^2 = 8y has coordinates (0, 2) -/
theorem parabola_focus (x y : ℝ) :
  (x^2 = 8*y) → (∃ p : ℝ, p = 2 ∧ (0, p) = (0, 2)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_l1876_187642


namespace NUMINAMATH_CALUDE_ramanujan_number_l1876_187636

def hardy : ℂ := Complex.mk 7 4

theorem ramanujan_number (r : ℂ) : r * hardy = Complex.mk 60 (-18) → r = Complex.mk (174/65) (-183/65) := by
  sorry

end NUMINAMATH_CALUDE_ramanujan_number_l1876_187636


namespace NUMINAMATH_CALUDE_trig_identity_l1876_187637

theorem trig_identity (α : ℝ) (h : Real.tan α = 3) :
  2 * (Real.sin α)^2 - (Real.sin α) * (Real.cos α) + (Real.cos α)^2 = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1876_187637


namespace NUMINAMATH_CALUDE_base_ten_proof_l1876_187649

/-- Converts a number from base b to decimal --/
def to_decimal (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Converts a number from decimal to base b --/
def from_decimal (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Checks if the equation 162_b + 235_b = 407_b holds for a given base b --/
def equation_holds (b : ℕ) : Prop :=
  to_decimal 162 b + to_decimal 235 b = to_decimal 407 b

theorem base_ten_proof :
  ∃! b : ℕ, b > 1 ∧ equation_holds b ∧ b = 10 :=
sorry

end NUMINAMATH_CALUDE_base_ten_proof_l1876_187649


namespace NUMINAMATH_CALUDE_square_eq_four_solutions_l1876_187638

theorem square_eq_four_solutions (x : ℝ) : (x - 1)^2 = 4 ↔ x = 3 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_square_eq_four_solutions_l1876_187638


namespace NUMINAMATH_CALUDE_smoking_lung_cancer_study_l1876_187692

theorem smoking_lung_cancer_study (confidence : Real) 
  (h1 : confidence = 0.99) : 
  let error_probability := 1 - confidence
  error_probability ≤ 0.01 := by
sorry

end NUMINAMATH_CALUDE_smoking_lung_cancer_study_l1876_187692


namespace NUMINAMATH_CALUDE_hundred_to_fifty_zeros_l1876_187671

theorem hundred_to_fifty_zeros (n : ℕ) : 100^50 = 10^100 := by
  sorry

end NUMINAMATH_CALUDE_hundred_to_fifty_zeros_l1876_187671


namespace NUMINAMATH_CALUDE_money_distribution_l1876_187643

theorem money_distribution (a b c : ℕ) 
  (total : a + b + c = 700)
  (ac_sum : a + c = 300)
  (bc_sum : b + c = 600) :
  c = 200 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l1876_187643


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1876_187657

theorem necessary_but_not_sufficient : 
  (∀ x : ℝ, x > 3 → x > 1) ∧ 
  (∃ x : ℝ, x > 1 ∧ ¬(x > 3)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1876_187657


namespace NUMINAMATH_CALUDE_x_plus_y_equals_three_l1876_187673

theorem x_plus_y_equals_three (x y : ℝ) (h : |x - 1| + (y - 2)^2 = 0) : x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_three_l1876_187673


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l1876_187672

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 15) (h2 : x * y = 36) : 
  1 / x + 1 / y = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l1876_187672


namespace NUMINAMATH_CALUDE_class_size_l1876_187603

/-- The number of students who borrowed at least 3 books -/
def R : ℕ := 16

/-- The total number of students in the class -/
def S : ℕ := 42

theorem class_size :
  (∃ (R : ℕ),
    (0 * 2 + 1 * 12 + 2 * 12 + 3 * R) / S = 2 ∧
    S = 2 + 12 + 12 + R) →
  S = 42 := by sorry

end NUMINAMATH_CALUDE_class_size_l1876_187603


namespace NUMINAMATH_CALUDE_min_blocks_for_specific_wall_l1876_187661

/-- Represents the dimensions of a wall --/
structure WallDimensions where
  length : ℕ
  height : ℕ

/-- Represents the dimensions of a block --/
structure BlockDimensions where
  length : ℕ
  height : ℕ

/-- Calculates the minimum number of blocks needed to build a wall --/
def minBlocksNeeded (wall : WallDimensions) (blocks : List BlockDimensions) : ℕ :=
  sorry

/-- Theorem: The minimum number of blocks needed for the specified wall is 404 --/
theorem min_blocks_for_specific_wall :
  let wall : WallDimensions := ⟨120, 8⟩
  let blocks : List BlockDimensions := [⟨2, 1⟩, ⟨3, 1⟩, ⟨1, 1⟩]
  minBlocksNeeded wall blocks = 404 :=
by
  sorry

#check min_blocks_for_specific_wall

end NUMINAMATH_CALUDE_min_blocks_for_specific_wall_l1876_187661


namespace NUMINAMATH_CALUDE_min_a1_arithmetic_sequence_l1876_187634

def arithmetic_sequence (a : ℕ → ℕ) := ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem min_a1_arithmetic_sequence (a : ℕ → ℕ) 
  (h_arith : arithmetic_sequence a) 
  (h_pos : ∀ n, a n > 0)
  (h_a9 : a 9 = 2023) :
  a 1 ≥ 7 ∧ ∃ b : ℕ → ℕ, arithmetic_sequence b ∧ (∀ n, b n > 0) ∧ b 9 = 2023 ∧ b 1 = 7 :=
sorry

end NUMINAMATH_CALUDE_min_a1_arithmetic_sequence_l1876_187634


namespace NUMINAMATH_CALUDE_system_solution_l1876_187680

theorem system_solution :
  let s : Set (ℚ × ℚ) := {(1/2, 5), (1, 3), (3/2, 2), (5/2, 1)}
  ∀ x y : ℚ, (2*x + y + 2*x*y = 11 ∧ 2*x^2*y + x*y^2 = 15) ↔ (x, y) ∈ s := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1876_187680


namespace NUMINAMATH_CALUDE_quadratic_root_value_l1876_187663

theorem quadratic_root_value (n : ℝ) : n^2 - 5*n + 4 = 0 → n^2 - 5*n = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l1876_187663


namespace NUMINAMATH_CALUDE_hcl_required_l1876_187601

-- Define the chemical reaction
structure Reaction where
  hcl : ℕ  -- moles of Hydrochloric acid
  koh : ℕ  -- moles of Potassium hydroxide
  h2o : ℕ  -- moles of Water
  kcl : ℕ  -- moles of Potassium chloride

-- Define the balanced equation
def balanced_equation (r : Reaction) : Prop :=
  r.hcl = r.koh ∧ r.hcl = r.h2o ∧ r.hcl = r.kcl

-- Define the given conditions
def given_conditions (r : Reaction) : Prop :=
  r.koh = 2 ∧ r.h2o = 2 ∧ r.kcl = 2

-- Theorem to prove
theorem hcl_required (r : Reaction) 
  (h1 : balanced_equation r) 
  (h2 : given_conditions r) : 
  r.hcl = 2 := by
  sorry

#check hcl_required

end NUMINAMATH_CALUDE_hcl_required_l1876_187601
