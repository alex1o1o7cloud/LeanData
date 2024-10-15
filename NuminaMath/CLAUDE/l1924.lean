import Mathlib

namespace NUMINAMATH_CALUDE_monotonic_decreasing_implies_a_leq_one_l1924_192403

def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*a*x + 1

theorem monotonic_decreasing_implies_a_leq_one (a : ℝ) :
  (∀ x y, 1 ≤ x ∧ x < y ∧ y ≤ 2 → f a x > f a y) →
  a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_implies_a_leq_one_l1924_192403


namespace NUMINAMATH_CALUDE_square_root_of_product_l1924_192494

theorem square_root_of_product : Real.sqrt ((90 + 6) * (90 - 6)) = 90 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_product_l1924_192494


namespace NUMINAMATH_CALUDE_negative_square_opposite_l1924_192450

-- Define opposite numbers
def opposite (a b : ℤ) : Prop := a = -b

-- Theorem statement
theorem negative_square_opposite : opposite (-2^2) ((-2)^2) := by
  sorry

end NUMINAMATH_CALUDE_negative_square_opposite_l1924_192450


namespace NUMINAMATH_CALUDE_walter_bus_time_l1924_192482

def wake_up_time : Nat := 6 * 60
def leave_time : Nat := 7 * 60
def return_time : Nat := 16 * 60 + 30
def num_classes : Nat := 7
def class_duration : Nat := 45
def lunch_duration : Nat := 45
def additional_time : Nat := 90

def total_away_time : Nat := return_time - leave_time
def total_school_time : Nat := num_classes * class_duration + lunch_duration + additional_time

theorem walter_bus_time :
  total_away_time - total_school_time = 120 := by
  sorry

end NUMINAMATH_CALUDE_walter_bus_time_l1924_192482


namespace NUMINAMATH_CALUDE_angle_of_inclination_negative_slope_one_l1924_192441

/-- The angle of inclination of a line given by the equation x + y + 3 = 0 is 3π/4 -/
theorem angle_of_inclination_negative_slope_one (x y : ℝ) :
  x + y + 3 = 0 → Real.arctan (-1) = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_of_inclination_negative_slope_one_l1924_192441


namespace NUMINAMATH_CALUDE_perp_plane_necessary_not_sufficient_l1924_192465

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation for planes and lines
variable (perp_planes : Plane → Plane → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- Define the "line in plane" relation
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem perp_plane_necessary_not_sufficient
  (α β : Plane) (m : Line)
  (h_diff : α ≠ β)
  (h_m_in_α : line_in_plane m α) :
  (∀ m, line_in_plane m α → perp_line_plane m β → perp_planes α β) ∧
  ¬(perp_planes α β → perp_line_plane m β) :=
sorry

end NUMINAMATH_CALUDE_perp_plane_necessary_not_sufficient_l1924_192465


namespace NUMINAMATH_CALUDE_divisibility_theorem_l1924_192474

theorem divisibility_theorem (n : ℕ) (h : n ≥ 1) :
  ∃ (a b : ℤ), (n : ℤ) ∣ (4 * a^2 + 9 * b^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l1924_192474


namespace NUMINAMATH_CALUDE_simplify_fraction_l1924_192488

theorem simplify_fraction (a : ℝ) : 
  (1 + a^2 / (1 + 2*a)) / ((1 + a) / (1 + 2*a)) = 1 + a :=
by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1924_192488


namespace NUMINAMATH_CALUDE_model4_best_fitting_l1924_192499

-- Define the structure for a regression model
structure RegressionModel where
  name : String
  r_squared : Float

-- Define the principle of better fitting
def better_fitting (m1 m2 : RegressionModel) : Prop :=
  m1.r_squared > m2.r_squared

-- Define the four models
def model1 : RegressionModel := ⟨"Model 1", 0.55⟩
def model2 : RegressionModel := ⟨"Model 2", 0.65⟩
def model3 : RegressionModel := ⟨"Model 3", 0.79⟩
def model4 : RegressionModel := ⟨"Model 4", 0.95⟩

-- Define a list of all models
def all_models : List RegressionModel := [model1, model2, model3, model4]

-- Theorem: Model 4 has the best fitting effect
theorem model4_best_fitting :
  ∀ m ∈ all_models, m ≠ model4 → better_fitting model4 m :=
by sorry

end NUMINAMATH_CALUDE_model4_best_fitting_l1924_192499


namespace NUMINAMATH_CALUDE_hexagon_side_sum_l1924_192400

/-- A polygon with six vertices -/
structure Hexagon :=
  (P Q R S T U : ℝ × ℝ)

/-- The area of a polygon -/
def area (h : Hexagon) : ℝ := sorry

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Theorem: For a hexagon PQRSTU with area 40, PQ = 6, QR = 7, and TU = 4, ST + TU = 7 -/
theorem hexagon_side_sum (h : Hexagon) 
  (h_area : area h = 40)
  (h_PQ : distance h.P h.Q = 6)
  (h_QR : distance h.Q h.R = 7)
  (h_TU : distance h.T h.U = 4) :
  distance h.S h.T + distance h.T h.U = 7 := by sorry

end NUMINAMATH_CALUDE_hexagon_side_sum_l1924_192400


namespace NUMINAMATH_CALUDE_total_trees_planted_l1924_192431

/-- The total number of trees planted by a family in spring -/
theorem total_trees_planted (apricot peach cherry : ℕ) : 
  apricot = 58 →
  peach = 3 * apricot →
  cherry = 5 * peach →
  apricot + peach + cherry = 1102 := by
  sorry

end NUMINAMATH_CALUDE_total_trees_planted_l1924_192431


namespace NUMINAMATH_CALUDE_butterflies_in_garden_l1924_192462

theorem butterflies_in_garden (initial : ℕ) (remaining : ℕ) : 
  remaining = 6 ∧ 3 * remaining = 2 * initial → initial = 9 := by
  sorry

end NUMINAMATH_CALUDE_butterflies_in_garden_l1924_192462


namespace NUMINAMATH_CALUDE_alices_preferred_number_l1924_192481

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem alices_preferred_number (n : ℕ) 
  (h1 : 70 < n ∧ n < 140)
  (h2 : n % 13 = 0)
  (h3 : n % 3 ≠ 0)
  (h4 : sum_of_digits n % 4 = 0) :
  n = 130 := by
sorry

end NUMINAMATH_CALUDE_alices_preferred_number_l1924_192481


namespace NUMINAMATH_CALUDE_total_dress_cost_l1924_192483

theorem total_dress_cost (pauline_dress : ℕ) (h1 : pauline_dress = 30)
  (jean_dress : ℕ) (h2 : jean_dress = pauline_dress - 10)
  (ida_dress : ℕ) (h3 : ida_dress = jean_dress + 30)
  (patty_dress : ℕ) (h4 : patty_dress = ida_dress + 10) :
  pauline_dress + jean_dress + ida_dress + patty_dress = 160 := by
sorry

end NUMINAMATH_CALUDE_total_dress_cost_l1924_192483


namespace NUMINAMATH_CALUDE_binary_conversion_l1924_192423

-- Define the binary number
def binary_num : List Nat := [1, 0, 1, 1, 0, 0, 1]

-- Define the function to convert binary to decimal
def binary_to_decimal (bin : List Nat) : Nat :=
  bin.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

-- Define the function to convert decimal to octal
def decimal_to_octal (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
  aux n []

-- Theorem statement
theorem binary_conversion :
  binary_to_decimal binary_num = 89 ∧
  decimal_to_octal (binary_to_decimal binary_num) = [1, 3, 1] := by
  sorry

end NUMINAMATH_CALUDE_binary_conversion_l1924_192423


namespace NUMINAMATH_CALUDE_journey_distance_l1924_192498

theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_time = 10)
  (h2 : speed1 = 21)
  (h3 : speed2 = 24) :
  ∃ (distance : ℝ), 
    distance = 224 ∧ 
    total_time = (distance / 2) / speed1 + (distance / 2) / speed2 :=
by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l1924_192498


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2_to_2014_l1924_192454

/-- The number of terms in an arithmetic sequence -/
def arithmetic_sequence_length (a₁ aₙ d : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

/-- Theorem: The arithmetic sequence starting with 2, ending with 2014, 
    and having a common difference of 4 contains exactly 504 terms -/
theorem arithmetic_sequence_2_to_2014 : 
  arithmetic_sequence_length 2 2014 4 = 504 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2_to_2014_l1924_192454


namespace NUMINAMATH_CALUDE_cube_rotation_theorem_l1924_192484

/-- Represents the orientation of a picture on the top face of a cube -/
inductive PictureOrientation
| Original
| Rotated90
| Rotated180

/-- Represents a cube with a picture on its top face -/
structure Cube :=
  (orientation : PictureOrientation)

/-- Represents the action of rolling a cube across its edges -/
def roll (c : Cube) : Cube :=
  sorry

/-- Represents a sequence of rolls that returns the cube to its original position -/
def rollSequence (c : Cube) : Cube :=
  sorry

theorem cube_rotation_theorem (c : Cube) :
  (∃ (seq : Cube → Cube), seq c = Cube.mk PictureOrientation.Rotated180) ∧
  (∀ (seq : Cube → Cube), seq c ≠ Cube.mk PictureOrientation.Rotated90) :=
sorry

end NUMINAMATH_CALUDE_cube_rotation_theorem_l1924_192484


namespace NUMINAMATH_CALUDE_butter_price_is_correct_l1924_192411

/-- Represents the milk and butter sales problem --/
structure MilkButterSales where
  milk_price : ℚ
  milk_to_butter_ratio : ℚ
  num_cows : ℕ
  milk_per_cow : ℚ
  num_customers : ℕ
  milk_per_customer : ℚ
  total_earnings : ℚ

/-- Calculates the price per stick of butter --/
def butter_price (s : MilkButterSales) : ℚ :=
  let total_milk := s.num_cows * s.milk_per_cow
  let milk_sold := s.num_customers * s.milk_per_customer
  let milk_for_butter := total_milk - milk_sold
  let butter_sticks := milk_for_butter * s.milk_to_butter_ratio
  let milk_earnings := milk_sold * s.milk_price
  let butter_earnings := s.total_earnings - milk_earnings
  butter_earnings / butter_sticks

/-- Theorem stating that the butter price is $1.50 given the problem conditions --/
theorem butter_price_is_correct (s : MilkButterSales) 
  (h1 : s.milk_price = 3)
  (h2 : s.milk_to_butter_ratio = 2)
  (h3 : s.num_cows = 12)
  (h4 : s.milk_per_cow = 4)
  (h5 : s.num_customers = 6)
  (h6 : s.milk_per_customer = 6)
  (h7 : s.total_earnings = 144) :
  butter_price s = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_butter_price_is_correct_l1924_192411


namespace NUMINAMATH_CALUDE_remainder_theorem_polynomial_division_remainder_l1924_192414

def f (x : ℝ) : ℝ := 5 * x^3 - 10 * x^2 + 15 * x - 20

def divisor (x : ℝ) : ℝ := 5 * x - 10

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - a) * q x + f a := by sorry

theorem polynomial_division_remainder :
  ∃ q : ℝ → ℝ, ∀ x, f x = divisor x * q x + 10 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_polynomial_division_remainder_l1924_192414


namespace NUMINAMATH_CALUDE_x_fourth_minus_inverse_fourth_l1924_192490

theorem x_fourth_minus_inverse_fourth (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/x^4 = 527 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_minus_inverse_fourth_l1924_192490


namespace NUMINAMATH_CALUDE_original_triangle_area_l1924_192409

theorem original_triangle_area (original_side : ℝ) (new_side : ℝ) (new_area : ℝ) :
  new_side = 5 * original_side →
  new_area = 125 →
  (original_side^2 * Real.sqrt 3) / 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_original_triangle_area_l1924_192409


namespace NUMINAMATH_CALUDE_banknote_probability_l1924_192452

/-- Represents a bag of banknotes -/
structure Bag :=
  (ten : ℕ)    -- Number of ten-yuan banknotes
  (five : ℕ)   -- Number of five-yuan banknotes
  (one : ℕ)    -- Number of one-yuan banknotes

/-- Calculate the total value of banknotes in a bag -/
def bagValue (b : Bag) : ℕ :=
  10 * b.ten + 5 * b.five + b.one

/-- Calculate the number of ways to choose 2 items from n items -/
def choose2 (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- The probability of drawing at least one 5-yuan note from bag B -/
def probAtLeastOne5 (b : Bag) : ℚ :=
  (choose2 b.five.succ + b.five * b.one) / choose2 (b.five + b.one)

theorem banknote_probability :
  let bagA : Bag := ⟨2, 0, 3⟩
  let bagB : Bag := ⟨0, 4, 3⟩
  let totalDraws := choose2 (bagValue bagA) * choose2 (bagValue bagB)
  let favorableDraws := choose2 bagA.one * (choose2 bagB.five.succ + bagB.five * bagB.one)
  (favorableDraws : ℚ) / totalDraws = 9 / 35 := by
  sorry

end NUMINAMATH_CALUDE_banknote_probability_l1924_192452


namespace NUMINAMATH_CALUDE_ascending_six_digit_numbers_count_l1924_192405

/-- The number of six-digit natural numbers with digits in ascending order -/
def ascending_six_digit_numbers : ℕ :=
  Nat.choose 9 3

theorem ascending_six_digit_numbers_count : ascending_six_digit_numbers = 84 := by
  sorry

end NUMINAMATH_CALUDE_ascending_six_digit_numbers_count_l1924_192405


namespace NUMINAMATH_CALUDE_rectangle_area_l1924_192472

/-- Given a rectangle with diagonal length x and length three times its width, 
    prove that its area is (3/10)x^2 -/
theorem rectangle_area (x : ℝ) (h : x > 0) : 
  ∃ w : ℝ, w > 0 ∧ 
    w^2 + (3*w)^2 = x^2 ∧ 
    3 * w^2 = (3/10) * x^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1924_192472


namespace NUMINAMATH_CALUDE_parallel_vectors_y_value_l1924_192497

theorem parallel_vectors_y_value (a b : ℝ × ℝ) :
  a = (6, 2) →
  b.2 = 3 →
  (∃ k : ℝ, k ≠ 0 ∧ a = k • b) →
  b.1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_y_value_l1924_192497


namespace NUMINAMATH_CALUDE_hexagon_to_rhombus_l1924_192451

/-- A regular hexagon -/
structure RegularHexagon where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- A rhombus -/
structure Rhombus where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- A part of the hexagon after cutting -/
structure HexagonPart where
  area : ℝ
  area_pos : area > 0

/-- Function to cut a regular hexagon into three parts -/
def cut_hexagon (h : RegularHexagon) : (HexagonPart × HexagonPart × HexagonPart) :=
  sorry

/-- Function to form a rhombus from three hexagon parts -/
def form_rhombus (p1 p2 p3 : HexagonPart) : Rhombus :=
  sorry

/-- Theorem stating that a regular hexagon can be cut into three parts that form a rhombus -/
theorem hexagon_to_rhombus (h : RegularHexagon) :
  ∃ (p1 p2 p3 : HexagonPart), 
    let (p1', p2', p3') := cut_hexagon h
    p1 = p1' ∧ p2 = p2' ∧ p3 = p3' ∧
    ∃ (r : Rhombus), r = form_rhombus p1 p2 p3 :=
  sorry

end NUMINAMATH_CALUDE_hexagon_to_rhombus_l1924_192451


namespace NUMINAMATH_CALUDE_bag_of_balls_l1924_192426

/-- Given a bag of balls, prove that the total number of balls is 15 -/
theorem bag_of_balls (total_balls : ℕ) 
  (prob_red : ℚ) 
  (num_red : ℕ) : 
  prob_red = 1/5 → 
  num_red = 3 → 
  total_balls = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_bag_of_balls_l1924_192426


namespace NUMINAMATH_CALUDE_smallest_difference_fractions_l1924_192424

theorem smallest_difference_fractions :
  ∃ (x y a b : ℤ),
    (0 < x) ∧ (x < 8) ∧ (0 < y) ∧ (y < 13) ∧
    (0 < a) ∧ (a < 8) ∧ (0 < b) ∧ (b < 13) ∧
    (x / 8 ≠ y / 13) ∧ (a / 8 ≠ b / 13) ∧
    (|x / 8 - y / 13| = |13 * x - 8 * y| / 104) ∧
    (|a / 8 - b / 13| = |13 * a - 8 * b| / 104) ∧
    (|13 * x - 8 * y| = 1) ∧ (|13 * a - 8 * b| = 1) ∧
    ∀ (p q : ℤ), (0 < p) → (p < 8) → (0 < q) → (q < 13) → (p / 8 ≠ q / 13) →
      |p / 8 - q / 13| ≥ |x / 8 - y / 13| ∧
      |p / 8 - q / 13| ≥ |a / 8 - b / 13| :=
by
  sorry

#check smallest_difference_fractions

end NUMINAMATH_CALUDE_smallest_difference_fractions_l1924_192424


namespace NUMINAMATH_CALUDE_gummy_bear_spending_percentage_l1924_192470

-- Define the given constants
def hourly_rate : ℚ := 12.5
def hours_worked : ℕ := 40
def tax_rate : ℚ := 0.2
def remaining_money : ℚ := 340

-- Define the function to calculate the percentage spent on gummy bears
def gummy_bear_percentage (rate : ℚ) (hours : ℕ) (tax : ℚ) (remaining : ℚ) : ℚ :=
  let gross_pay := rate * hours
  let net_pay := gross_pay * (1 - tax)
  let spent_on_gummy_bears := net_pay - remaining
  (spent_on_gummy_bears / net_pay) * 100

-- Theorem statement
theorem gummy_bear_spending_percentage :
  gummy_bear_percentage hourly_rate hours_worked tax_rate remaining_money = 15 :=
sorry

end NUMINAMATH_CALUDE_gummy_bear_spending_percentage_l1924_192470


namespace NUMINAMATH_CALUDE_pencil_count_l1924_192440

/-- Given an initial number of pencils and a number of pencils added, 
    calculate the total number of pencils after addition. -/
def total_pencils (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem stating that given 33 initial pencils and 27 added pencils, 
    the total number of pencils is 60. -/
theorem pencil_count : total_pencils 33 27 = 60 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l1924_192440


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l1924_192413

theorem min_value_theorem (x : ℝ) (h : x > -3) :
  2 * x + 1 / (x + 3) ≥ 2 * Real.sqrt 2 - 6 :=
by sorry

theorem min_value_achievable :
  ∃ x : ℝ, x > -3 ∧ 2 * x + 1 / (x + 3) = 2 * Real.sqrt 2 - 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l1924_192413


namespace NUMINAMATH_CALUDE_min_value_theorem_l1924_192433

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2 * x + 3 * y = 8) :
  (2 / x + 3 / y) ≥ 25 / 8 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 2 * x + 3 * y = 8 ∧ 2 / x + 3 / y = 25 / 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1924_192433


namespace NUMINAMATH_CALUDE_jessie_weight_loss_l1924_192486

/-- Calculates the weight loss given initial and current weights -/
def weight_loss (initial_weight current_weight : ℕ) : ℕ :=
  initial_weight - current_weight

/-- Proves that Jessie's weight loss is 7 kilograms -/
theorem jessie_weight_loss : weight_loss 74 67 = 7 := by
  sorry

end NUMINAMATH_CALUDE_jessie_weight_loss_l1924_192486


namespace NUMINAMATH_CALUDE_oranges_in_basket_l1924_192415

/-- The number of oranges in a fruit basket -/
def num_oranges : ℕ := 6

/-- The number of apples in the fruit basket -/
def num_apples : ℕ := num_oranges - 2

/-- The number of bananas in the fruit basket -/
def num_bananas : ℕ := 3 * num_apples

/-- The number of peaches in the fruit basket -/
def num_peaches : ℕ := num_bananas / 2

/-- Theorem: The number of oranges in the fruit basket is 6 -/
theorem oranges_in_basket : 
  num_oranges + num_apples + num_bananas + num_peaches = 28 → num_oranges = 6 := by
  sorry


end NUMINAMATH_CALUDE_oranges_in_basket_l1924_192415


namespace NUMINAMATH_CALUDE_cricket_matches_count_l1924_192402

theorem cricket_matches_count (total_average : ℝ) (first_four_average : ℝ) (last_three_average : ℝ) :
  total_average = 56 →
  first_four_average = 46 →
  last_three_average = 69.33333333333333 →
  ∃ (n : ℕ), n = 7 ∧ n * total_average = 4 * first_four_average + 3 * last_three_average :=
by sorry

end NUMINAMATH_CALUDE_cricket_matches_count_l1924_192402


namespace NUMINAMATH_CALUDE_book_loss_percentage_l1924_192417

/-- Proves the loss percentage on a book given specific conditions --/
theorem book_loss_percentage
  (total_cost : ℝ)
  (cost_book1 : ℝ)
  (gain_percentage : ℝ)
  (h1 : total_cost = 540)
  (h2 : cost_book1 = 315)
  (h3 : gain_percentage = 19)
  (h4 : ∃ (selling_price : ℝ),
    selling_price = cost_book1 * (1 - loss_percentage / 100) ∧
    selling_price = (total_cost - cost_book1) * (1 + gain_percentage / 100)) :
  ∃ (loss_percentage : ℝ), loss_percentage = 15 := by
sorry


end NUMINAMATH_CALUDE_book_loss_percentage_l1924_192417


namespace NUMINAMATH_CALUDE_wily_person_exists_l1924_192495

inductive PersonType
  | Knight
  | Liar
  | Wily

structure Person where
  type : PersonType
  statement : Prop

def is_truthful (p : Person) : Prop :=
  match p.type with
  | PersonType.Knight => p.statement
  | PersonType.Liar => ¬p.statement
  | PersonType.Wily => True

theorem wily_person_exists (people : Fin 3 → Person)
  (h1 : (people 0).statement = ∃ i, (people i).type = PersonType.Liar)
  (h2 : (people 1).statement = ∀ i j, i ≠ j → ((people i).type = PersonType.Liar ∨ (people j).type = PersonType.Liar))
  (h3 : (people 2).statement = ∀ i, (people i).type = PersonType.Liar)
  : ∃ i, (people i).type = PersonType.Wily :=
by
  sorry

end NUMINAMATH_CALUDE_wily_person_exists_l1924_192495


namespace NUMINAMATH_CALUDE_irregular_shape_impossible_l1924_192446

/-- Represents a shape formed by two equilateral triangles --/
structure TwoTriangleShape where
  -- Add necessary fields to describe the shape

/-- Predicate to check if a shape is regular (has symmetry or regularity) --/
def is_regular (s : TwoTriangleShape) : Prop :=
  sorry  -- Definition of regularity

/-- Predicate to check if a shape can be formed by two equilateral triangles --/
def can_be_formed_by_triangles (s : TwoTriangleShape) : Prop :=
  sorry  -- Definition based on triangle placement rules

theorem irregular_shape_impossible (s : TwoTriangleShape) :
  ¬(is_regular s) → ¬(can_be_formed_by_triangles s) :=
  sorry  -- The proof would go here

end NUMINAMATH_CALUDE_irregular_shape_impossible_l1924_192446


namespace NUMINAMATH_CALUDE_min_a_value_l1924_192412

theorem min_a_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 9 * x + y = x * y) :
  ∃ (a : ℝ), a > 0 ∧ (∀ (x y : ℝ), x > 0 → y > 0 → a * x + y ≥ 25) ∧
  (∀ (b : ℝ), b > 0 → (∀ (x y : ℝ), x > 0 → y > 0 → b * x + y ≥ 25) → b ≥ a) ∧
  a = 4 :=
sorry

end NUMINAMATH_CALUDE_min_a_value_l1924_192412


namespace NUMINAMATH_CALUDE_distance_ratios_sum_to_one_l1924_192419

theorem distance_ratios_sum_to_one (x y z : ℝ) :
  let r := Real.sqrt (x^2 + y^2 + z^2)
  let c := x / r
  let s := y / r
  let z_r := z / r
  s^2 - c^2 + z_r^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_distance_ratios_sum_to_one_l1924_192419


namespace NUMINAMATH_CALUDE_no_y_intercepts_l1924_192448

theorem no_y_intercepts (y : ℝ) : ¬ ∃ y, 3 * y^2 - y + 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_y_intercepts_l1924_192448


namespace NUMINAMATH_CALUDE_at_most_one_obtuse_l1924_192464

-- Define a triangle
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_180 : angle1 + angle2 + angle3 = 180
  all_positive : 0 < angle1 ∧ 0 < angle2 ∧ 0 < angle3

-- Define an obtuse angle
def is_obtuse (angle : ℝ) : Prop := 90 < angle

-- Theorem statement
theorem at_most_one_obtuse (t : Triangle) : 
  ¬(is_obtuse t.angle1 ∧ is_obtuse t.angle2) ∧ 
  ¬(is_obtuse t.angle1 ∧ is_obtuse t.angle3) ∧ 
  ¬(is_obtuse t.angle2 ∧ is_obtuse t.angle3) :=
sorry

end NUMINAMATH_CALUDE_at_most_one_obtuse_l1924_192464


namespace NUMINAMATH_CALUDE_two_truth_tellers_l1924_192473

/-- Represents the four Knaves -/
inductive Knave : Type
  | Hearts
  | Clubs
  | Diamonds
  | Spades

/-- Represents whether a Knave is telling the truth or lying -/
def Truthfulness : Type := Knave → Bool

/-- A consistent arrangement of truthfulness satisfies the interdependence of Knaves' statements -/
def is_consistent (t : Truthfulness) : Prop :=
  t Knave.Hearts = (t Knave.Clubs = false ∧ t Knave.Diamonds = true ∧ t Knave.Spades = false)

/-- Counts the number of truth-telling Knaves -/
def count_truth_tellers (t : Truthfulness) : Nat :=
  (if t Knave.Hearts then 1 else 0) +
  (if t Knave.Clubs then 1 else 0) +
  (if t Knave.Diamonds then 1 else 0) +
  (if t Knave.Spades then 1 else 0)

/-- Main theorem: Any consistent arrangement has exactly two truth-tellers -/
theorem two_truth_tellers (t : Truthfulness) (h : is_consistent t) :
  count_truth_tellers t = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_truth_tellers_l1924_192473


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l1924_192418

theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : 4 * 2 = k) :
  x * (-5) = k → x = -8/5 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l1924_192418


namespace NUMINAMATH_CALUDE_min_value_product_l1924_192447

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 8) :
  (x + 3 * y) * (y + 3 * z) * (3 * x * z + 1) ≥ 72 := by
  sorry

end NUMINAMATH_CALUDE_min_value_product_l1924_192447


namespace NUMINAMATH_CALUDE_charity_donation_proof_l1924_192420

/-- Calculates the donation amount for a charity draw ticket given initial amount,
    winnings, purchases, and final amount. -/
def calculate_donation (initial_amount : ℤ) (prize : ℤ) (lottery_win : ℤ) 
                       (water_cost : ℤ) (lottery_cost : ℤ) (final_amount : ℤ) : ℤ :=
  initial_amount + prize + lottery_win - water_cost - lottery_cost - final_amount

/-- Proves that the donation for the charity draw ticket was $4 given the problem conditions. -/
theorem charity_donation_proof (initial_amount : ℤ) (prize : ℤ) (lottery_win : ℤ) 
                               (water_cost : ℤ) (lottery_cost : ℤ) (final_amount : ℤ) 
                               (h1 : initial_amount = 10)
                               (h2 : prize = 90)
                               (h3 : lottery_win = 65)
                               (h4 : water_cost = 1)
                               (h5 : lottery_cost = 1)
                               (h6 : final_amount = 94) :
  calculate_donation initial_amount prize lottery_win water_cost lottery_cost final_amount = 4 :=
by sorry

#eval calculate_donation 10 90 65 1 1 94

end NUMINAMATH_CALUDE_charity_donation_proof_l1924_192420


namespace NUMINAMATH_CALUDE_goose_egg_calculation_l1924_192410

theorem goose_egg_calculation (total_survived : ℕ) 
  (hatch_rate : ℚ) (first_month_survival : ℚ) 
  (first_year_death : ℚ) (migration_rate : ℚ) 
  (predator_survival : ℚ) :
  hatch_rate = 1/3 →
  first_month_survival = 4/5 →
  first_year_death = 3/5 →
  migration_rate = 1/4 →
  predator_survival = 2/3 →
  total_survived = 140 →
  ∃ (total_eggs : ℕ), 
    total_eggs = 1050 ∧
    (total_eggs : ℚ) * hatch_rate * first_month_survival * (1 - first_year_death) * 
    (1 - migration_rate) * predator_survival = total_survived := by
  sorry

#eval 1050

end NUMINAMATH_CALUDE_goose_egg_calculation_l1924_192410


namespace NUMINAMATH_CALUDE_total_pencils_l1924_192436

/-- The number of pencils Jessica, Sandy, and Jason have in total is 24, 
    given that each of them has 8 pencils. -/
theorem total_pencils (jessica_pencils sandy_pencils jason_pencils : ℕ) 
  (h1 : jessica_pencils = 8)
  (h2 : sandy_pencils = 8)
  (h3 : jason_pencils = 8) :
  jessica_pencils + sandy_pencils + jason_pencils = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l1924_192436


namespace NUMINAMATH_CALUDE_student_gathering_problem_l1924_192479

theorem student_gathering_problem (male_count : ℕ) (female_count : ℕ) : 
  female_count = male_count + 6 →
  (female_count : ℚ) / (male_count + female_count) = 2 / 3 →
  male_count + female_count = 18 :=
by sorry

end NUMINAMATH_CALUDE_student_gathering_problem_l1924_192479


namespace NUMINAMATH_CALUDE_slips_with_three_count_l1924_192455

/-- Given a bag of slips with either 3 or 8 written on them, 
    this function calculates the expected value of a randomly drawn slip. -/
def expected_value (total_slips : ℕ) (slips_with_three : ℕ) : ℚ :=
  (3 * slips_with_three + 8 * (total_slips - slips_with_three)) / total_slips

/-- Theorem stating that given the conditions of the problem, 
    the number of slips with 3 written on them is 8. -/
theorem slips_with_three_count : 
  ∃ (x : ℕ), x ≤ 15 ∧ expected_value 15 x = 5.4 ∧ x = 8 := by
  sorry


end NUMINAMATH_CALUDE_slips_with_three_count_l1924_192455


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l1924_192421

theorem least_common_multiple_first_ten : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k ∣ n) ∧ 
  (∀ m : ℕ, m < n → ∃ k : ℕ, k ≤ 10 ∧ ¬(k ∣ m)) ∧
  n = 2520 := by
sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l1924_192421


namespace NUMINAMATH_CALUDE_quadratic_function_comparison_l1924_192460

/-- Proves that for points A(x₁, y₁) and B(x₂, y₂) on the graph of y = (x - 1)² + 1, 
    if x₁ > x₂ > 1, then y₁ > y₂. -/
theorem quadratic_function_comparison (x₁ x₂ y₁ y₂ : ℝ) 
    (h1 : y₁ = (x₁ - 1)^2 + 1)
    (h2 : y₂ = (x₂ - 1)^2 + 1)
    (h3 : x₁ > x₂)
    (h4 : x₂ > 1) : 
  y₁ > y₂ := by
  sorry


end NUMINAMATH_CALUDE_quadratic_function_comparison_l1924_192460


namespace NUMINAMATH_CALUDE_rock_collection_inconsistency_l1924_192416

theorem rock_collection_inconsistency (J : ℤ) : ¬ (∃ (jose albert : ℤ),
  jose = J - 14 ∧
  albert = jose + 20 ∧
  albert = J + 6) := by
  sorry

end NUMINAMATH_CALUDE_rock_collection_inconsistency_l1924_192416


namespace NUMINAMATH_CALUDE_ball_hit_ground_time_l1924_192485

/-- The time when a ball hits the ground given its height equation -/
theorem ball_hit_ground_time (t : ℝ) : 
  let y : ℝ → ℝ := λ t => -4.9 * t^2 + 4 * t + 6
  y t = 0 → t = 78 / 49 := by
sorry

end NUMINAMATH_CALUDE_ball_hit_ground_time_l1924_192485


namespace NUMINAMATH_CALUDE_max_value_equation_l1924_192425

theorem max_value_equation (p : ℕ) (x y : ℕ) (h_prime : Nat.Prime p) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_eq : 9 * x * y = p * (p + 3 * x + 6 * y)) :
  p^2 + x^2 + y^2 ≤ 29 ∧ ∃ (p' x' y' : ℕ), 
    Nat.Prime p' ∧ x' > 0 ∧ y' > 0 ∧ 
    9 * x' * y' = p' * (p' + 3 * x' + 6 * y') ∧
    p'^2 + x'^2 + y'^2 = 29 :=
by sorry


end NUMINAMATH_CALUDE_max_value_equation_l1924_192425


namespace NUMINAMATH_CALUDE_greater_number_proof_l1924_192471

theorem greater_number_proof (x y : ℝ) (sum_eq : x + y = 36) (diff_eq : x - y = 12) : 
  max x y = 24 := by
sorry

end NUMINAMATH_CALUDE_greater_number_proof_l1924_192471


namespace NUMINAMATH_CALUDE_student_fail_marks_l1924_192496

theorem student_fail_marks (total_marks passing_percentage student_marks : ℕ) 
  (h1 : total_marks = 700)
  (h2 : passing_percentage = 33)
  (h3 : student_marks = 175) :
  (total_marks * passing_percentage / 100 : ℕ) - student_marks = 56 :=
by sorry

end NUMINAMATH_CALUDE_student_fail_marks_l1924_192496


namespace NUMINAMATH_CALUDE_quadratic_polynomial_root_l1924_192428

theorem quadratic_polynomial_root (x : ℂ) : 
  let p : ℂ → ℂ := λ z => 3 * z^2 - 24 * z + 51
  (p (4 + I) = 0) ∧ (∀ z : ℂ, p z = 3 * z^2 + ((-24) * z + 51)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_root_l1924_192428


namespace NUMINAMATH_CALUDE_not_multiple_of_three_l1924_192468

theorem not_multiple_of_three (n : ℕ) (h : ∃ m : ℕ, n * (n + 3) = m ^ 2) : ¬ (3 ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_not_multiple_of_three_l1924_192468


namespace NUMINAMATH_CALUDE_binomial_inequality_l1924_192489

theorem binomial_inequality (n : ℤ) (x : ℝ) (h : x > 0) : (1 + x)^n ≥ 1 + n * x := by
  sorry

end NUMINAMATH_CALUDE_binomial_inequality_l1924_192489


namespace NUMINAMATH_CALUDE_max_value_cos_sin_l1924_192444

theorem max_value_cos_sin (x : ℝ) : 3 * Real.cos x + 4 * Real.sin x ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_l1924_192444


namespace NUMINAMATH_CALUDE_value_of_a_l1924_192487

-- Define set A
def A : Set ℝ := {x | x^2 ≠ 1}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | x = a}

-- Theorem statement
theorem value_of_a (a : ℝ) (h : B a ⊆ A) : a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l1924_192487


namespace NUMINAMATH_CALUDE_bathtub_jello_cost_l1924_192456

/-- The cost to fill a bathtub with jello given specific ratios and measurements -/
theorem bathtub_jello_cost :
  let jello_mix_per_pound : ℚ := 3/2  -- 1.5 tablespoons per pound
  let bathtub_volume : ℚ := 6         -- 6 cubic feet
  let gallons_per_cubic_foot : ℚ := 15/2  -- 7.5 gallons per cubic foot
  let pounds_per_gallon : ℚ := 8      -- 8 pounds per gallon
  let cost_per_tablespoon : ℚ := 1/2  -- $0.50 per tablespoon
  
  let total_gallons : ℚ := bathtub_volume * gallons_per_cubic_foot
  let total_pounds : ℚ := total_gallons * pounds_per_gallon
  let total_tablespoons : ℚ := total_pounds * jello_mix_per_pound
  let total_cost : ℚ := total_tablespoons * cost_per_tablespoon

  total_cost = 270 := by
    sorry


end NUMINAMATH_CALUDE_bathtub_jello_cost_l1924_192456


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l1924_192493

/-- Expresses 0.3̄56 as a rational number -/
theorem repeating_decimal_to_fraction : 
  ∃ (n d : ℕ), d ≠ 0 ∧ (0.3 + (56 : ℚ) / 99 / 10) = (n : ℚ) / d := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l1924_192493


namespace NUMINAMATH_CALUDE_greatest_two_digit_with_product_12_and_smallest_sum_l1924_192459

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

theorem greatest_two_digit_with_product_12_and_smallest_sum :
  ∃ (n : ℕ), is_two_digit n ∧ 
             digit_product n = 12 ∧
             (∀ m : ℕ, is_two_digit m → digit_product m = 12 → digit_sum m ≥ digit_sum n) ∧
             (∀ k : ℕ, is_two_digit k → digit_product k = 12 → digit_sum k = digit_sum n → k ≤ n) ∧
             n = 43 :=
by sorry

end NUMINAMATH_CALUDE_greatest_two_digit_with_product_12_and_smallest_sum_l1924_192459


namespace NUMINAMATH_CALUDE_tank_length_calculation_l1924_192401

/-- Given a rectangular field and a tank dug within it, this theorem proves
    the length of the tank when the excavated earth raises the field level. -/
theorem tank_length_calculation (field_length field_width tank_width tank_depth level_rise : ℝ)
  (h1 : field_length = 90)
  (h2 : field_width = 50)
  (h3 : tank_width = 20)
  (h4 : tank_depth = 4)
  (h5 : level_rise = 0.5)
  (h6 : tank_width < field_width)
  (h7 : ∀ tank_length, tank_length > 0 → tank_length < field_length) :
  ∃ tank_length : ℝ,
    tank_length > 0 ∧
    tank_length < field_length ∧
    tank_length * tank_width * tank_depth =
      (field_length * field_width - tank_length * tank_width) * level_rise ∧
    tank_length = 25 := by
  sorry


end NUMINAMATH_CALUDE_tank_length_calculation_l1924_192401


namespace NUMINAMATH_CALUDE_division_multiplication_problem_l1924_192492

theorem division_multiplication_problem : (150 : ℚ) / ((30 : ℚ) / 3) * 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_problem_l1924_192492


namespace NUMINAMATH_CALUDE_sum_of_fractions_less_than_target_l1924_192478

theorem sum_of_fractions_less_than_target : 
  (1/2 : ℚ) + (-5/6 : ℚ) + (1/5 : ℚ) + (1/4 : ℚ) + (-9/20 : ℚ) + (-9/20 : ℚ) < (45/100 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_less_than_target_l1924_192478


namespace NUMINAMATH_CALUDE_size_relationship_l1924_192445

theorem size_relationship (a b c : ℝ) 
  (ha : a = Real.sqrt 3)
  (hb : b = Real.sqrt 15 - Real.sqrt 7)
  (hc : c = Real.sqrt 11 - Real.sqrt 3) :
  a > c ∧ c > b := by
  sorry

end NUMINAMATH_CALUDE_size_relationship_l1924_192445


namespace NUMINAMATH_CALUDE_sentences_at_start_l1924_192437

-- Define the typing rate
def typing_rate : ℕ := 6

-- Define the typing durations
def first_session : ℕ := 20
def second_session : ℕ := 15
def third_session : ℕ := 18

-- Define the number of erased sentences
def erased_sentences : ℕ := 40

-- Define the total number of sentences at the end of the day
def total_sentences : ℕ := 536

-- Theorem to prove
theorem sentences_at_start : 
  total_sentences - (typing_rate * (first_session + second_session + third_session) - erased_sentences) = 258 :=
by sorry

end NUMINAMATH_CALUDE_sentences_at_start_l1924_192437


namespace NUMINAMATH_CALUDE_mixed_doubles_handshakes_l1924_192469

/-- Represents a mixed doubles tennis tournament -/
structure MixedDoublesTournament where
  teams : Nat
  players_per_team : Nat
  opposite_gender_players : Nat

/-- Calculates the number of handshakes in a mixed doubles tournament -/
def handshakes (tournament : MixedDoublesTournament) : Nat :=
  tournament.teams * (tournament.opposite_gender_players - 1)

/-- Theorem: In a mixed doubles tennis tournament with 4 teams, 
    where each player shakes hands once with every player of the 
    opposite gender except their own partner, the total number 
    of handshakes is 12. -/
theorem mixed_doubles_handshakes :
  let tournament : MixedDoublesTournament := {
    teams := 4,
    players_per_team := 2,
    opposite_gender_players := 4
  }
  handshakes tournament = 12 := by
  sorry

end NUMINAMATH_CALUDE_mixed_doubles_handshakes_l1924_192469


namespace NUMINAMATH_CALUDE_p_plus_q_equals_twenty_one_halves_l1924_192461

theorem p_plus_q_equals_twenty_one_halves 
  (p q : ℝ) 
  (hp : p^3 - 21*p^2 + 35*p - 105 = 0) 
  (hq : 5*q^3 - 35*q^2 - 175*q + 1225 = 0) : 
  p + q = 21/2 := by sorry

end NUMINAMATH_CALUDE_p_plus_q_equals_twenty_one_halves_l1924_192461


namespace NUMINAMATH_CALUDE_new_person_weight_l1924_192406

theorem new_person_weight (W : ℝ) (new_weight : ℝ) :
  (W + new_weight - 25) / 12 = W / 12 + 3 →
  new_weight = 61 :=
by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l1924_192406


namespace NUMINAMATH_CALUDE_alice_favorite_number_l1924_192491

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def satisfies_conditions (n : ℕ) : Prop :=
  70 < n ∧ n < 150 ∧
  n % 13 = 0 ∧
  ¬(n % 3 = 0) ∧
  is_prime (digit_sum n)

theorem alice_favorite_number :
  ∀ n : ℕ, satisfies_conditions n ↔ n = 104 :=
sorry

end NUMINAMATH_CALUDE_alice_favorite_number_l1924_192491


namespace NUMINAMATH_CALUDE_algebraic_expression_symmetry_l1924_192435

/-- Given an algebraic expression ax^5 + bx^3 + cx - 8, if its value is 6 when x = 5,
    then its value is -22 when x = -5 -/
theorem algebraic_expression_symmetry (a b c : ℝ) :
  (5^5 * a + 5^3 * b + 5 * c - 8 = 6) →
  ((-5)^5 * a + (-5)^3 * b + (-5) * c - 8 = -22) :=
by sorry

end NUMINAMATH_CALUDE_algebraic_expression_symmetry_l1924_192435


namespace NUMINAMATH_CALUDE_marion_ella_score_ratio_l1924_192422

/-- Prove that the ratio of Marion's score to Ella's score is 2:3 -/
theorem marion_ella_score_ratio :
  let total_items : ℕ := 40
  let ella_incorrect : ℕ := 4
  let marion_score : ℕ := 24
  let ella_score : ℕ := total_items - ella_incorrect
  (marion_score : ℚ) / ella_score = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_marion_ella_score_ratio_l1924_192422


namespace NUMINAMATH_CALUDE_fifteenth_digit_is_one_l1924_192443

/-- The decimal representation of 1/9 as a sequence of digits after the decimal point -/
def decimal_1_9 : ℕ → ℕ
  | n => 1

/-- The decimal representation of 1/11 as a sequence of digits after the decimal point -/
def decimal_1_11 : ℕ → ℕ
  | n => if n % 3 = 0 then 0 else 9

/-- The sum of the decimal representations of 1/9 and 1/11 as a sequence of digits after the decimal point -/
def sum_decimals : ℕ → ℕ
  | n => (decimal_1_9 n + decimal_1_11 n) % 10

theorem fifteenth_digit_is_one :
  sum_decimals 14 = 1 := by sorry

end NUMINAMATH_CALUDE_fifteenth_digit_is_one_l1924_192443


namespace NUMINAMATH_CALUDE_hexagonal_pyramid_base_edge_length_l1924_192480

/-- A pyramid with a regular hexagon base -/
structure HexagonalPyramid where
  base_edge_length : ℝ
  side_edge_length : ℝ
  total_edge_length : ℝ

/-- The property that the pyramid satisfies the given conditions -/
def satisfies_conditions (p : HexagonalPyramid) : Prop :=
  p.side_edge_length = 8 ∧ p.total_edge_length = 120

/-- The theorem stating that if a hexagonal pyramid satisfies the conditions, 
    its base edge length is 12 -/
theorem hexagonal_pyramid_base_edge_length 
  (p : HexagonalPyramid) (h : satisfies_conditions p) : 
  p.base_edge_length = 12 := by
  sorry

end NUMINAMATH_CALUDE_hexagonal_pyramid_base_edge_length_l1924_192480


namespace NUMINAMATH_CALUDE_four_fixed_points_iff_c_in_range_l1924_192449

/-- A quadratic function f(x) = x^2 - cx + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - c*x + c

/-- The composition of f with itself -/
def f_comp_f (c : ℝ) (x : ℝ) : ℝ := f c (f c x)

/-- Predicate for f ∘ f having four distinct fixed points -/
def has_four_distinct_fixed_points (c : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ x₄ : ℝ, 
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    f_comp_f c x₁ = x₁ ∧ f_comp_f c x₂ = x₂ ∧ f_comp_f c x₃ = x₃ ∧ f_comp_f c x₄ = x₄

theorem four_fixed_points_iff_c_in_range :
  ∀ c : ℝ, has_four_distinct_fixed_points c ↔ (c < -1 ∨ c > 3) :=
sorry

end NUMINAMATH_CALUDE_four_fixed_points_iff_c_in_range_l1924_192449


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_with_reversed_composite_l1924_192457

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ d ∣ n

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

theorem smallest_two_digit_prime_with_reversed_composite :
  ∃ n : ℕ,
    is_two_digit n ∧
    is_prime n ∧
    (n / 10 ≥ 3) ∧
    is_composite (reverse_digits n) ∧
    (∀ m : ℕ, is_two_digit m → is_prime m → (m / 10 ≥ 3) → is_composite (reverse_digits m) → n ≤ m) ∧
    n = 41 :=
  sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_with_reversed_composite_l1924_192457


namespace NUMINAMATH_CALUDE_factorization_proof_l1924_192442

theorem factorization_proof (x : ℝ) : 18 * x^3 + 12 * x^2 = 6 * x^2 * (3 * x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1924_192442


namespace NUMINAMATH_CALUDE_sum_of_squares_geq_product_l1924_192466

theorem sum_of_squares_geq_product (x₁ x₂ x₃ x₄ x₅ : ℝ) :
  x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₅^2 ≥ x₁ * (x₂ + x₃ + x₄ + x₅) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_geq_product_l1924_192466


namespace NUMINAMATH_CALUDE_running_yardage_l1924_192458

/-- The star running back's total yardage -/
def total_yardage : ℕ := 150

/-- The star running back's passing yardage -/
def passing_yardage : ℕ := 60

/-- Theorem: The star running back's running yardage is 90 yards -/
theorem running_yardage : total_yardage - passing_yardage = 90 := by
  sorry

end NUMINAMATH_CALUDE_running_yardage_l1924_192458


namespace NUMINAMATH_CALUDE_complex_roots_theorem_l1924_192453

theorem complex_roots_theorem (p q r : ℂ) 
  (sum_eq : p + q + r = -1)
  (sum_prod_eq : p*q + p*r + q*r = -1)
  (prod_eq : p*q*r = -1) :
  (p = -1 ∧ q = 1 ∧ r = 1) ∨
  (p = -1 ∧ q = 1 ∧ r = 1) ∨
  (p = 1 ∧ q = -1 ∧ r = 1) ∨
  (p = 1 ∧ q = 1 ∧ r = -1) ∨
  (p = 1 ∧ q = -1 ∧ r = 1) ∨
  (p = -1 ∧ q = 1 ∧ r = 1) := by
  sorry

end NUMINAMATH_CALUDE_complex_roots_theorem_l1924_192453


namespace NUMINAMATH_CALUDE_rectangular_to_cylindrical_l1924_192467

theorem rectangular_to_cylindrical :
  let x : ℝ := 3
  let y : ℝ := -3 * Real.sqrt 3
  let z : ℝ := 2
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := 5 * π / 3
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧
  r = 6 ∧
  θ = 5 * π / 3 ∧
  z = 2 ∧
  x = r * Real.cos θ ∧
  y = r * Real.sin θ := by
sorry

end NUMINAMATH_CALUDE_rectangular_to_cylindrical_l1924_192467


namespace NUMINAMATH_CALUDE_card_selection_count_l1924_192408

theorem card_selection_count (n : ℕ) (h : n > 0) : 
  (Nat.choose (2 * n) n : ℚ) = (Nat.factorial (2 * n)) / ((Nat.factorial n) * (Nat.factorial n)) :=
by sorry

end NUMINAMATH_CALUDE_card_selection_count_l1924_192408


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_equals_two_l1924_192438

/-- Two vectors a and b in 2D space -/
def a : Fin 2 → ℝ := ![(-2 : ℝ), 3]
def b : ℝ → Fin 2 → ℝ := λ m => ![3, m]

/-- The dot product of two 2D vectors -/
def dot_product (v w : Fin 2 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1)

/-- Theorem: If vectors a and b are perpendicular, then m = 2 -/
theorem perpendicular_vectors_m_equals_two :
  ∀ m : ℝ, dot_product a (b m) = 0 → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_equals_two_l1924_192438


namespace NUMINAMATH_CALUDE_min_value_tan_sum_l1924_192476

/-- For any acute-angled triangle ABC, the expression 
    3 tan B tan C + 2 tan A tan C + tan A tan B 
    is always greater than or equal to 6 + 2√3 + 2√2 + 2√6 -/
theorem min_value_tan_sum (A B C : ℝ) (h_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2) 
    (h_triangle : A + B + C = π) : 
  3 * Real.tan B * Real.tan C + 2 * Real.tan A * Real.tan C + Real.tan A * Real.tan B 
    ≥ 6 + 2 * Real.sqrt 3 + 2 * Real.sqrt 2 + 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_tan_sum_l1924_192476


namespace NUMINAMATH_CALUDE_probability_at_least_one_male_l1924_192404

/-- The probability of selecting at least one male out of 3 contestants from a group of 8 finalists (5 females and 3 males) is 23/28. -/
theorem probability_at_least_one_male (total : ℕ) (females : ℕ) (males : ℕ) (selected : ℕ) :
  total = 8 →
  females = 5 →
  males = 3 →
  selected = 3 →
  (Nat.choose total selected - Nat.choose females selected : ℚ) / Nat.choose total selected = 23 / 28 := by
  sorry

#eval (Nat.choose 8 3 - Nat.choose 5 3 : ℚ) / Nat.choose 8 3 == 23 / 28

end NUMINAMATH_CALUDE_probability_at_least_one_male_l1924_192404


namespace NUMINAMATH_CALUDE_smallest_prime_pair_l1924_192475

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem smallest_prime_pair : 
  ∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ q = 13 * p + 2 ∧ 
  (∀ (p' : ℕ), is_prime p' ∧ p' < p → ¬(is_prime (13 * p' + 2))) ∧
  p = 3 ∧ q = 41 := by
sorry

end NUMINAMATH_CALUDE_smallest_prime_pair_l1924_192475


namespace NUMINAMATH_CALUDE_volume_of_cube_with_triple_surface_area_l1924_192439

def cube_volume (s : ℝ) : ℝ := s^3
def cube_surface_area (s : ℝ) : ℝ := 6 * s^2

theorem volume_of_cube_with_triple_surface_area (cube_a_side : ℝ) (cube_b_side : ℝ) :
  cube_volume cube_a_side = 8 →
  cube_surface_area cube_b_side = 3 * cube_surface_area cube_a_side →
  cube_volume cube_b_side = 24 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_cube_with_triple_surface_area_l1924_192439


namespace NUMINAMATH_CALUDE_even_function_iff_b_zero_l1924_192477

/-- For real numbers a and b, and function f(x) = a*cos(x) + b*sin(x),
    f(x) is an even function if and only if b = 0 -/
theorem even_function_iff_b_zero (a b : ℝ) :
  (∀ x, a * Real.cos x + b * Real.sin x = a * Real.cos (-x) + b * Real.sin (-x)) ↔ b = 0 :=
by sorry

end NUMINAMATH_CALUDE_even_function_iff_b_zero_l1924_192477


namespace NUMINAMATH_CALUDE_factorization_equality_l1924_192407

theorem factorization_equality (x : ℝ) :
  (x - 1)^4 + x * (2*x + 1) * (2*x - 1) + 5*x = (x^2 + 3 + 2*Real.sqrt 2) * (x^2 + 3 - 2*Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1924_192407


namespace NUMINAMATH_CALUDE_chocolate_eggs_problem_l1924_192427

theorem chocolate_eggs_problem (egg_weight : ℕ) (num_boxes : ℕ) (remaining_weight : ℕ) : 
  egg_weight = 10 →
  num_boxes = 4 →
  remaining_weight = 90 →
  ∃ (total_eggs : ℕ), 
    total_eggs = num_boxes * (remaining_weight / (egg_weight * (num_boxes - 1))) ∧
    total_eggs = 12 := by
sorry

end NUMINAMATH_CALUDE_chocolate_eggs_problem_l1924_192427


namespace NUMINAMATH_CALUDE_bruce_fruit_purchase_cost_l1924_192432

/-- Calculates the total cost of Bruce's fruit purchase in US dollars -/
def fruit_purchase_cost (grapes_kg : ℝ) (grapes_price : ℝ) (mangoes_kg : ℝ) (mangoes_price : ℝ)
  (oranges_kg : ℝ) (oranges_price : ℝ) (apples_kg : ℝ) (apples_price : ℝ)
  (grapes_discount : ℝ) (mangoes_tax : ℝ) (oranges_premium : ℝ)
  (euro_to_usd : ℝ) (pound_to_usd : ℝ) (yen_to_usd : ℝ) : ℝ :=
  let grapes_cost := grapes_kg * grapes_price * (1 - grapes_discount)
  let mangoes_cost := mangoes_kg * mangoes_price * euro_to_usd * (1 + mangoes_tax)
  let oranges_cost := oranges_kg * oranges_price * pound_to_usd * (1 + oranges_premium)
  let apples_cost := apples_kg * apples_price * yen_to_usd
  grapes_cost + mangoes_cost + oranges_cost + apples_cost

/-- Theorem stating that Bruce's fruit purchase cost is $1563.10 -/
theorem bruce_fruit_purchase_cost :
  fruit_purchase_cost 8 70 8 55 5 40 10 3000 0.1 0.05 0.03 1.15 1.25 0.009 = 1563.10 := by
  sorry

end NUMINAMATH_CALUDE_bruce_fruit_purchase_cost_l1924_192432


namespace NUMINAMATH_CALUDE_remainder_7325_div_11_l1924_192430

theorem remainder_7325_div_11 : 7325 % 11 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_7325_div_11_l1924_192430


namespace NUMINAMATH_CALUDE_tan_sin_expression_simplification_l1924_192463

theorem tan_sin_expression_simplification :
  Real.tan (70 * π / 180) * Real.sin (80 * π / 180) * (Real.sqrt 3 * Real.tan (20 * π / 180) - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sin_expression_simplification_l1924_192463


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l1924_192434

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x - a| + |2*x - 1|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f x (-1) ≤ 2} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 1/2} :=
sorry

-- Part 2
theorem range_of_a_part2 :
  (∀ x ∈ Set.Icc (1/2) 1, f x a ≤ |2*x + 1|) →
  (0 ≤ a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l1924_192434


namespace NUMINAMATH_CALUDE_customers_left_l1924_192429

theorem customers_left (initial : ℕ) (new : ℕ) (final : ℕ) : 
  initial = 33 → new = 26 → final = 28 → initial - (initial - new + final) = 31 := by
sorry

end NUMINAMATH_CALUDE_customers_left_l1924_192429
