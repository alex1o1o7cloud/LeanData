import Mathlib

namespace scientific_notation_5690_l1006_100627

theorem scientific_notation_5690 : 
  5690 = 5.69 * (10 : ℝ)^3 := by sorry

end scientific_notation_5690_l1006_100627


namespace fruit_eating_arrangements_l1006_100685

def num_apples : ℕ := 4
def num_oranges : ℕ := 2
def num_bananas : ℕ := 2

def total_fruits : ℕ := num_apples + num_oranges + num_bananas

theorem fruit_eating_arrangements :
  (total_fruits.factorial) / (num_oranges.factorial * num_bananas.factorial) = 6 :=
by sorry

end fruit_eating_arrangements_l1006_100685


namespace orange_juice_mixture_l1006_100687

theorem orange_juice_mixture (pitcher_capacity : ℚ) 
  (first_pitcher_fraction : ℚ) (second_pitcher_fraction : ℚ) : 
  pitcher_capacity > 0 →
  first_pitcher_fraction = 1/4 →
  second_pitcher_fraction = 3/7 →
  (first_pitcher_fraction * pitcher_capacity + 
   second_pitcher_fraction * pitcher_capacity) / 
  (2 * pitcher_capacity) = 95/280 := by
sorry

end orange_juice_mixture_l1006_100687


namespace prob_calculations_l1006_100636

/-- Represents a box containing balls of two colors -/
structure Box where
  white : ℕ
  red : ℕ

/-- Calculates the probability of drawing two red balls without replacement -/
def prob_two_red (b : Box) : ℚ :=
  (b.red * (b.red - 1)) / ((b.white + b.red) * (b.white + b.red - 1))

/-- Calculates the probability of drawing a red ball after transferring two balls -/
def prob_red_after_transfer (b1 b2 : Box) : ℚ :=
  let total_ways := (b1.white + b1.red) * (b1.white + b1.red - 1) / 2
  let p_two_red := (b1.red * (b1.red - 1) / 2) / total_ways
  let p_one_each := (b1.red * b1.white) / total_ways
  let p_two_white := (b1.white * (b1.white - 1) / 2) / total_ways
  
  let p_red_given_two_red := (b2.red + 2) / (b2.white + b2.red + 2)
  let p_red_given_one_each := (b2.red + 1) / (b2.white + b2.red + 2)
  let p_red_given_two_white := b2.red / (b2.white + b2.red + 2)
  
  p_two_red * p_red_given_two_red + p_one_each * p_red_given_one_each + p_two_white * p_red_given_two_white

theorem prob_calculations (b1 b2 : Box) 
  (h1 : b1.white = 2) (h2 : b1.red = 4) (h3 : b2.white = 5) (h4 : b2.red = 3) :
  prob_two_red b1 = 2/5 ∧ prob_red_after_transfer b1 b2 = 13/30 := by
  sorry

end prob_calculations_l1006_100636


namespace identity_function_satisfies_equation_l1006_100693

theorem identity_function_satisfies_equation (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x - f (x - y)) + x = f (x + y)) →
  (∀ x : ℝ, f x = x) :=
by sorry

end identity_function_satisfies_equation_l1006_100693


namespace fish_count_l1006_100615

theorem fish_count (total_tables : ℕ) (special_table_fish : ℕ) (regular_table_fish : ℕ)
  (h1 : total_tables = 32)
  (h2 : special_table_fish = 3)
  (h3 : regular_table_fish = 2) :
  (total_tables - 1) * regular_table_fish + special_table_fish = 65 := by
  sorry

end fish_count_l1006_100615


namespace min_value_X_l1006_100623

/-- Represents a digit from 1 to 9 -/
def Digit := Fin 9

/-- Converts a four-digit number to its integer representation -/
def fourDigitToInt (a b c d : Digit) : ℕ :=
  1000 * (a.val + 1) + 100 * (b.val + 1) + 10 * (c.val + 1) + (d.val + 1)

/-- Converts a two-digit number to its integer representation -/
def twoDigitToInt (e f : Digit) : ℕ :=
  10 * (e.val + 1) + (f.val + 1)

theorem min_value_X (a b c d e f g h i : Digit) 
  (h1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i)
  (h2 : b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i)
  (h3 : c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i)
  (h4 : d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i)
  (h5 : e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i)
  (h6 : f ≠ g ∧ f ≠ h ∧ f ≠ i)
  (h7 : g ≠ h ∧ g ≠ i)
  (h8 : h ≠ i) :
  ∃ (x : ℕ), x = fourDigitToInt a b c d + twoDigitToInt e f * twoDigitToInt g h - (i.val + 1) ∧
    x ≥ 2369 ∧
    (∀ (a' b' c' d' e' f' g' h' i' : Digit),
      (a' ≠ b' ∧ a' ≠ c' ∧ a' ≠ d' ∧ a' ≠ e' ∧ a' ≠ f' ∧ a' ≠ g' ∧ a' ≠ h' ∧ a' ≠ i') →
      (b' ≠ c' ∧ b' ≠ d' ∧ b' ≠ e' ∧ b' ≠ f' ∧ b' ≠ g' ∧ b' ≠ h' ∧ b' ≠ i') →
      (c' ≠ d' ∧ c' ≠ e' ∧ c' ≠ f' ∧ c' ≠ g' ∧ c' ≠ h' ∧ c' ≠ i') →
      (d' ≠ e' ∧ d' ≠ f' ∧ d' ≠ g' ∧ d' ≠ h' ∧ d' ≠ i') →
      (e' ≠ f' ∧ e' ≠ g' ∧ e' ≠ h' ∧ e' ≠ i') →
      (f' ≠ g' ∧ f' ≠ h' ∧ f' ≠ i') →
      (g' ≠ h' ∧ g' ≠ i') →
      (h' ≠ i') →
      x ≤ fourDigitToInt a' b' c' d' + twoDigitToInt e' f' * twoDigitToInt g' h' - (i'.val + 1)) :=
by
  sorry

end min_value_X_l1006_100623


namespace final_sum_is_212_l1006_100677

/-- Represents a person in the debt settlement problem -/
inductive Person
| Earl
| Fred
| Greg
| Hannah

/-- Represents the initial amount of money each person has -/
def initial_amount (p : Person) : Int :=
  match p with
  | Person.Earl => 90
  | Person.Fred => 48
  | Person.Greg => 36
  | Person.Hannah => 72

/-- Represents the amount one person owes to another -/
def debt (debtor receiver : Person) : Int :=
  match debtor, receiver with
  | Person.Earl, Person.Fred => 28
  | Person.Earl, Person.Hannah => 30
  | Person.Fred, Person.Greg => 32
  | Person.Fred, Person.Hannah => 10
  | Person.Greg, Person.Earl => 40
  | Person.Greg, Person.Hannah => 20
  | Person.Hannah, Person.Greg => 15
  | Person.Hannah, Person.Earl => 25
  | _, _ => 0

/-- Calculates the final amount a person has after settling all debts -/
def final_amount (p : Person) : Int :=
  initial_amount p
  + (debt Person.Earl p + debt Person.Fred p + debt Person.Greg p + debt Person.Hannah p)
  - (debt p Person.Earl + debt p Person.Fred + debt p Person.Greg + debt p Person.Hannah)

/-- Theorem stating that the sum of Greg's, Earl's, and Hannah's money after settling debts is $212 -/
theorem final_sum_is_212 :
  final_amount Person.Greg + final_amount Person.Earl + final_amount Person.Hannah = 212 :=
by sorry

end final_sum_is_212_l1006_100677


namespace negation_of_proposition_l1006_100663

theorem negation_of_proposition (p : ℝ → Prop) :
  (∀ x : ℝ, x ≥ 2 → p x) ↔ ¬(∃ x : ℝ, x < 2 ∧ ¬(p x)) :=
by sorry

end negation_of_proposition_l1006_100663


namespace vector_dot_product_theorem_l1006_100694

def vector_a (x : ℝ) : Fin 2 → ℝ := ![x, 1]
def vector_b (y : ℝ) : Fin 2 → ℝ := ![1, y]
def vector_c : Fin 2 → ℝ := ![3, -6]

def dot_product (u v : Fin 2 → ℝ) : ℝ := (u 0) * (v 0) + (u 1) * (v 1)

def perpendicular (u v : Fin 2 → ℝ) : Prop := dot_product u v = 0

def parallel (u v : Fin 2 → ℝ) : Prop := ∃ (k : ℝ), ∀ (i : Fin 2), u i = k * (v i)

theorem vector_dot_product_theorem (x y : ℝ) :
  perpendicular (vector_a x) vector_c →
  parallel (vector_b y) vector_c →
  dot_product (vector_a x + vector_b y) vector_c = 15 := by
  sorry

end vector_dot_product_theorem_l1006_100694


namespace symmetrical_line_slope_range_l1006_100652

/-- Given a line l: y = kx - 1 intersecting with x + y - 1 = 0,
    the range of k for which a symmetrical line can be derived is (1, +∞) -/
theorem symmetrical_line_slope_range (k : ℝ) : 
  (∃ (x y : ℝ), y = k * x - 1 ∧ x + y - 1 = 0) →
  (∃ (m : ℝ), m ≠ k ∧ ∃ (x₀ y₀ : ℝ), (∀ (x y : ℝ), y - y₀ = m * (x - x₀) ↔ y = k * x - 1)) ↔
  k > 1 :=
sorry

end symmetrical_line_slope_range_l1006_100652


namespace intersection_M_N_l1006_100686

/-- Set M is defined as the set of all real numbers x where 0 < x < 4 -/
def M : Set ℝ := {x : ℝ | 0 < x ∧ x < 4}

/-- Set N is defined as the set of all real numbers x where 1/3 ≤ x ≤ 5 -/
def N : Set ℝ := {x : ℝ | 1/3 ≤ x ∧ x ≤ 5}

/-- The intersection of sets M and N -/
theorem intersection_M_N : M ∩ N = {x : ℝ | 1/3 ≤ x ∧ x < 4} := by
  sorry

end intersection_M_N_l1006_100686


namespace candy_bar_cost_l1006_100684

/-- The cost of a candy bar given initial and final amounts --/
theorem candy_bar_cost (initial : ℕ) (final : ℕ) (h : initial = 4) (h' : final = 3) :
  initial - final = 1 := by
  sorry

end candy_bar_cost_l1006_100684


namespace chromium_percentage_in_combined_alloy_l1006_100605

/-- Calculates the percentage of chromium in a new alloy formed by combining two other alloys -/
theorem chromium_percentage_in_combined_alloy 
  (chromium_percent1 : ℝ) 
  (weight1 : ℝ) 
  (chromium_percent2 : ℝ) 
  (weight2 : ℝ) 
  (h1 : chromium_percent1 = 12)
  (h2 : weight1 = 15)
  (h3 : chromium_percent2 = 8)
  (h4 : weight2 = 40) :
  let total_chromium := (chromium_percent1 / 100) * weight1 + (chromium_percent2 / 100) * weight2
  let total_weight := weight1 + weight2
  (total_chromium / total_weight) * 100 = 9.09 := by
  sorry

end chromium_percentage_in_combined_alloy_l1006_100605


namespace tuesday_rainfall_amount_l1006_100683

/-- The amount of rainfall on Monday in inches -/
def monday_rain : ℝ := 0.9

/-- The difference in rainfall between Monday and Tuesday in inches -/
def rain_difference : ℝ := 0.7

/-- The amount of rainfall on Tuesday in inches -/
def tuesday_rain : ℝ := monday_rain - rain_difference

/-- Theorem stating that the amount of rain on Tuesday is 0.2 inches -/
theorem tuesday_rainfall_amount : tuesday_rain = 0.2 := by
  sorry

end tuesday_rainfall_amount_l1006_100683


namespace slope_range_l1006_100618

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := 4 * x^2 + 25 * y^2 = 100

-- Define the line equation
def line (m : ℝ) (x y : ℝ) : Prop := y = m * x - 3

-- Define the intersection condition
def intersects (m : ℝ) : Prop := ∃ x y : ℝ, ellipse x y ∧ line m x y

-- State the theorem
theorem slope_range (m : ℝ) : 
  intersects m ↔ m ≤ -Real.sqrt (1/5) ∨ m ≥ Real.sqrt (1/5) :=
sorry

end slope_range_l1006_100618


namespace arithmetic_sequence_common_difference_l1006_100699

/-- An arithmetic sequence with given terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_a2 : a 2 = 3)
  (h_a5 : a 5 = 9) :
  ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end arithmetic_sequence_common_difference_l1006_100699


namespace tea_mixture_price_l1006_100695

/-- Given two types of tea mixed in a 1:1 ratio, where one tea costs 62 rupees per kg
    and the mixture is worth 67 rupees per kg, prove that the price of the second tea
    is 72 rupees per kg. -/
theorem tea_mixture_price (price_tea1 price_mixture : ℚ) (ratio : ℚ × ℚ) :
  price_tea1 = 62 →
  price_mixture = 67 →
  ratio = (1, 1) →
  ∃ price_tea2 : ℚ, price_tea2 = 72 ∧
    (price_tea1 * ratio.1 + price_tea2 * ratio.2) / (ratio.1 + ratio.2) = price_mixture :=
by sorry

end tea_mixture_price_l1006_100695


namespace solve_equation_l1006_100681

theorem solve_equation : ∃ x : ℚ, (3 * x - 4) / 6 = 15 ∧ x = 94 / 3 := by sorry

end solve_equation_l1006_100681


namespace couple_driving_exam_probability_l1006_100657

/-- Represents the probability of passing an exam for each attempt -/
structure ExamProbability where
  male : ℚ
  female : ℚ

/-- Represents the exam attempt limits and fee structure -/
structure ExamRules where
  free_attempts : ℕ
  max_attempts : ℕ
  fee : ℚ

/-- Calculates the probability of a couple passing the exam under given conditions -/
def couple_exam_probability (prob : ExamProbability) (rules : ExamRules) : ℚ × ℚ :=
  sorry

theorem couple_driving_exam_probability :
  let prob := ExamProbability.mk (3/4) (2/3)
  let rules := ExamRules.mk 2 5 200
  let result := couple_exam_probability prob rules
  result.1 = 5/6 ∧ result.2 = 1/9 :=
sorry

end couple_driving_exam_probability_l1006_100657


namespace initial_eggs_count_l1006_100606

/-- Given a person shares eggs among 8 friends, with each friend receiving 2 eggs,
    prove that the initial number of eggs is 16. -/
theorem initial_eggs_count (num_friends : ℕ) (eggs_per_friend : ℕ) 
  (h1 : num_friends = 8) (h2 : eggs_per_friend = 2) : 
  num_friends * eggs_per_friend = 16 := by
  sorry

#check initial_eggs_count

end initial_eggs_count_l1006_100606


namespace prime_sequence_l1006_100697

theorem prime_sequence (n : ℕ) (h1 : n ≥ 2) 
  (h2 : ∀ k : ℕ, 0 ≤ k ∧ k ≤ Real.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) :
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ n - 2 → Nat.Prime (k^2 + k + n) :=
sorry

end prime_sequence_l1006_100697


namespace p_is_8x_squared_minus_8_l1006_100692

-- Define the numerator polynomial
def num (x : ℝ) : ℝ := x^4 - 2*x^3 - 7*x + 6

-- Define the properties of p(x)
def has_vertical_asymptotes (p : ℝ → ℝ) : Prop :=
  p 1 = 0 ∧ p (-1) = 0

def no_horizontal_asymptote (p : ℝ → ℝ) : Prop :=
  ∃ n : ℕ, ∀ x : ℝ, ∃ c : ℝ, |p x| ≤ c * |x|^n

-- Main theorem
theorem p_is_8x_squared_minus_8 (p : ℝ → ℝ) :
  has_vertical_asymptotes p →
  no_horizontal_asymptote p →
  p 2 = 24 →
  ∀ x : ℝ, p x = 8*x^2 - 8 :=
by sorry

end p_is_8x_squared_minus_8_l1006_100692


namespace shirt_price_calculation_l1006_100617

theorem shirt_price_calculation (final_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  final_price = 105 ∧ 
  discount1 = 19.954259576901087 ∧ 
  discount2 = 12.55 →
  ∃ (original_price : ℝ), 
    original_price = 150 ∧ 
    final_price = original_price * (1 - discount1 / 100) * (1 - discount2 / 100) :=
by sorry

end shirt_price_calculation_l1006_100617


namespace orthocenter_diameter_bisection_l1006_100641

/-- A point in a 2D plane. -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A triangle defined by three points. -/
structure Triangle :=
  (A B C : Point)

/-- The orthocenter of a triangle. -/
def orthocenter (t : Triangle) : Point := sorry

/-- The circumcircle of a triangle. -/
def circumcircle (t : Triangle) : Set Point := sorry

/-- A diameter of a circle. -/
def is_diameter (A A' : Point) (circle : Set Point) : Prop := sorry

/-- A segment bisects another segment. -/
def bisects (P Q : Point) (R S : Point) : Prop := sorry

/-- Main theorem: If H is the orthocenter of triangle ABC and AA' is a diameter
    of its circumcircle, then A'H bisects the side BC. -/
theorem orthocenter_diameter_bisection
  (t : Triangle) (A' : Point) (H : Point) :
  H = orthocenter t →
  is_diameter t.A A' (circumcircle t) →
  bisects A' H t.B t.C :=
sorry

end orthocenter_diameter_bisection_l1006_100641


namespace movie_original_length_l1006_100620

/-- The original length of a movie, given the length of a cut scene and the final length -/
def original_length (cut_scene_length final_length : ℕ) : ℕ :=
  final_length + cut_scene_length

/-- Theorem: The original length of the movie is 60 minutes -/
theorem movie_original_length : original_length 6 54 = 60 := by
  sorry

end movie_original_length_l1006_100620


namespace share_ratio_proof_l1006_100691

theorem share_ratio_proof (total amount : ℕ) (a_share b_share c_share : ℕ) 
  (h1 : amount = 595)
  (h2 : a_share + b_share + c_share = amount)
  (h3 : a_share = 420)
  (h4 : b_share = 105)
  (h5 : c_share = 70)
  (h6 : 3 * a_share = 2 * b_share) : 
  b_share * 2 = c_share * 3 := by
  sorry

end share_ratio_proof_l1006_100691


namespace unique_solution_in_interval_l1006_100614

theorem unique_solution_in_interval (x : Real) :
  x ∈ Set.Icc 0 (Real.pi / 2) →
  ((2 - Real.sin (2 * x)) * Real.sin (x + Real.pi / 4) = 1) ↔
  (x = Real.pi / 4) := by
  sorry

end unique_solution_in_interval_l1006_100614


namespace data_average_problem_l1006_100665

theorem data_average_problem (x : ℝ) : 
  (6 + x + 2 + 4) / 4 = 5 → x = 8 := by
  sorry

end data_average_problem_l1006_100665


namespace smallest_x_for_1680x_perfect_cube_l1006_100621

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

theorem smallest_x_for_1680x_perfect_cube : 
  (∀ x : ℕ, x > 0 ∧ x < 44100 → ¬(is_perfect_cube (1680 * x))) ∧
  (is_perfect_cube (1680 * 44100)) :=
by sorry

end smallest_x_for_1680x_perfect_cube_l1006_100621


namespace largest_inscribed_triangle_area_l1006_100678

theorem largest_inscribed_triangle_area (r : ℝ) (h : r = 8) :
  let circle_area := π * r^2
  let diameter := 2 * r
  let max_height := r
  let triangle_area := (1/2) * diameter * max_height
  triangle_area = 64 := by sorry

end largest_inscribed_triangle_area_l1006_100678


namespace tan_value_from_sin_cos_equation_l1006_100666

theorem tan_value_from_sin_cos_equation (α : ℝ) 
  (h : 3 * Real.sin ((33 * π) / 14 + α) = -5 * Real.cos ((5 * π) / 14 + α)) : 
  Real.tan ((5 * π) / 14 + α) = -5/3 := by
  sorry

end tan_value_from_sin_cos_equation_l1006_100666


namespace divisibility_and_modulo_l1006_100601

theorem divisibility_and_modulo (n : ℤ) (h : 11 ∣ (4 * n + 3)) : 
  n % 11 = 2 ∧ n^4 % 11 = 5 := by
  sorry

end divisibility_and_modulo_l1006_100601


namespace equilateral_triangle_area_perimeter_ratio_l1006_100632

/-- The ratio of the area to the square of the perimeter for an equilateral triangle with side length 10 -/
theorem equilateral_triangle_area_perimeter_ratio : 
  let side_length : ℝ := 10
  let perimeter : ℝ := 3 * side_length
  let height : ℝ := side_length * (Real.sqrt 3 / 2)
  let area : ℝ := (1 / 2) * side_length * height
  area / (perimeter ^ 2) = Real.sqrt 3 / 36 := by
  sorry

end equilateral_triangle_area_perimeter_ratio_l1006_100632


namespace hedge_cost_proof_l1006_100679

/-- Calculates the total cost of concrete blocks for a hedge --/
def total_cost (sections : ℕ) (blocks_per_section : ℕ) (cost_per_block : ℕ) : ℕ :=
  sections * blocks_per_section * cost_per_block

/-- Proves that the total cost of concrete blocks for the hedge is $480 --/
theorem hedge_cost_proof :
  total_cost 8 30 2 = 480 := by
  sorry

end hedge_cost_proof_l1006_100679


namespace pentagon_fifth_angle_l1006_100628

/-- A pentagon with four known angles -/
structure Pentagon where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  angle4 : ℝ
  angle5 : ℝ
  sum_of_angles : angle1 + angle2 + angle3 + angle4 + angle5 = 540

/-- The theorem to prove -/
theorem pentagon_fifth_angle (p : Pentagon) 
  (h1 : p.angle1 = 270)
  (h2 : p.angle2 = 70)
  (h3 : p.angle3 = 60)
  (h4 : p.angle4 = 90) :
  p.angle5 = 50 := by
  sorry


end pentagon_fifth_angle_l1006_100628


namespace rectangle_area_l1006_100630

/-- A rectangle with perimeter 100 meters and length three times the width has an area of 468.75 square meters. -/
theorem rectangle_area (l w : ℝ) (h1 : 2 * l + 2 * w = 100) (h2 : l = 3 * w) : l * w = 468.75 := by
  sorry

end rectangle_area_l1006_100630


namespace students_playing_both_sports_l1006_100659

/-- Given a school with 1200 students, where 875 play football, 450 play cricket, 
    and 100 neither play football nor cricket, prove that 225 students play both sports. -/
theorem students_playing_both_sports (total : ℕ) (football : ℕ) (cricket : ℕ) (neither : ℕ) :
  total = 1200 →
  football = 875 →
  cricket = 450 →
  neither = 100 →
  total - neither = football + cricket - 225 :=
by sorry

end students_playing_both_sports_l1006_100659


namespace cube_root_implies_value_l1006_100668

theorem cube_root_implies_value (x : ℝ) : 
  (2 * x - 14) ^ (1/3 : ℝ) = -2 → 2 * x + 3 = 9 := by
  sorry

end cube_root_implies_value_l1006_100668


namespace largest_three_digit_multiple_of_6_with_digit_sum_15_l1006_100603

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_multiple_of_6_with_digit_sum_15 :
  ∀ n : ℕ, is_three_digit n → n % 6 = 0 → digit_sum n = 15 → n ≤ 690 :=
by sorry

end largest_three_digit_multiple_of_6_with_digit_sum_15_l1006_100603


namespace range_of_a_complete_theorem_l1006_100674

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | 0 < x ∧ x < a}
def Q : Set ℝ := {x | -5 < x ∧ x < 1}

-- State the theorem
theorem range_of_a (a : ℝ) (ha : 0 < a) (h_union : P a ∪ Q = Q) : a ≤ 1 := by
  sorry

-- The complete theorem combining all conditions
theorem complete_theorem :
  ∃ a : ℝ, 0 < a ∧ P a ∪ Q = Q ∧ a ≤ 1 := by
  sorry

end range_of_a_complete_theorem_l1006_100674


namespace smallest_four_digit_in_pascal_l1006_100653

/-- Pascal's triangle function -/
def pascal (n : ℕ) (k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.choose n k

/-- Predicate for a number being in Pascal's triangle -/
def inPascalTriangle (m : ℕ) : Prop :=
  ∃ (n : ℕ) (k : ℕ), pascal n k = m

theorem smallest_four_digit_in_pascal : 
  (∀ m : ℕ, m < 1000 → ¬(inPascalTriangle m ∧ m ≥ 1000)) ∧ 
  inPascalTriangle 1000 := by
  sorry

end smallest_four_digit_in_pascal_l1006_100653


namespace frustum_volume_l1006_100664

/-- The volume of a frustum of a right square pyramid inscribed in a sphere -/
theorem frustum_volume (R : ℝ) (β : ℝ) : 
  (R > 0) → (π/4 < β) → (β < π/2) →
  ∃ V : ℝ, V = (2/3) * R^3 * Real.sin (2*β) * (1 + Real.cos (2*β)^2 - Real.cos (2*β)) :=
by sorry

end frustum_volume_l1006_100664


namespace man_work_days_l1006_100680

/-- Given a woman can complete a piece of work in 40 days and a man is 25% more efficient than a woman,
    prove that the man can complete the same piece of work in 32 days. -/
theorem man_work_days (woman_days : ℕ) (man_efficiency : ℚ) :
  woman_days = 40 →
  man_efficiency = 5 / 4 →
  ∃ (man_days : ℕ), man_days = 32 ∧ (man_days : ℚ) * man_efficiency = woman_days := by
  sorry

end man_work_days_l1006_100680


namespace constant_term_zero_l1006_100672

theorem constant_term_zero (x : ℝ) (x_pos : x > 0) : 
  (∃ k : ℕ, k ≤ 10 ∧ (10 - k) / 2 - k = 0) → False :=
sorry

end constant_term_zero_l1006_100672


namespace x1_range_proof_l1006_100649

theorem x1_range_proof (f : ℝ → ℝ) (h_incr : Monotone f) :
  (∀ x₁ x₂ : ℝ, x₁ + x₂ = 1 → f x₁ + f 0 > f x₂ + f 1) →
  ∀ x₁ : ℝ, (∃ x₂ : ℝ, x₁ + x₂ = 1 ∧ f x₁ + f 0 > f x₂ + f 1) → x₁ > 1 :=
by sorry

end x1_range_proof_l1006_100649


namespace cos_two_theta_value_l1006_100651

theorem cos_two_theta_value (θ : Real) 
  (h : Real.sin (θ / 2) + Real.cos (θ / 2) = 2 * Real.sqrt 2 / 3) : 
  Real.cos (2 * θ) = 79 / 81 := by
  sorry

end cos_two_theta_value_l1006_100651


namespace set_C_characterization_l1006_100661

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}

def B (a : ℝ) : Set ℝ := {x | a*x - 2 = 0}

def C : Set ℝ := {0, 1, 2}

theorem set_C_characterization :
  ∀ a : ℝ, (A ∪ B a = A) ↔ a ∈ C :=
sorry

end set_C_characterization_l1006_100661


namespace decode_1236_is_rand_l1006_100633

/-- Represents a coding scheme for words -/
structure CodeScheme where
  range_code : String
  random_code : String

/-- Decodes a given code based on the coding scheme -/
def decode (scheme : CodeScheme) (code : String) : String :=
  sorry

/-- The specific coding scheme used in the problem -/
def problem_scheme : CodeScheme :=
  { range_code := "12345", random_code := "123678" }

/-- The theorem stating that 1236 decodes to "rand" under the given scheme -/
theorem decode_1236_is_rand :
  decode problem_scheme "1236" = "rand" :=
sorry

end decode_1236_is_rand_l1006_100633


namespace tom_annual_cost_l1006_100642

/-- Calculates the annual cost of medication and doctor visits for Tom --/
def annual_cost (pills_per_day : ℕ) (doctor_visit_interval_months : ℕ) (doctor_visit_cost : ℕ) 
  (pill_cost : ℕ) (insurance_coverage_percent : ℕ) : ℕ :=
  let daily_medication_cost := pills_per_day * (pill_cost * (100 - insurance_coverage_percent) / 100)
  let annual_medication_cost := daily_medication_cost * 365
  let annual_doctor_visits := 12 / doctor_visit_interval_months
  let annual_doctor_cost := annual_doctor_visits * doctor_visit_cost
  annual_medication_cost + annual_doctor_cost

/-- Theorem stating that Tom's annual cost is $1530 --/
theorem tom_annual_cost : 
  annual_cost 2 6 400 5 80 = 1530 := by
  sorry

end tom_annual_cost_l1006_100642


namespace intersection_probability_is_four_sevenths_l1006_100690

/-- A rectangular prism with dimensions 3, 4, and 5 units -/
structure RectangularPrism where
  length : ℕ := 3
  width : ℕ := 4
  height : ℕ := 5

/-- The probability that a plane determined by three randomly chosen distinct vertices
    intersects the interior of the prism -/
def intersection_probability (prism : RectangularPrism) : ℚ :=
  4/7

/-- Theorem stating that the probability of intersection is 4/7 -/
theorem intersection_probability_is_four_sevenths (prism : RectangularPrism) :
  intersection_probability prism = 4/7 := by
  sorry

end intersection_probability_is_four_sevenths_l1006_100690


namespace special_sequence_2011_l1006_100612

/-- A sequence satisfying the given conditions -/
def special_sequence (a : ℕ → ℤ) : Prop :=
  a 201 = 2 ∧ ∀ n : ℕ, n > 0 → a n + a (n + 1) = 0

/-- The 2011th term of the special sequence is 2 -/
theorem special_sequence_2011 (a : ℕ → ℤ) (h : special_sequence a) : a 2011 = 2 := by
  sorry

end special_sequence_2011_l1006_100612


namespace probability_increases_l1006_100610

/-- The probability of player A winning a game of 2n rounds -/
noncomputable def P (n : ℕ) : ℝ :=
  1/2 * (1 - (Nat.choose (2*n) n : ℝ) / 2^(2*n))

/-- Theorem stating that the probability of winning increases with the number of rounds -/
theorem probability_increases (n : ℕ) : P (n+1) > P n := by
  sorry

end probability_increases_l1006_100610


namespace not_perfect_square_property_l1006_100629

def S : Set ℕ := {2, 5, 13}

theorem not_perfect_square_property (d : ℕ) (h1 : d ∉ S) (h2 : d > 0) :
  ∃ a b : ℕ, a ∈ S ∪ {d} ∧ b ∈ S ∪ {d} ∧ a ≠ b ∧ ¬∃ k : ℕ, a * b - 1 = k^2 :=
by sorry

end not_perfect_square_property_l1006_100629


namespace angle_D_value_l1006_100600

-- Define the angles as real numbers
variable (A B C D : ℝ)

-- State the theorem
theorem angle_D_value :
  A + B = 180 →  -- Condition 1
  C = D →        -- Condition 2
  C + 50 + 60 = 180 →  -- Condition 3
  D = 70 :=  -- Conclusion
by
  sorry  -- Proof is omitted as per instructions


end angle_D_value_l1006_100600


namespace right_triangle_hypotenuse_l1006_100639

noncomputable def hypotenuse_length (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2)

theorem right_triangle_hypotenuse (a b : ℝ) 
  (ha : a > 0) (hb : b > 0)
  (vol1 : (1/3) * Real.pi * b^2 * a = 1250 * Real.pi)
  (vol2 : (1/3) * Real.pi * a^2 * b = 2700 * Real.pi) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  abs (hypotenuse_length a b - 21.33) < ε :=
sorry

end right_triangle_hypotenuse_l1006_100639


namespace sixth_score_achieves_target_mean_l1006_100602

def test_scores : List ℝ := [78, 84, 76, 82, 88]
def sixth_score : ℝ := 102
def target_mean : ℝ := 85

theorem sixth_score_achieves_target_mean :
  (List.sum test_scores + sixth_score) / (test_scores.length + 1) = target_mean := by
  sorry

end sixth_score_achieves_target_mean_l1006_100602


namespace train_length_proof_l1006_100609

theorem train_length_proof (platform_crossing_time : ℝ) (pole_crossing_time : ℝ) (platform_length : ℝ) :
  platform_crossing_time = 39 →
  pole_crossing_time = 18 →
  platform_length = 1050 →
  ∃ train_length : ℝ, train_length = 900 ∧
    train_length / pole_crossing_time = (train_length + platform_length) / platform_crossing_time :=
by
  sorry

end train_length_proof_l1006_100609


namespace number_problem_l1006_100619

theorem number_problem : 
  ∃ (n : ℝ), n - (102 / 20.4) = 5095 ∧ n = 5100 := by
  sorry

end number_problem_l1006_100619


namespace sum_of_products_is_negative_one_l1006_100662

-- Define the polynomial Q(x)
def Q (x : ℝ) : ℝ := x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1

-- Define the theorem
theorem sum_of_products_is_negative_one 
  (d₁ d₂ d₃ d₄ e₁ e₂ e₃ e₄ : ℝ) 
  (h : ∀ x : ℝ, Q x = (x^2 + d₁*x + e₁) * (x^2 + d₂*x + e₂) * (x^2 + d₃*x + e₃) * (x^2 + d₄*x + e₄)) : 
  d₁*e₁ + d₂*e₂ + d₃*e₃ + d₄*e₄ = -1 := by
sorry

end sum_of_products_is_negative_one_l1006_100662


namespace cubic_minus_linear_factorization_l1006_100631

theorem cubic_minus_linear_factorization (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) := by
  sorry

end cubic_minus_linear_factorization_l1006_100631


namespace parabola_focus_directrix_l1006_100654

/-- For a parabola with equation y² = ax, if the distance from its focus to its directrix is 2, then a = ±4 -/
theorem parabola_focus_directrix (a : ℝ) : 
  (∃ (y x : ℝ), y^2 = a*x) →  -- parabola equation
  (∃ (p : ℝ), p = 2) →        -- distance from focus to directrix
  (a = 4 ∨ a = -4) :=
by sorry

end parabola_focus_directrix_l1006_100654


namespace sin_plus_2cos_period_l1006_100647

open Real

/-- The function f(x) = sin x + 2cos x has a period of 2π. -/
theorem sin_plus_2cos_period : ∃ (k : ℝ), k > 0 ∧ ∀ x, sin x + 2 * cos x = sin (x + k) + 2 * cos (x + k) := by
  use 2 * π
  constructor
  · exact two_pi_pos
  · intro x
    sorry


end sin_plus_2cos_period_l1006_100647


namespace tank_fill_time_l1006_100675

/-- Represents a machine that can fill or empty a tank -/
structure Machine where
  fillRate : ℚ  -- Rate at which the machine fills the tank (fraction per minute)
  emptyRate : ℚ -- Rate at which the machine empties the tank (fraction per minute)

/-- Calculates the net rate of a machine that alternates between filling and emptying -/
def alternatingRate (fillTime emptyTime cycleTime : ℚ) : ℚ :=
  (fillTime / cycleTime) * (1 / fillTime) + (emptyTime / cycleTime) * (-1 / emptyTime)

/-- The main theorem stating the time to fill the tank -/
theorem tank_fill_time :
  let machineA : Machine := ⟨1/25, 0⟩
  let machineB : Machine := ⟨0, 1/50⟩
  let machineC : Machine := ⟨alternatingRate 5 5 10, 0⟩
  let combinedRate := machineA.fillRate - machineB.emptyRate + machineC.fillRate
  let remainingVolume := 1/2
  ⌈remainingVolume / combinedRate⌉ = 20 := by
  sorry


end tank_fill_time_l1006_100675


namespace last_three_average_l1006_100669

theorem last_three_average (list : List ℝ) : 
  list.length = 6 →
  list.sum / 6 = 60 →
  (list.take 3).sum / 3 = 55 →
  (list.drop 3).sum / 3 = 65 := by
sorry

end last_three_average_l1006_100669


namespace largest_square_factor_of_10_factorial_l1006_100655

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem largest_square_factor_of_10_factorial :
  ∀ n : ℕ, n ≤ 10 → (factorial n)^2 ≤ factorial 10 →
  (factorial n)^2 ≤ (factorial 6)^2 := by
  sorry

end largest_square_factor_of_10_factorial_l1006_100655


namespace cube_sum_given_sum_and_square_sum_l1006_100689

theorem cube_sum_given_sum_and_square_sum (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x^2 + y^2 = 17) : 
  x^3 + y^3 = 65 := by
  sorry

end cube_sum_given_sum_and_square_sum_l1006_100689


namespace molar_mass_calculation_l1006_100650

/-- Given a chemical compound where 10 moles weigh 2070 grams, 
    prove that its molar mass is 207 grams/mole. -/
theorem molar_mass_calculation (total_weight : ℝ) (num_moles : ℝ) 
  (h1 : total_weight = 2070)
  (h2 : num_moles = 10) :
  total_weight / num_moles = 207 := by
  sorry

end molar_mass_calculation_l1006_100650


namespace paint_distribution_l1006_100611

theorem paint_distribution (total_paint : ℝ) (num_colors : ℕ) (paint_per_color : ℝ) :
  total_paint = 15 →
  num_colors = 3 →
  paint_per_color * num_colors = total_paint →
  paint_per_color = 5 := by
  sorry

end paint_distribution_l1006_100611


namespace ede_viv_properties_l1006_100670

theorem ede_viv_properties :
  let ede : ℕ := 242
  let viv : ℕ := 303
  (100 ≤ ede ∧ ede < 1000) ∧  -- EDE is a three-digit number
  (100 ≤ viv ∧ viv < 1000) ∧  -- VIV is a three-digit number
  (ede ≠ viv) ∧               -- EDE and VIV are distinct
  (Nat.gcd ede viv = 1) ∧     -- EDE and VIV are relatively prime
  (ede / viv = 242 / 303) ∧   -- The fraction is correct
  (∃ n : ℕ, (1000 * ede) / viv = 798 + n * 999) -- The decimal repeats as 0.798679867...
  := by sorry

end ede_viv_properties_l1006_100670


namespace john_nails_count_l1006_100645

/-- Calculates the total number of nails used in John's house wall construction --/
def total_nails (nails_per_plank : ℕ) (additional_nails : ℕ) (num_planks : ℕ) : ℕ :=
  nails_per_plank * num_planks + additional_nails

/-- Proves that John used 11 nails in total for his house wall construction --/
theorem john_nails_count :
  let nails_per_plank : ℕ := 3
  let additional_nails : ℕ := 8
  let num_planks : ℕ := 1
  total_nails nails_per_plank additional_nails num_planks = 11 := by
  sorry

end john_nails_count_l1006_100645


namespace ratio_x_to_y_l1006_100673

theorem ratio_x_to_y (x y : ℝ) (h : (12 * x - 5 * y) / (16 * x - 3 * y) = 5 / 7) : 
  x / y = 5 / 1 := by sorry

end ratio_x_to_y_l1006_100673


namespace quadratic_polynomial_property_l1006_100671

def P (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

theorem quadratic_polynomial_property (a b : ℝ) :
  P a b 10 + P a b 30 = 40 → P a b 20 = -80 := by
  sorry

end quadratic_polynomial_property_l1006_100671


namespace area_ratio_theorem_l1006_100637

/-- A right triangle with a point on its hypotenuse and parallel lines dividing it -/
structure DividedRightTriangle where
  -- The rectangle formed by the parallel lines
  rectangle_area : ℝ
  -- The area of one of the smaller right triangles
  small_triangle_area : ℝ
  -- The condition that the area of one small triangle is n times the rectangle area
  area_condition : ∃ n : ℝ, small_triangle_area = n * rectangle_area

/-- The theorem stating the ratio of areas -/
theorem area_ratio_theorem (t : DividedRightTriangle) : 
  ∃ n : ℝ, t.small_triangle_area = n * t.rectangle_area → 
  ∃ other_triangle_area : ℝ, other_triangle_area / t.rectangle_area = 1 / (4 * n) :=
sorry

end area_ratio_theorem_l1006_100637


namespace common_point_tangent_line_l1006_100646

theorem common_point_tangent_line (a : ℝ) (h_a : a > 0) :
  ∃ x : ℝ, x > 0 ∧ 
    a * Real.sqrt x = Real.log (Real.sqrt x) ∧
    (a / (2 * Real.sqrt x) = 1 / (2 * x)) →
  a = Real.exp (-1) := by
  sorry

end common_point_tangent_line_l1006_100646


namespace pyramid_volume_change_l1006_100626

theorem pyramid_volume_change (s h : ℝ) : 
  s > 0 → h > 0 → (1/3 : ℝ) * s^2 * h = 60 → 
  (1/3 : ℝ) * (3*s)^2 * (2*h) = 1080 := by
sorry

end pyramid_volume_change_l1006_100626


namespace toys_per_box_l1006_100667

/-- Given that Paul filled up four boxes and packed a total of 32 toys,
    prove that the number of toys in each box is 8. -/
theorem toys_per_box (total_toys : ℕ) (num_boxes : ℕ) (h1 : total_toys = 32) (h2 : num_boxes = 4) :
  total_toys / num_boxes = 8 := by
  sorry

end toys_per_box_l1006_100667


namespace probability_A_and_B_selected_is_three_tenths_l1006_100607

def total_students : ℕ := 5
def selected_students : ℕ := 3

def probability_A_and_B_selected : ℚ :=
  (Nat.choose (total_students - 2) (selected_students - 2)) /
  (Nat.choose total_students selected_students)

theorem probability_A_and_B_selected_is_three_tenths :
  probability_A_and_B_selected = 3 / 10 := by
  sorry

end probability_A_and_B_selected_is_three_tenths_l1006_100607


namespace melody_reading_pages_l1006_100640

theorem melody_reading_pages (english : ℕ) (civics : ℕ) (chinese : ℕ) (science : ℕ) :
  english = 20 →
  civics = 8 →
  chinese = 12 →
  (english / 4 + civics / 4 + chinese / 4 + science / 4 : ℚ) = 14 →
  science = 16 := by
sorry

end melody_reading_pages_l1006_100640


namespace smallest_sum_20_consecutive_triangular_l1006_100643

/-- The sum of 20 consecutive integers starting from n -/
def sum_20_consecutive (n : ℤ) : ℤ := 10 * (2 * n + 19)

/-- A triangular number -/
def triangular_number (m : ℕ) : ℕ := m * (m + 1) / 2

/-- Proposition: 190 is the smallest sum of 20 consecutive integers that is also a triangular number -/
theorem smallest_sum_20_consecutive_triangular :
  ∃ (m : ℕ), 
    (∀ (n : ℤ), sum_20_consecutive n ≥ 190) ∧ 
    (sum_20_consecutive 0 = 190) ∧ 
    (triangular_number m = 190) :=
sorry

end smallest_sum_20_consecutive_triangular_l1006_100643


namespace square_gt_of_abs_lt_l1006_100682

theorem square_gt_of_abs_lt (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end square_gt_of_abs_lt_l1006_100682


namespace probability_one_red_ball_l1006_100608

theorem probability_one_red_ball (total_balls : ℕ) (red_balls : ℕ) (yellow_balls : ℕ) 
  (h1 : total_balls = red_balls + yellow_balls)
  (h2 : red_balls = 3)
  (h3 : yellow_balls = 2)
  (h4 : total_balls ≥ 2) :
  (red_balls.choose 1 * yellow_balls.choose 1 : ℚ) / total_balls.choose 2 = 3/5 := by
  sorry

end probability_one_red_ball_l1006_100608


namespace miss_darlington_blueberries_l1006_100644

/-- The number of blueberries in Miss Darlington's basket problem -/
theorem miss_darlington_blueberries 
  (initial_basket : ℕ) 
  (additional_baskets : ℕ) 
  (h1 : initial_basket = 20)
  (h2 : additional_baskets = 9) : 
  initial_basket + additional_baskets * initial_basket = 200 := by
  sorry

end miss_darlington_blueberries_l1006_100644


namespace loyalty_program_benefits_l1006_100616

-- Define the structure for a bank
structure Bank where
  cardUsage : ℝ
  customerLoyalty : ℝ
  transactionVolume : ℝ

-- Define the structure for the Central Bank
structure CentralBank where
  nationalPaymentSystemUsage : ℝ
  consumerSpending : ℝ

-- Define the effect of the loyalty program
def loyaltyProgramEffect (bank : Bank) (centralBank : CentralBank) : Bank × CentralBank :=
  let newBank : Bank := {
    cardUsage := bank.cardUsage * 1.2,
    customerLoyalty := bank.customerLoyalty * 1.15,
    transactionVolume := bank.transactionVolume * 1.25
  }
  let newCentralBank : CentralBank := {
    nationalPaymentSystemUsage := centralBank.nationalPaymentSystemUsage * 1.3,
    consumerSpending := centralBank.consumerSpending * 1.1
  }
  (newBank, newCentralBank)

-- Theorem stating the benefits of the loyalty program
theorem loyalty_program_benefits 
  (bank : Bank) 
  (centralBank : CentralBank) :
  let (newBank, newCentralBank) := loyaltyProgramEffect bank centralBank
  newBank.cardUsage > bank.cardUsage ∧
  newBank.customerLoyalty > bank.customerLoyalty ∧
  newBank.transactionVolume > bank.transactionVolume ∧
  newCentralBank.nationalPaymentSystemUsage > centralBank.nationalPaymentSystemUsage ∧
  newCentralBank.consumerSpending > centralBank.consumerSpending :=
by
  sorry


end loyalty_program_benefits_l1006_100616


namespace walters_age_l1006_100696

theorem walters_age (walter_age_2005 : ℕ) (grandmother_age_2005 : ℕ) : 
  walter_age_2005 = grandmother_age_2005 / 3 →
  (2005 - walter_age_2005) + (2005 - grandmother_age_2005) = 3858 →
  walter_age_2005 + 5 = 43 :=
by
  sorry

end walters_age_l1006_100696


namespace solve_y_l1006_100625

theorem solve_y (x y : ℝ) (h1 : x - y = 16) (h2 : x + y = 10) : y = -3 := by
  sorry

end solve_y_l1006_100625


namespace N2O3_molecular_weight_l1006_100698

/-- The atomic weight of nitrogen in atomic mass units (amu) -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of oxygen in atomic mass units (amu) -/
def oxygen_weight : ℝ := 16.00

/-- The number of nitrogen atoms in N2O3 -/
def nitrogen_count : ℕ := 2

/-- The number of oxygen atoms in N2O3 -/
def oxygen_count : ℕ := 3

/-- The molecular weight of N2O3 in atomic mass units (amu) -/
def N2O3_weight : ℝ := nitrogen_weight * nitrogen_count + oxygen_weight * oxygen_count

theorem N2O3_molecular_weight : N2O3_weight = 76.02 := by sorry

end N2O3_molecular_weight_l1006_100698


namespace binary_1111111111_equals_1023_l1006_100622

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1111111111 -/
def binary_1111111111 : List Bool :=
  [true, true, true, true, true, true, true, true, true, true]

theorem binary_1111111111_equals_1023 :
  binary_to_decimal binary_1111111111 = 1023 := by
  sorry

end binary_1111111111_equals_1023_l1006_100622


namespace rotated_line_equation_l1006_100658

/-- Represents a line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : a * x + b * y + c = 0

/-- Rotates a line counterclockwise by π/2 around its y-axis intersection --/
def rotate_line_pi_over_2 (l : Line) : Line :=
  sorry

theorem rotated_line_equation :
  let original_line : Line := ⟨2, -1, -2, by sorry⟩
  let rotated_line := rotate_line_pi_over_2 original_line
  rotated_line.a = 1 ∧ rotated_line.b = 2 ∧ rotated_line.c = 4 :=
sorry

end rotated_line_equation_l1006_100658


namespace greatest_prime_factor_f_28_l1006_100676

def f (m : ℕ) : ℕ := Finset.prod (Finset.range (m/2 + 1)) (λ i => 2 * i)

theorem greatest_prime_factor_f_28 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ f 28 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ f 28 → q ≤ p :=
by sorry

end greatest_prime_factor_f_28_l1006_100676


namespace shirt_cost_l1006_100638

/-- Proves that the cost of each shirt is $50 given the sales and commission information --/
theorem shirt_cost (commission_rate : ℝ) (suit_price : ℝ) (suit_count : ℕ)
  (shirt_count : ℕ) (loafer_price : ℝ) (loafer_count : ℕ) (total_commission : ℝ) :
  commission_rate = 0.15 →
  suit_price = 700 →
  suit_count = 2 →
  shirt_count = 6 →
  loafer_price = 150 →
  loafer_count = 2 →
  total_commission = 300 →
  ∃ (shirt_price : ℝ), 
    total_commission = commission_rate * (suit_price * suit_count + shirt_price * shirt_count + loafer_price * loafer_count) ∧
    shirt_price = 50 := by
  sorry

end shirt_cost_l1006_100638


namespace star_equation_solution_l1006_100648

/-- Custom binary operation -/
def star (a b : ℝ) : ℝ := a * b + a - 2 * b

/-- Theorem stating that if 3 star m = 17, then m = 14 -/
theorem star_equation_solution :
  ∀ m : ℝ, star 3 m = 17 → m = 14 := by
  sorry

end star_equation_solution_l1006_100648


namespace total_paintable_area_l1006_100604

/-- Calculate the total paintable area for four bedrooms --/
theorem total_paintable_area (
  num_bedrooms : ℕ)
  (length width height : ℝ)
  (window_area : ℝ) :
  num_bedrooms = 4 →
  length = 14 →
  width = 11 →
  height = 9 →
  window_area = 70 →
  (num_bedrooms : ℝ) * ((2 * (length * height + width * height)) - window_area) = 1520 := by
  sorry

end total_paintable_area_l1006_100604


namespace reflection_line_sum_l1006_100613

/-- Given a line y = mx + b, if the reflection of point (2, 2) across this line is (8, 6), then m + b = 10 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∀ (x y : ℝ), (x - 2)^2 + (y - 2)^2 = (8 - x)^2 + (6 - y)^2 → y = m*x + b) →
  m + b = 10 := by sorry

end reflection_line_sum_l1006_100613


namespace expected_terms_is_ten_l1006_100634

/-- A fair tetrahedral die with faces numbered 1 to 4 -/
structure TetrahedralDie :=
  (faces : Finset Nat)
  (fair : faces = {1, 2, 3, 4})

/-- The state of the sequence -/
inductive SequenceState
| Zero  : SequenceState  -- No distinct numbers seen
| One   : SequenceState  -- One distinct number seen
| Two   : SequenceState  -- Two distinct numbers seen
| Three : SequenceState  -- Three distinct numbers seen
| Four  : SequenceState  -- All four numbers seen

/-- Expected number of terms to complete the sequence from a given state -/
noncomputable def expectedTerms (s : SequenceState) : ℝ :=
  match s with
  | SequenceState.Zero  => sorry
  | SequenceState.One   => sorry
  | SequenceState.Two   => sorry
  | SequenceState.Three => sorry
  | SequenceState.Four  => 0

/-- Main theorem: The expected number of terms in the sequence is 10 -/
theorem expected_terms_is_ten (d : TetrahedralDie) : 
  expectedTerms SequenceState.Zero = 10 := by sorry

end expected_terms_is_ten_l1006_100634


namespace maria_piggy_bank_theorem_l1006_100635

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "dime" => 10
  | "quarter" => 25
  | "nickel" => 5
  | _ => 0

/-- Calculates the total value of coins in dollars -/
def total_value (dimes quarters nickels additional_quarters : ℕ) : ℚ :=
  (dimes * coin_value "dime" +
   (quarters + additional_quarters) * coin_value "quarter" +
   nickels * coin_value "nickel") / 100

theorem maria_piggy_bank_theorem (dimes quarters nickels additional_quarters : ℕ)
  (h1 : dimes = 4)
  (h2 : quarters = 4)
  (h3 : nickels = 7)
  (h4 : additional_quarters = 5) :
  total_value dimes quarters nickels additional_quarters = 3 :=
sorry

end maria_piggy_bank_theorem_l1006_100635


namespace bus_left_seats_l1006_100624

/-- Represents the seating configuration of a bus -/
structure BusSeating where
  leftSeats : ℕ
  rightSeats : ℕ
  backSeatCapacity : ℕ
  seatCapacity : ℕ
  totalCapacity : ℕ

/-- The bus seating configuration satisfies the given conditions -/
def validBusSeating (bus : BusSeating) : Prop :=
  bus.rightSeats = bus.leftSeats - 3 ∧
  bus.backSeatCapacity = 12 ∧
  bus.seatCapacity = 3 ∧
  bus.totalCapacity = 93 ∧
  bus.totalCapacity = bus.seatCapacity * (bus.leftSeats + bus.rightSeats) + bus.backSeatCapacity

theorem bus_left_seats (bus : BusSeating) (h : validBusSeating bus) : bus.leftSeats = 15 := by
  sorry

end bus_left_seats_l1006_100624


namespace geometric_series_sum_l1006_100660

theorem geometric_series_sum (a r : ℚ) (n : ℕ) (h : r ≠ 1) :
  let S := (a * (1 - r^n)) / (1 - r)
  a = 1 → r = 1/4 → n = 6 → S = 1365/1024 := by
  sorry

end geometric_series_sum_l1006_100660


namespace inverse_proportion_problem_l1006_100656

/-- Given that y and x are inversely proportional -/
def inversely_proportional (y x : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ y * x = k

/-- The theorem to prove -/
theorem inverse_proportion_problem (y₁ y₂ : ℝ) :
  inversely_proportional y₁ 4 ∧ y₁ = 30 →
  inversely_proportional y₂ 10 →
  y₂ = 12 := by
  sorry

end inverse_proportion_problem_l1006_100656


namespace average_payment_is_657_l1006_100688

/-- Represents the payment structure for a debt over a year -/
structure DebtPayment where
  base : ℕ  -- Base payment amount
  increment1 : ℕ  -- Increment for second segment
  increment2 : ℕ  -- Increment for third segment
  increment3 : ℕ  -- Increment for fourth segment
  increment4 : ℕ  -- Increment for fifth segment

/-- Calculates the average payment given the debt payment structure -/
def averagePayment (dp : DebtPayment) : ℚ :=
  let total := 
    20 * dp.base + 
    30 * (dp.base + dp.increment1) + 
    40 * (dp.base + dp.increment1 + dp.increment2) + 
    50 * (dp.base + dp.increment1 + dp.increment2 + dp.increment3) + 
    60 * (dp.base + dp.increment1 + dp.increment2 + dp.increment3 + dp.increment4)
  total / 200

/-- Theorem stating that the average payment for the given structure is $657 -/
theorem average_payment_is_657 (dp : DebtPayment) 
    (h1 : dp.base = 450)
    (h2 : dp.increment1 = 80)
    (h3 : dp.increment2 = 65)
    (h4 : dp.increment3 = 105)
    (h5 : dp.increment4 = 95) : 
  averagePayment dp = 657 := by
  sorry

end average_payment_is_657_l1006_100688
