import Mathlib

namespace NUMINAMATH_CALUDE_magic_square_solution_l2429_242984

/-- Represents a 3x3 magic square -/
structure MagicSquare :=
  (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℤ)

/-- The sum of any row, column, or diagonal in a magic square is the same -/
def is_magic (s : MagicSquare) : Prop :=
  let sum := s.a11 + s.a12 + s.a13
  sum = s.a21 + s.a22 + s.a23 ∧
  sum = s.a31 + s.a32 + s.a33 ∧
  sum = s.a11 + s.a21 + s.a31 ∧
  sum = s.a12 + s.a22 + s.a32 ∧
  sum = s.a13 + s.a23 + s.a33 ∧
  sum = s.a11 + s.a22 + s.a33 ∧
  sum = s.a13 + s.a22 + s.a31

theorem magic_square_solution :
  ∀ (s : MagicSquare),
    is_magic s →
    s.a11 = s.a11 ∧ s.a12 = 25 ∧ s.a13 = 75 ∧ s.a21 = 5 →
    s.a11 = 310 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_solution_l2429_242984


namespace NUMINAMATH_CALUDE_shoe_selection_theorem_l2429_242959

/-- The number of pairs of shoes in the bag -/
def total_pairs : ℕ := 10

/-- The number of shoes randomly selected -/
def selected_shoes : ℕ := 4

/-- The number of ways to select 4 shoes such that none of them form a pair -/
def no_pairs : ℕ := 3360

/-- The number of ways to select 4 shoes such that exactly 2 pairs are formed -/
def two_pairs : ℕ := 45

/-- The number of ways to select 4 shoes such that 2 shoes form a pair and the other 2 do not -/
def one_pair : ℕ := 1440

theorem shoe_selection_theorem :
  (Nat.choose total_pairs selected_shoes * 2^selected_shoes = no_pairs) ∧
  (Nat.choose total_pairs 2 = two_pairs) ∧
  (total_pairs * Nat.choose (total_pairs - 1) 2 * 2^2 = one_pair) :=
by sorry

end NUMINAMATH_CALUDE_shoe_selection_theorem_l2429_242959


namespace NUMINAMATH_CALUDE_no_natural_solution_l2429_242945

theorem no_natural_solution :
  ¬ ∃ (x y : ℕ), x^4 - y^4 = x^3 + y^3 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solution_l2429_242945


namespace NUMINAMATH_CALUDE_baseball_card_difference_l2429_242914

theorem baseball_card_difference (marcus_cards carter_cards : ℕ) 
  (h1 : marcus_cards = 210) 
  (h2 : carter_cards = 152) : 
  marcus_cards - carter_cards = 58 := by
sorry

end NUMINAMATH_CALUDE_baseball_card_difference_l2429_242914


namespace NUMINAMATH_CALUDE_income_percentage_l2429_242937

theorem income_percentage (juan tim mart : ℝ) 
  (h1 : mart = tim + 0.6 * tim) 
  (h2 : tim = juan - 0.6 * juan) : 
  mart = 0.64 * juan := by
sorry

end NUMINAMATH_CALUDE_income_percentage_l2429_242937


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l2429_242955

theorem binomial_coefficient_two (n : ℕ+) : Nat.choose n.val 2 = n.val * (n.val - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l2429_242955


namespace NUMINAMATH_CALUDE_symmetry_of_graphs_l2429_242947

-- Define a function f from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Define a real number a
variable (a : ℝ)

-- Define symmetry about a vertical line
def symmetricAboutVerticalLine (g h : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x y, g x = y ↔ h (2*c - x) = y

-- State the theorem
theorem symmetry_of_graphs :
  symmetricAboutVerticalLine (fun x ↦ f (a - x)) (fun x ↦ f (x - a)) a :=
sorry

end NUMINAMATH_CALUDE_symmetry_of_graphs_l2429_242947


namespace NUMINAMATH_CALUDE_faucet_filling_time_faucet_filling_time_is_135_l2429_242920

/-- If three faucets can fill a 100-gallon tub in 6 minutes, 
    then four faucets will fill a 50-gallon tub in 135 seconds. -/
theorem faucet_filling_time : ℝ → Prop :=
  fun time_seconds =>
    let three_faucet_volume : ℝ := 100  -- gallons
    let three_faucet_time : ℝ := 6    -- minutes
    let four_faucet_volume : ℝ := 50   -- gallons
    
    let one_faucet_rate : ℝ := three_faucet_volume / (3 * three_faucet_time)
    let four_faucet_rate : ℝ := 4 * one_faucet_rate
    
    time_seconds = (four_faucet_volume / four_faucet_rate) * 60

theorem faucet_filling_time_is_135 : faucet_filling_time 135 := by sorry

end NUMINAMATH_CALUDE_faucet_filling_time_faucet_filling_time_is_135_l2429_242920


namespace NUMINAMATH_CALUDE_hyundai_dodge_ratio_l2429_242912

theorem hyundai_dodge_ratio (total : ℕ) (dodge : ℕ) (kia : ℕ) (hyundai : ℕ) :
  total = 400 →
  dodge = total / 2 →
  kia = 100 →
  hyundai = total - dodge - kia →
  (hyundai : ℚ) / dodge = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_hyundai_dodge_ratio_l2429_242912


namespace NUMINAMATH_CALUDE_goldies_earnings_l2429_242926

/-- Calculates the total earnings for pet-sitting over two weeks -/
def total_earnings (hourly_rate : ℕ) (hours_week1 : ℕ) (hours_week2 : ℕ) : ℕ :=
  hourly_rate * hours_week1 + hourly_rate * hours_week2

/-- Proves that Goldie's total earnings for two weeks of pet-sitting is $250 -/
theorem goldies_earnings : total_earnings 5 20 30 = 250 := by
  sorry

end NUMINAMATH_CALUDE_goldies_earnings_l2429_242926


namespace NUMINAMATH_CALUDE_baker_new_cakes_l2429_242902

theorem baker_new_cakes (initial_cakes sold_cakes current_cakes : ℕ) 
  (h1 : initial_cakes = 121)
  (h2 : sold_cakes = 105)
  (h3 : current_cakes = 186) :
  current_cakes - (initial_cakes - sold_cakes) = 170 := by
  sorry

end NUMINAMATH_CALUDE_baker_new_cakes_l2429_242902


namespace NUMINAMATH_CALUDE_chicken_purchase_equation_l2429_242971

/-- Represents a group purchase scenario -/
structure GroupPurchase where
  numPeople : ℕ
  itemPrice : ℕ

/-- Calculates the surplus or shortage in a group purchase -/
def calculateDifference (g : GroupPurchase) (contribution : ℕ) : ℤ :=
  (g.numPeople * contribution : ℤ) - g.itemPrice

/-- Theorem stating the correct equation for the chicken purchase problem -/
theorem chicken_purchase_equation (g : GroupPurchase) :
  calculateDifference g 9 = 11 ∧ calculateDifference g 6 = -16 →
  9 * g.numPeople - 11 = 6 * g.numPeople + 16 := by
  sorry


end NUMINAMATH_CALUDE_chicken_purchase_equation_l2429_242971


namespace NUMINAMATH_CALUDE_ellipse_properties_l2429_242986

-- Define the ellipse C
def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the line l
def line (x y k m : ℝ) : Prop := y = k * x + m

-- Define the conditions
def conditions (a b k m : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ 2 * b = 2 ∧ (a^2 - b^2) / a^2 = 1/2

-- Define the perpendicular bisector condition
def perp_bisector_condition (k m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse x₁ y₁ (Real.sqrt 2) 1 ∧
    ellipse x₂ y₂ (Real.sqrt 2) 1 ∧
    line x₁ y₁ k m ∧
    line x₂ y₂ k m ∧
    (y₁ + y₂) / 2 + 1/2 = -1/k * ((x₁ + x₂) / 2)

-- Define the theorem
theorem ellipse_properties (a b k m : ℝ) :
  conditions a b k m →
  (∀ x y, ellipse x y a b ↔ ellipse x y (Real.sqrt 2) 1) ∧
  (perp_bisector_condition k m → 2 * k^2 + 1 = 2 * m) ∧
  (∃ (S : ℝ → ℝ), (∀ k m, perp_bisector_condition k m → S m ≤ Real.sqrt 2 / 2) ∧
                  (∃ k₀ m₀, perp_bisector_condition k₀ m₀ ∧ S m₀ = Real.sqrt 2 / 2)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2429_242986


namespace NUMINAMATH_CALUDE_parabola_focus_l2429_242999

/-- The parabola equation: x = -1/4 * y^2 + 2 -/
def parabola_equation (x y : ℝ) : Prop := x = -1/4 * y^2 + 2

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (1, 0)

/-- Theorem: The focus of the parabola x = -1/4 * y^2 + 2 is at the point (1, 0) -/
theorem parabola_focus :
  ∀ (x y : ℝ), parabola_equation x y →
  let (fx, fy) := focus
  (x - fx)^2 + (y - fy)^2 = (x - (fx + 2))^2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_l2429_242999


namespace NUMINAMATH_CALUDE_factorization_2m_cubed_minus_8m_l2429_242922

theorem factorization_2m_cubed_minus_8m (m : ℝ) : 2*m^3 - 8*m = 2*m*(m+2)*(m-2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_2m_cubed_minus_8m_l2429_242922


namespace NUMINAMATH_CALUDE_yellow_given_popped_l2429_242956

/-- Represents the types of kernels in the bag -/
inductive KernelType
  | White
  | Yellow
  | Brown

/-- The probability of selecting a kernel of a given type -/
def selectionProb (k : KernelType) : ℚ :=
  match k with
  | KernelType.White => 3/5
  | KernelType.Yellow => 1/5
  | KernelType.Brown => 1/5

/-- The probability of a kernel popping given its type -/
def poppingProb (k : KernelType) : ℚ :=
  match k with
  | KernelType.White => 1/3
  | KernelType.Yellow => 3/4
  | KernelType.Brown => 1/2

/-- The probability of selecting and popping a kernel of a given type -/
def selectAndPopProb (k : KernelType) : ℚ :=
  selectionProb k * poppingProb k

/-- The total probability of selecting and popping any kernel -/
def totalPopProb : ℚ :=
  selectAndPopProb KernelType.White + selectAndPopProb KernelType.Yellow + selectAndPopProb KernelType.Brown

/-- The probability that a popped kernel is yellow -/
theorem yellow_given_popped :
  selectAndPopProb KernelType.Yellow / totalPopProb = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_yellow_given_popped_l2429_242956


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2429_242988

theorem quadratic_equations_solutions :
  (∀ x : ℝ, (x + 4)^2 - 5*(x + 4) = 0 ↔ x = -4 ∨ x = 1) ∧
  (∀ x : ℝ, x^2 - 2*x - 15 = 0 ↔ x = -3 ∨ x = 5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2429_242988


namespace NUMINAMATH_CALUDE_replacement_count_l2429_242900

-- Define the replacement percentage
def replacement_percentage : ℝ := 0.2

-- Define the final milk percentage
def final_milk_percentage : ℝ := 0.5120000000000001

-- Define the function to calculate the remaining milk percentage after n replacements
def remaining_milk (n : ℕ) : ℝ := (1 - replacement_percentage) ^ n

-- Theorem statement
theorem replacement_count : ∃ n : ℕ, remaining_milk n = final_milk_percentage ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_replacement_count_l2429_242900


namespace NUMINAMATH_CALUDE_directional_vector_of_line_l2429_242980

/-- Given a line with equation 3x + 2y - 1 = 0, prove that (2, -3) is a directional vector --/
theorem directional_vector_of_line (x y : ℝ) :
  (3 * x + 2 * y - 1 = 0) → (2 * 3 + (-3) * 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_directional_vector_of_line_l2429_242980


namespace NUMINAMATH_CALUDE_head_start_is_90_meters_l2429_242987

/-- The head start distance in a race between Cristina and Nicky -/
def head_start_distance (cristina_speed nicky_speed : ℝ) (catch_up_time : ℝ) : ℝ :=
  nicky_speed * catch_up_time

/-- Theorem: Given Cristina's speed of 5 m/s, Nicky's speed of 3 m/s, 
    and a catch-up time of 30 seconds, the head start distance is 90 meters -/
theorem head_start_is_90_meters :
  head_start_distance 5 3 30 = 90 := by sorry

end NUMINAMATH_CALUDE_head_start_is_90_meters_l2429_242987


namespace NUMINAMATH_CALUDE_race_finish_distance_l2429_242951

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  distance : ℝ

/-- Represents the state of the race -/
structure Race where
  sasha : Runner
  lyosha : Runner
  kolya : Runner
  raceLength : ℝ

/-- The theorem to be proved -/
theorem race_finish_distance (r : Race) : 
  r.raceLength = 100 ∧ 
  r.sasha.distance - r.lyosha.distance = 10 ∧
  r.lyosha.distance - r.kolya.distance = 10 ∧
  r.sasha.distance = r.raceLength →
  r.sasha.distance - r.kolya.distance = 19 := by
  sorry

end NUMINAMATH_CALUDE_race_finish_distance_l2429_242951


namespace NUMINAMATH_CALUDE_complex_expression_1_complex_expression_2_l2429_242919

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the properties of i
axiom i_squared : i^2 = -1
axiom i_fourth : i^4 = 1

-- Theorem for the first expression
theorem complex_expression_1 :
  (4 - i^5) * (6 + 2*i^7) + (7 + i^11) * (4 - 3*i) = 57 - 39*i :=
by sorry

-- Theorem for the second expression
theorem complex_expression_2 :
  (5 * (4 + i)^2) / (i * (2 + i)) = -47 - 98*i :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_1_complex_expression_2_l2429_242919


namespace NUMINAMATH_CALUDE_rhombus_converse_and_inverse_false_l2429_242904

-- Define what it means for a polygon to be a rhombus
def is_rhombus (p : Polygon) : Prop := sorry

-- Define what it means for a polygon to have all sides of equal length
def has_equal_sides (p : Polygon) : Prop := sorry

-- Define a polygon (we don't need to specify its properties here)
def Polygon : Type := sorry

theorem rhombus_converse_and_inverse_false :
  (∃ p : Polygon, has_equal_sides p ∧ ¬is_rhombus p) ∧
  (∃ p : Polygon, ¬is_rhombus p ∧ has_equal_sides p) :=
sorry

end NUMINAMATH_CALUDE_rhombus_converse_and_inverse_false_l2429_242904


namespace NUMINAMATH_CALUDE_smallest_positive_angle_theorem_l2429_242944

theorem smallest_positive_angle_theorem (θ : Real) : 
  (θ > 0) → 
  (10 * Real.sin θ * (Real.cos θ)^3 - 10 * (Real.sin θ)^3 * Real.cos θ = Real.sqrt 2) →
  (∀ φ, φ > 0 → 10 * Real.sin φ * (Real.cos φ)^3 - 10 * (Real.sin φ)^3 * Real.cos φ = Real.sqrt 2 → θ ≤ φ) →
  θ = (1/4) * Real.arcsin ((2 * Real.sqrt 2) / 5) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_theorem_l2429_242944


namespace NUMINAMATH_CALUDE_tan_sum_of_roots_l2429_242949

theorem tan_sum_of_roots (α β : Real) : 
  (∃ x y : Real, x^2 + 6*x + 7 = 0 ∧ y^2 + 6*y + 7 = 0 ∧ x = Real.tan α ∧ y = Real.tan β) → 
  Real.tan (α + β) = 1 := by
sorry

end NUMINAMATH_CALUDE_tan_sum_of_roots_l2429_242949


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2429_242952

theorem arithmetic_calculation : 5 * 7 + 9 * 4 - 35 / 5 = 64 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2429_242952


namespace NUMINAMATH_CALUDE_platform_length_l2429_242998

/-- The length of a platform given train parameters -/
theorem platform_length (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  train_speed_kmph = 60 →
  crossing_time = 15 →
  ∃ (platform_length : ℝ), abs (platform_length - 130.05) < 0.01 :=
by
  sorry


end NUMINAMATH_CALUDE_platform_length_l2429_242998


namespace NUMINAMATH_CALUDE_pyramid_volume_l2429_242913

/-- Given a triangular pyramid SABC with base ABC being an equilateral triangle
    with side length a and edge SA = b, where the lateral faces are congruent,
    this theorem proves the possible volumes of the pyramid based on the
    relationship between a and b. -/
theorem pyramid_volume (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let V1 := a^2 / 12 * Real.sqrt (3 * b^2 - a^2)
  let V2 := a^2 * Real.sqrt 3 / 12 * Real.sqrt (b^2 - a^2)
  let V3 := a^2 * Real.sqrt 3 / 12 * Real.sqrt (b^2 - 3 * a^2)
  (a / Real.sqrt 3 < b ∧ b ≤ a → volume_pyramid = V1) ∧
  (a < b ∧ b ≤ a * Real.sqrt 3 → volume_pyramid = V1 ∨ volume_pyramid = V2) ∧
  (b > a * Real.sqrt 3 → volume_pyramid = V1 ∨ volume_pyramid = V2 ∨ volume_pyramid = V3) :=
by sorry

def volume_pyramid : ℝ := sorry

end NUMINAMATH_CALUDE_pyramid_volume_l2429_242913


namespace NUMINAMATH_CALUDE_x_gt_1_sufficient_not_necessary_for_abs_x_gt_1_l2429_242930

theorem x_gt_1_sufficient_not_necessary_for_abs_x_gt_1 :
  (∀ x : ℝ, x > 1 → |x| > 1) ∧
  (∃ x : ℝ, |x| > 1 ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_1_sufficient_not_necessary_for_abs_x_gt_1_l2429_242930


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2429_242968

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
    a = 90 → 
    b = 120 → 
    c^2 = a^2 + b^2 → 
    c = 150 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2429_242968


namespace NUMINAMATH_CALUDE_problem_solution_l2429_242906

theorem problem_solution : (1 / (Real.sqrt 2 + 1) - Real.sqrt 8 + (Real.sqrt 3 + 1) ^ 0) = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2429_242906


namespace NUMINAMATH_CALUDE_rectangle_unique_property_l2429_242962

-- Define the properties
def opposite_sides_equal (shape : Type) : Prop := sorry
def opposite_angles_equal (shape : Type) : Prop := sorry
def diagonals_equal (shape : Type) : Prop := sorry
def opposite_sides_parallel (shape : Type) : Prop := sorry

-- Define rectangles and parallelograms
class Rectangle (shape : Type) : Prop :=
  (opp_sides_eq : opposite_sides_equal shape)
  (opp_angles_eq : opposite_angles_equal shape)
  (diag_eq : diagonals_equal shape)
  (opp_sides_para : opposite_sides_parallel shape)

class Parallelogram (shape : Type) : Prop :=
  (opp_sides_eq : opposite_sides_equal shape)
  (opp_angles_eq : opposite_angles_equal shape)
  (opp_sides_para : opposite_sides_parallel shape)

-- Theorem statement
theorem rectangle_unique_property :
  ∀ (shape : Type),
    Rectangle shape →
    Parallelogram shape →
    (diagonals_equal shape ↔ Rectangle shape) ∧
    (¬(opposite_sides_equal shape ↔ Rectangle shape)) ∧
    (¬(opposite_angles_equal shape ↔ Rectangle shape)) ∧
    (¬(opposite_sides_parallel shape ↔ Rectangle shape)) :=
sorry

end NUMINAMATH_CALUDE_rectangle_unique_property_l2429_242962


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l2429_242901

/-- Given vectors a and b in R^2, prove that their difference has magnitude 1 -/
theorem vector_difference_magnitude (a b : ℝ × ℝ) :
  a.1 = Real.cos (15 * π / 180) ∧
  a.2 = Real.sin (15 * π / 180) ∧
  b.1 = Real.sin (15 * π / 180) ∧
  b.2 = Real.cos (15 * π / 180) →
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 1 := by
  sorry

#check vector_difference_magnitude

end NUMINAMATH_CALUDE_vector_difference_magnitude_l2429_242901


namespace NUMINAMATH_CALUDE_sequence_2011th_term_l2429_242911

def sequence_term (n : ℕ) : ℕ → ℕ
  | 0 => 52
  | (m + 1) => 
    let prev := sequence_term n m
    let last_digit := prev % 10
    let remaining := prev / 10
    last_digit ^ 2 + 2 * remaining

def is_cyclic (seq : ℕ → ℕ) (start : ℕ) (length : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ start → seq k = seq (k % length + start)

theorem sequence_2011th_term :
  ∃ (start length : ℕ),
    start > 0 ∧
    length > 0 ∧
    is_cyclic (sequence_term 0) start length ∧
    sequence_term 0 2010 = 18 := by
  sorry

end NUMINAMATH_CALUDE_sequence_2011th_term_l2429_242911


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l2429_242924

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- Definition of the first line -/
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 6 = 0

/-- Definition of the second line -/
def line2 (a : ℝ) (x y : ℝ) : Prop := x + (a - 1) * y + a^2 - 1 = 0

/-- The theorem to be proved -/
theorem parallel_lines_a_value :
  ∃ a : ℝ, (∀ x y : ℝ, line1 a x y ↔ line2 a x y) ∧ a = -1 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l2429_242924


namespace NUMINAMATH_CALUDE_neither_necessary_nor_sufficient_l2429_242970

open Real

/-- A function f is increasing on (0,∞) -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ x < y → f x < f y

theorem neither_necessary_nor_sufficient :
  ∃ (f₁ f₂ : ℝ → ℝ),
    (∀ x, x > 0 → f₁ x ≠ 0 ∧ f₂ x ≠ 0) ∧
    IsIncreasing f₁ ∧
    ¬IsIncreasing (fun x ↦ x * f₁ x) ∧
    ¬IsIncreasing f₂ ∧
    IsIncreasing (fun x ↦ x * f₂ x) :=
by sorry

end NUMINAMATH_CALUDE_neither_necessary_nor_sufficient_l2429_242970


namespace NUMINAMATH_CALUDE_problem_statement_l2429_242933

theorem problem_statement (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -7)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 8) :
  b / (a + b) + c / (b + c) + a / (c + a) = 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2429_242933


namespace NUMINAMATH_CALUDE_floor_sum_example_l2429_242979

theorem floor_sum_example : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_example_l2429_242979


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l2429_242929

theorem triangle_third_side_length 
  (a b x : ℕ) 
  (h1 : a = 2) 
  (h2 : b = 6) 
  (h3 : Even x) 
  (h4 : x > (b - a)) 
  (h5 : x < (b + a)) : 
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l2429_242929


namespace NUMINAMATH_CALUDE_hike_taxi_count_hike_taxi_count_is_six_l2429_242946

/-- Calculates the number of taxis required for a hike --/
theorem hike_taxi_count (total_people : ℕ) (car_count : ℕ) (van_count : ℕ) 
  (people_per_car : ℕ) (people_per_van : ℕ) (people_per_taxi : ℕ) : ℕ :=
  let people_in_cars := car_count * people_per_car
  let people_in_vans := van_count * people_per_van
  let people_in_taxis := total_people - (people_in_cars + people_in_vans)
  people_in_taxis / people_per_taxi

/-- Proves that 6 taxis were required for the hike --/
theorem hike_taxi_count_is_six : 
  hike_taxi_count 58 3 2 4 5 6 = 6 := by
  sorry

end NUMINAMATH_CALUDE_hike_taxi_count_hike_taxi_count_is_six_l2429_242946


namespace NUMINAMATH_CALUDE_line_through_points_l2429_242966

def line_equation (k m x : ℝ) : ℝ := k * x + m

theorem line_through_points (a b k : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : ∃ n : ℕ, b = n * a) :
  (∃ m : ℝ, line_equation k m a = a ∧ line_equation k m b = 8 * b) →
  k ∈ ({9, 15} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l2429_242966


namespace NUMINAMATH_CALUDE_probability_neither_red_nor_purple_l2429_242938

theorem probability_neither_red_nor_purple (total : ℕ) (red : ℕ) (purple : ℕ) 
  (h1 : total = 60) (h2 : red = 15) (h3 : purple = 3) : 
  (total - (red + purple)) / total = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_neither_red_nor_purple_l2429_242938


namespace NUMINAMATH_CALUDE_function_inequality_and_range_l2429_242985

/-- Given functions f and g, prove that if |f(x)| ≤ |g(x)| for all x, then f = g/2 - 4.
    Also prove that if f(x) ≥ (m + 2)x - m - 15 for all x > 2, then m ≤ 2. -/
theorem function_inequality_and_range (a b m : ℝ) : 
  let f := fun (x : ℝ) => x^2 + a*x + b
  let g := fun (x : ℝ) => 2*x^2 - 4*x - 16
  (∀ x, |f x| ≤ |g x|) →
  (a = -2 ∧ b = -8) ∧
  ((∀ x > 2, f x ≥ (m + 2)*x - m - 15) → m ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_and_range_l2429_242985


namespace NUMINAMATH_CALUDE_parallel_line_slope_parallel_line_slope_is_negative_four_thirds_l2429_242953

/-- The slope of a line parallel to the line containing the points (2, -3) and (-4, 5) is -4/3 -/
theorem parallel_line_slope : ℝ → ℝ → Prop :=
  fun x y =>
    let point1 : ℝ × ℝ := (2, -3)
    let point2 : ℝ × ℝ := (-4, 5)
    let slope : ℝ := (point2.2 - point1.2) / (point2.1 - point1.1)
    slope = -4/3

/-- The theorem statement -/
theorem parallel_line_slope_is_negative_four_thirds :
  ∃ (x y : ℝ), parallel_line_slope x y :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_line_slope_parallel_line_slope_is_negative_four_thirds_l2429_242953


namespace NUMINAMATH_CALUDE_no_torn_cards_l2429_242943

/-- The number of baseball cards Mary initially had -/
def initial_cards : ℕ := 18

/-- The number of baseball cards Fred gave to Mary -/
def fred_cards : ℕ := 26

/-- The number of baseball cards Mary bought -/
def bought_cards : ℕ := 40

/-- The total number of baseball cards Mary has now -/
def total_cards : ℕ := 84

/-- The number of torn baseball cards in Mary's initial collection -/
def torn_cards : ℕ := initial_cards - (total_cards - fred_cards - bought_cards)

theorem no_torn_cards : torn_cards = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_torn_cards_l2429_242943


namespace NUMINAMATH_CALUDE_benny_spent_amount_l2429_242993

/-- Represents the total amount spent in US dollars -/
def total_spent (initial_amount : ℝ) (remaining_amount : ℝ) : ℝ :=
  initial_amount - remaining_amount

/-- Theorem stating that given the initial amount of 200 US dollars and
    the remaining amount of 45 US dollars, the total amount spent is 155 US dollars -/
theorem benny_spent_amount :
  total_spent 200 45 = 155 := by sorry

end NUMINAMATH_CALUDE_benny_spent_amount_l2429_242993


namespace NUMINAMATH_CALUDE_chocolate_chip_cookies_l2429_242942

/-- The number of cookies in each bag -/
def cookies_per_bag : ℕ := 9

/-- The number of oatmeal cookies -/
def oatmeal_cookies : ℕ := 41

/-- The number of baggies that can be made with all cookies -/
def total_baggies : ℕ := 6

/-- The theorem stating the number of chocolate chip cookies -/
theorem chocolate_chip_cookies : 
  cookies_per_bag * total_baggies - oatmeal_cookies = 13 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_chip_cookies_l2429_242942


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2429_242925

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  (c = Real.sqrt 3 * a * Real.sin C - c * Real.cos A) →
  (a = 2) →
  (Real.sin (B - C) + Real.sin A = Real.sin (2 * C)) →
  (A = Real.pi / 3) ∧
  ((1/2 * a * b * Real.sin (Real.pi / 3) = 2 * Real.sqrt 3 / 3) ∨
   (1/2 * a * b * Real.sin (Real.pi / 3) = Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2429_242925


namespace NUMINAMATH_CALUDE_fruit_seller_loss_percentage_l2429_242994

/-- Calculates the percentage loss for a fruit seller given selling price, break-even price, and profit percentage. -/
def calculate_loss_percentage (selling_price profit_price profit_percentage : ℚ) : ℚ :=
  let cost_price := profit_price / (1 + profit_percentage / 100)
  let loss := cost_price - selling_price
  (loss / cost_price) * 100

/-- Theorem stating that under given conditions, the fruit seller's loss percentage is 15%. -/
theorem fruit_seller_loss_percentage :
  let selling_price : ℚ := 12
  let profit_price : ℚ := 14823529411764707 / 1000000000000000
  let profit_percentage : ℚ := 5
  calculate_loss_percentage selling_price profit_price profit_percentage = 15 := by
  sorry

#eval calculate_loss_percentage 12 (14823529411764707 / 1000000000000000) 5

end NUMINAMATH_CALUDE_fruit_seller_loss_percentage_l2429_242994


namespace NUMINAMATH_CALUDE_complement_of_A_relative_to_U_l2429_242941

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3, 4, 5}

theorem complement_of_A_relative_to_U :
  (U \ A) = {2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_relative_to_U_l2429_242941


namespace NUMINAMATH_CALUDE_x_squared_congruence_l2429_242958

theorem x_squared_congruence (x : ℤ) : 
  (5 * x ≡ 15 [ZMOD 25]) → (4 * x ≡ 20 [ZMOD 25]) → (x^2 ≡ 0 [ZMOD 25]) := by
  sorry

end NUMINAMATH_CALUDE_x_squared_congruence_l2429_242958


namespace NUMINAMATH_CALUDE_olivias_remaining_money_l2429_242973

def olivias_wallet (initial_amount : ℕ) (atm_amount : ℕ) (extra_spent : ℕ) : ℕ :=
  initial_amount + atm_amount - (atm_amount + extra_spent)

theorem olivias_remaining_money :
  olivias_wallet 53 91 39 = 14 :=
by sorry

end NUMINAMATH_CALUDE_olivias_remaining_money_l2429_242973


namespace NUMINAMATH_CALUDE_ellipse_equation_1_ellipse_equation_2_x_axis_ellipse_equation_2_y_axis_l2429_242950

-- Define the ellipse type
structure Ellipse where
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis
  c : ℝ  -- distance from center to focus
  e : ℝ  -- eccentricity

-- Define the standard equation of an ellipse
def standardEquation (E : Ellipse) (x y : ℝ) : Prop :=
  x^2 / E.a^2 + y^2 / E.b^2 = 1

-- Theorem for the first condition
theorem ellipse_equation_1 :
  ∀ E : Ellipse,
  E.c = 6 →
  E.e = 2/3 →
  (∀ x y : ℝ, standardEquation E x y ↔ y^2/81 + x^2/45 = 1) :=
sorry

-- Theorem for the second condition (foci on x-axis)
theorem ellipse_equation_2_x_axis :
  ∀ E : Ellipse,
  E.a = 5 →
  E.c = 3 →
  (∀ x y : ℝ, standardEquation E x y ↔ x^2/25 + y^2/16 = 1) :=
sorry

-- Theorem for the second condition (foci on y-axis)
theorem ellipse_equation_2_y_axis :
  ∀ E : Ellipse,
  E.a = 5 →
  E.c = 3 →
  (∀ x y : ℝ, standardEquation E y x ↔ y^2/25 + x^2/16 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_1_ellipse_equation_2_x_axis_ellipse_equation_2_y_axis_l2429_242950


namespace NUMINAMATH_CALUDE_symmetric_with_x_minus_y_factor_implies_squared_factor_l2429_242991

-- Define a symmetric polynomial
def is_symmetric (p : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, p x y = p y x

-- Define what it means for (x - y) to be a factor
def has_x_minus_y_factor (p : ℝ → ℝ → ℝ) : Prop :=
  ∃ q : ℝ → ℝ → ℝ, ∀ x y, p x y = (x - y) * q x y

-- Define what it means for (x - y)^2 to be a factor
def has_x_minus_y_squared_factor (p : ℝ → ℝ → ℝ) : Prop :=
  ∃ r : ℝ → ℝ → ℝ, ∀ x y, p x y = (x - y)^2 * r x y

-- The theorem to be proved
theorem symmetric_with_x_minus_y_factor_implies_squared_factor
  (p : ℝ → ℝ → ℝ)
  (h_sym : is_symmetric p)
  (h_factor : has_x_minus_y_factor p) :
  has_x_minus_y_squared_factor p :=
sorry

end NUMINAMATH_CALUDE_symmetric_with_x_minus_y_factor_implies_squared_factor_l2429_242991


namespace NUMINAMATH_CALUDE_set_operations_and_range_l2429_242983

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 < x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | 5 - a < x ∧ x < a}

-- Theorem statement
theorem set_operations_and_range :
  (A ∪ B = {x : ℝ | 2 < x ∧ x < 10}) ∧
  ((Set.univ \ A) ∩ B = {x : ℝ | 7 ≤ x ∧ x < 10}) ∧
  (∀ a : ℝ, C a ⊆ B → a ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_and_range_l2429_242983


namespace NUMINAMATH_CALUDE_eleven_students_like_sports_l2429_242917

/-- The number of students who like basketball or cricket or both -/
def students_basketball_or_cricket (basketball : ℕ) (cricket : ℕ) (both : ℕ) : ℕ :=
  basketball + cricket - both

/-- Theorem stating that given the conditions, 11 students like basketball or cricket or both -/
theorem eleven_students_like_sports : students_basketball_or_cricket 9 8 6 = 11 := by
  sorry

end NUMINAMATH_CALUDE_eleven_students_like_sports_l2429_242917


namespace NUMINAMATH_CALUDE_power_function_not_through_origin_l2429_242909

theorem power_function_not_through_origin (m : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → (m^2 - 3*m + 3) * x^(m^2 - m - 2) ≠ 0) →
  m = 1 ∨ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_not_through_origin_l2429_242909


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l2429_242939

theorem no_positive_integer_solution :
  ¬ ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ 3 * a^2 = b^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l2429_242939


namespace NUMINAMATH_CALUDE_concatenation_equation_solution_l2429_242965

theorem concatenation_equation_solution :
  ∃ x : ℕ, x + (10 * x + x) = 12 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_concatenation_equation_solution_l2429_242965


namespace NUMINAMATH_CALUDE_meal_combinations_count_l2429_242982

/-- The number of items on the menu -/
def menu_items : ℕ := 15

/-- The index of the restricted item -/
def restricted_item : ℕ := 10

/-- A function that calculates the number of valid meal combinations -/
def valid_combinations (n : ℕ) (r : ℕ) : ℕ :=
  n * n - 1

/-- Theorem stating that the number of valid meal combinations is 224 -/
theorem meal_combinations_count :
  valid_combinations menu_items restricted_item = 224 := by
  sorry

#eval valid_combinations menu_items restricted_item

end NUMINAMATH_CALUDE_meal_combinations_count_l2429_242982


namespace NUMINAMATH_CALUDE_count_valid_pairs_l2429_242975

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

def valid_pair (a b : ℕ) : Prop :=
  is_odd a ∧ is_odd b ∧ a > 1 ∧ b > 1 ∧ a * b = 315

theorem count_valid_pairs :
  ∃! (pairs : Finset (ℕ × ℕ)), 
    (∀ p ∈ pairs, valid_pair p.1 p.2) ∧ 
    pairs.card = 5 :=
sorry

end NUMINAMATH_CALUDE_count_valid_pairs_l2429_242975


namespace NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l2429_242995

/-- Given a quadratic polynomial P(x) = x^2 + px + q where P(q) < 0, 
    exactly one root of P(x) lies in the interval (0, 1) -/
theorem quadratic_root_in_unit_interval (p q : ℝ) :
  let P : ℝ → ℝ := λ x => x^2 + p*x + q
  (P q < 0) →
  ∃! x : ℝ, P x = 0 ∧ 0 < x ∧ x < 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l2429_242995


namespace NUMINAMATH_CALUDE_total_bills_and_coins_l2429_242954

/-- Represents the payment details for a grocery bill -/
structure GroceryPayment where
  totalBill : ℕ
  billValue : ℕ
  coinValue : ℕ
  numBills : ℕ
  numCoins : ℕ

/-- Theorem stating the total number of bills and coins used in the payment -/
theorem total_bills_and_coins (payment : GroceryPayment) 
  (h1 : payment.totalBill = 285)
  (h2 : payment.billValue = 20)
  (h3 : payment.coinValue = 5)
  (h4 : payment.numBills = 11)
  (h5 : payment.numCoins = 11)
  : payment.numBills + payment.numCoins = 22 := by
  sorry


end NUMINAMATH_CALUDE_total_bills_and_coins_l2429_242954


namespace NUMINAMATH_CALUDE_descending_order_l2429_242915

-- Define the numbers
def a : ℝ := 0.8
def b : ℝ := 0.878
def c : ℝ := 0.877
def d : ℝ := 0.87

-- Theorem statement
theorem descending_order : b > c ∧ c > d ∧ d > a := by sorry

end NUMINAMATH_CALUDE_descending_order_l2429_242915


namespace NUMINAMATH_CALUDE_markers_per_box_l2429_242977

theorem markers_per_box (total_students : ℕ) (boxes : ℕ) 
  (group1_students : ℕ) (group1_markers : ℕ)
  (group2_students : ℕ) (group2_markers : ℕ)
  (group3_markers : ℕ) :
  total_students = 30 →
  boxes = 22 →
  group1_students = 10 →
  group1_markers = 2 →
  group2_students = 15 →
  group2_markers = 4 →
  group3_markers = 6 →
  (boxes : ℚ) * ((group1_students * group1_markers + 
                  group2_students * group2_markers + 
                  (total_students - group1_students - group2_students) * group3_markers) / boxes : ℚ) = 
  (boxes : ℚ) * (5 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_markers_per_box_l2429_242977


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l2429_242923

theorem solve_exponential_equation :
  ∀ x : ℝ, (16 : ℝ)^x * (16 : ℝ)^x * (16 : ℝ)^x * (16 : ℝ)^x = (256 : ℝ)^(10 : ℝ) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l2429_242923


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2429_242903

/-- If k, -1, and b form an arithmetic sequence, then the line y = kx + b passes through (1, -2) -/
theorem line_passes_through_fixed_point (k b : ℝ) :
  (∃ d : ℝ, k = -1 - d ∧ b = -1 + d) →
  k * 1 + b = -2 :=
by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2429_242903


namespace NUMINAMATH_CALUDE_seating_arrangements_l2429_242908

theorem seating_arrangements (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 5) :
  (n.factorial / (n - k).factorial) = 720 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_l2429_242908


namespace NUMINAMATH_CALUDE_y_in_terms_of_x_l2429_242957

theorem y_in_terms_of_x (x y : ℚ) : x - 2 = 4 * (y - 1) + 3 → y = (1/4) * x - (1/4) := by
  sorry

end NUMINAMATH_CALUDE_y_in_terms_of_x_l2429_242957


namespace NUMINAMATH_CALUDE_find_z_l2429_242978

/-- A structure representing the relationship between x, y, and z. -/
structure Relationship where
  x : ℝ
  y : ℝ
  z : ℝ
  k : ℝ
  prop : y = k * x^2 / z

/-- The theorem statement -/
theorem find_z (r : Relationship) (h1 : r.y = 8) (h2 : r.x = 2) (h3 : r.z = 4)
    (h4 : r.x = 4) (h5 : r.y = 72) : r.z = 16/9 := by
  sorry


end NUMINAMATH_CALUDE_find_z_l2429_242978


namespace NUMINAMATH_CALUDE_candy_consumption_theorem_l2429_242948

/-- Represents the number of candies eaten by each person -/
structure CandyConsumption where
  andrey : ℕ
  boris : ℕ
  denis : ℕ

/-- Represents the relative eating rates of the three people -/
structure EatingRates where
  andrey_boris : ℚ  -- Ratio of Andrey's rate to Boris's rate
  andrey_denis : ℚ  -- Ratio of Andrey's rate to Denis's rate

theorem candy_consumption_theorem (rates : EatingRates) (total : ℕ) : 
  rates.andrey_boris = 4/3 → 
  rates.andrey_denis = 6/7 → 
  total = 70 → 
  ∃ (consumption : CandyConsumption), 
    consumption.andrey = 24 ∧ 
    consumption.boris = 18 ∧ 
    consumption.denis = 28 ∧
    consumption.andrey + consumption.boris + consumption.denis = total :=
by sorry

end NUMINAMATH_CALUDE_candy_consumption_theorem_l2429_242948


namespace NUMINAMATH_CALUDE_percentage_increase_l2429_242967

theorem percentage_increase (x : ℝ) (h : x = 114.4) : 
  (x - 88) / 88 * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l2429_242967


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2429_242981

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- Define the solution set of the first inequality
def S := Set.Ioo 1 2

-- Define the second inequality
def g (a b c x : ℝ) := a - c * (x^2 - x - 1) - b * x

-- Define the solution set of the second inequality
def T := {x : ℝ | x ≤ -3/2 ∨ x ≥ 1}

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) (h : ∀ x, x ∈ S ↔ f a b c x > 0) :
  ∀ x, x ∈ T ↔ g a b c x ≥ 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2429_242981


namespace NUMINAMATH_CALUDE_erica_money_proof_l2429_242927

def total_money : ℕ := 91
def sam_money : ℕ := 38

theorem erica_money_proof :
  total_money - sam_money = 53 := by
  sorry

end NUMINAMATH_CALUDE_erica_money_proof_l2429_242927


namespace NUMINAMATH_CALUDE_tangent_line_properties_l2429_242918

-- Define the curve
def f (x : ℝ) : ℝ := 4 * x^2 - 6 * x + 3

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 8 * x - 6

theorem tangent_line_properties :
  -- Part a: Tangent line parallel to y = 2x at (1, 1)
  (f' 1 = 2 ∧ f 1 = 1) ∧
  -- Part b: Tangent line perpendicular to y = x/4 at (1/4, 7/4)
  (f' (1/4) = -4 ∧ f (1/4) = 7/4) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_properties_l2429_242918


namespace NUMINAMATH_CALUDE_dance_circle_partition_l2429_242974

/-- The number of ways to partition n distinguishable objects into k indistinguishable,
    non-empty subsets, where rotations within subsets are considered identical. -/
def partition_count (n k : ℕ) : ℕ :=
  if k > n ∨ k = 0 then 0
  else
    (Finset.range (n - k + 1)).sum (λ i =>
      Nat.choose n (i + 1) * Nat.factorial i * Nat.factorial (n - i - 2))
    / 2

/-- Theorem stating that there are 50 ways to partition 5 children into 2 dance circles. -/
theorem dance_circle_partition :
  partition_count 5 2 = 50 := by
  sorry


end NUMINAMATH_CALUDE_dance_circle_partition_l2429_242974


namespace NUMINAMATH_CALUDE_journey_speed_journey_speed_theorem_l2429_242989

/-- 
Given a journey of 24 km completed in 8 hours, where the first 4 hours are
traveled at speed v km/hr and the last 4 hours at 2 km/hr, prove that v = 4.
-/
theorem journey_speed : ℝ → Prop :=
  fun v : ℝ =>
    (4 * v + 4 * 2 = 24) →
    v = 4

-- The proof is omitted
axiom journey_speed_proof : journey_speed 4

#check journey_speed_proof

-- Proof
theorem journey_speed_theorem : ∃ v : ℝ, journey_speed v := by
  sorry

end NUMINAMATH_CALUDE_journey_speed_journey_speed_theorem_l2429_242989


namespace NUMINAMATH_CALUDE_tetrahedron_volume_bound_l2429_242972

/-- A tetrahedron is represented by its six edge lengths -/
structure Tetrahedron where
  edge1 : ℝ
  edge2 : ℝ
  edge3 : ℝ
  edge4 : ℝ
  edge5 : ℝ
  edge6 : ℝ

/-- The volume of a tetrahedron -/
noncomputable def volume (t : Tetrahedron) : ℝ := sorry

/-- A tetrahedron with five edges not exceeding 1 -/
def FiveEdgesLimitedTetrahedron : Type :=
  { t : Tetrahedron // t.edge1 ≤ 1 ∧ t.edge2 ≤ 1 ∧ t.edge3 ≤ 1 ∧ t.edge4 ≤ 1 ∧ t.edge5 ≤ 1 }

theorem tetrahedron_volume_bound (t : FiveEdgesLimitedTetrahedron) :
  volume t.val ≤ 1/8 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_bound_l2429_242972


namespace NUMINAMATH_CALUDE_simplify_expression_l2429_242935

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) : 
  8 * x^3 * y / (2 * x)^2 = 2 * x * y :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2429_242935


namespace NUMINAMATH_CALUDE_cubic_two_roots_l2429_242921

/-- A cubic function with a parameter c -/
def f (c : ℝ) : ℝ → ℝ := fun x ↦ x^3 - 3*x + c

/-- The derivative of f -/
def f' (c : ℝ) : ℝ → ℝ := fun x ↦ 3*x^2 - 3

theorem cubic_two_roots (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f c x₁ = 0 ∧ f c x₂ = 0 ∧
    ∀ x, f c x = 0 → x = x₁ ∨ x = x₂) →
  c = 2 ∨ c = -2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_two_roots_l2429_242921


namespace NUMINAMATH_CALUDE_solve_cricket_problem_l2429_242964

def cricket_problem (W : ℝ) : Prop :=
  let crickets_90F : ℝ := 4
  let crickets_100F : ℝ := 2 * crickets_90F
  let prop_90F : ℝ := 0.8
  let prop_100F : ℝ := 1 - prop_90F
  let total_crickets : ℝ := 72
  W * (crickets_90F * prop_90F + crickets_100F * prop_100F) = total_crickets

theorem solve_cricket_problem :
  ∃ W : ℝ, cricket_problem W ∧ W = 15 := by
  sorry

end NUMINAMATH_CALUDE_solve_cricket_problem_l2429_242964


namespace NUMINAMATH_CALUDE_log_inequality_l2429_242997

theorem log_inequality (a b c : ℝ) 
  (ha : a = Real.log 3 / Real.log (1/2))
  (hb : b = Real.log 5 / Real.log (1/2))
  (hc : c = Real.log (1/2) / Real.log 3) :
  b < a ∧ a < c := by
sorry

end NUMINAMATH_CALUDE_log_inequality_l2429_242997


namespace NUMINAMATH_CALUDE_actual_quarterly_earnings_l2429_242996

/-- Calculates the actual quarterly earnings per share given the dividend paid for 400 shares -/
theorem actual_quarterly_earnings
  (expected_earnings : ℝ)
  (expected_dividend_ratio : ℝ)
  (additional_dividend_rate : ℝ)
  (additional_earnings_threshold : ℝ)
  (shares : ℕ)
  (total_dividend : ℝ)
  (h1 : expected_earnings = 0.80)
  (h2 : expected_dividend_ratio = 0.5)
  (h3 : additional_dividend_rate = 0.04)
  (h4 : additional_earnings_threshold = 0.10)
  (h5 : shares = 400)
  (h6 : total_dividend = 208) :
  ∃ (actual_earnings : ℝ), actual_earnings = 1.10 ∧
  total_dividend = shares * (expected_earnings * expected_dividend_ratio +
    (actual_earnings - expected_earnings) * (additional_dividend_rate / additional_earnings_threshold)) :=
by sorry

end NUMINAMATH_CALUDE_actual_quarterly_earnings_l2429_242996


namespace NUMINAMATH_CALUDE_jay_used_zero_l2429_242940

/-- Represents the amount of paint in a gallon -/
def gallon : ℚ := 1

/-- Represents the amount of paint Dexter used in gallons -/
def dexter_used : ℚ := 3/8

/-- Represents the amount of paint left in gallons -/
def paint_left : ℚ := 1

/-- Represents the amount of paint Jay used in gallons -/
def jay_used : ℚ := gallon - dexter_used - paint_left

theorem jay_used_zero : jay_used = 0 := by sorry

end NUMINAMATH_CALUDE_jay_used_zero_l2429_242940


namespace NUMINAMATH_CALUDE_aarti_work_completion_l2429_242992

/-- Given that Aarti can complete a piece of work in 9 days, 
    this theorem proves that she will complete 3 times the same work in 27 days. -/
theorem aarti_work_completion :
  ∀ (work : ℕ) (days : ℕ),
    days = 9 →  -- Aarti can complete the work in 9 days
    (27 : ℚ) / days = 3 -- The ratio of 27 days to the original work duration is 3
    :=
by
  sorry

end NUMINAMATH_CALUDE_aarti_work_completion_l2429_242992


namespace NUMINAMATH_CALUDE_rectangle_area_problem_l2429_242905

theorem rectangle_area_problem : ∃ (x y : ℝ), 
  (x + 3.5) * (y - 1.5) = x * y ∧ 
  (x - 3.5) * (y + 2) = x * y ∧ 
  x * y = 294 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_problem_l2429_242905


namespace NUMINAMATH_CALUDE_point_movement_to_y_axis_l2429_242916

/-- Given a point P that is moved 1 unit to the right to point M on the y-axis, 
    prove that M has coordinates (0, -2) -/
theorem point_movement_to_y_axis (m : ℝ) : 
  let P : ℝ × ℝ := (m + 2, 2 * m + 4)
  let M : ℝ × ℝ := (P.1 + 1, P.2)
  M.1 = 0 → M = (0, -2) := by
  sorry

end NUMINAMATH_CALUDE_point_movement_to_y_axis_l2429_242916


namespace NUMINAMATH_CALUDE_herd_size_l2429_242936

theorem herd_size (herd : ℕ) : 
  (1 / 3 : ℚ) * herd + (1 / 6 : ℚ) * herd + (1 / 7 : ℚ) * herd + 15 = herd →
  herd = 42 := by
sorry

end NUMINAMATH_CALUDE_herd_size_l2429_242936


namespace NUMINAMATH_CALUDE_power_of_product_l2429_242934

theorem power_of_product (x y : ℝ) : (x^2 * y)^3 = x^6 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l2429_242934


namespace NUMINAMATH_CALUDE_sheep_to_horse_ratio_l2429_242960

def daily_horse_food_per_horse : ℕ := 230
def total_daily_horse_food : ℕ := 12880
def number_of_sheep : ℕ := 56

theorem sheep_to_horse_ratio :
  (total_daily_horse_food / daily_horse_food_per_horse = number_of_sheep) →
  (number_of_sheep : ℚ) / (total_daily_horse_food / daily_horse_food_per_horse : ℚ) = 1 := by
sorry

end NUMINAMATH_CALUDE_sheep_to_horse_ratio_l2429_242960


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l2429_242910

theorem quadratic_roots_theorem (c : ℝ) :
  (∀ x : ℝ, x^2 - 3*x + c = 0 ↔ x = (3 + Real.sqrt c) / 2 ∨ x = (3 - Real.sqrt c) / 2) →
  c = 9/5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l2429_242910


namespace NUMINAMATH_CALUDE_spending_theorem_l2429_242963

-- Define the fraction of savings spent on the stereo
def stereo_fraction : ℚ := 1/4

-- Define the fraction less spent on the television compared to the stereo
def tv_fraction_less : ℚ := 2/3

-- Calculate the fraction spent on the television
def tv_fraction : ℚ := stereo_fraction - tv_fraction_less * stereo_fraction

-- Define the total fraction spent on both items
def total_fraction : ℚ := stereo_fraction + tv_fraction

-- Theorem statement
theorem spending_theorem : total_fraction = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_spending_theorem_l2429_242963


namespace NUMINAMATH_CALUDE_parallelogram_height_l2429_242931

/-- The height of a parallelogram with given base and area -/
theorem parallelogram_height (base area height : ℝ) 
  (h_base : base = 14)
  (h_area : area = 336)
  (h_formula : area = base * height) : height = 24 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l2429_242931


namespace NUMINAMATH_CALUDE_apple_orange_price_l2429_242928

theorem apple_orange_price (x y z : ℝ) 
  (eq1 : 24 * x = 28 * y)
  (eq2 : 45 * x + 60 * y = 1350 * z) :
  30 * x + 40 * y = 118.2857 * z :=
by sorry

end NUMINAMATH_CALUDE_apple_orange_price_l2429_242928


namespace NUMINAMATH_CALUDE_seventy_third_digit_is_zero_l2429_242932

/-- The number consisting of 112 ones -/
def number_of_ones : ℕ := 11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111

/-- The square of the number consisting of 112 ones -/
def square_of_ones : ℕ := number_of_ones * number_of_ones

/-- The seventy-third digit from the end of a natural number -/
def seventy_third_digit_from_end (n : ℕ) : ℕ :=
  (n / 10^72) % 10

theorem seventy_third_digit_is_zero :
  seventy_third_digit_from_end square_of_ones = 0 := by
  sorry

end NUMINAMATH_CALUDE_seventy_third_digit_is_zero_l2429_242932


namespace NUMINAMATH_CALUDE_marbles_combination_l2429_242961

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of marbles -/
def total_marbles : ℕ := 9

/-- The number of marbles to choose -/
def marbles_to_choose : ℕ := 4

/-- Theorem stating that choosing 4 marbles from 9 results in 126 ways -/
theorem marbles_combination :
  choose total_marbles marbles_to_choose = 126 := by
  sorry

end NUMINAMATH_CALUDE_marbles_combination_l2429_242961


namespace NUMINAMATH_CALUDE_seating_arrangements_with_restriction_l2429_242990

def number_of_people : ℕ := 10

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem seating_arrangements_with_restriction :
  let total_arrangements := factorial (number_of_people - 1) * 7
  total_arrangements = 2540160 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_with_restriction_l2429_242990


namespace NUMINAMATH_CALUDE_interview_score_is_85_l2429_242907

/-- Calculate the interview score based on individual scores and their proportions -/
def interview_score (basic_knowledge : ℝ) (communication_skills : ℝ) (work_attitude : ℝ) 
  (basic_prop : ℝ) (comm_prop : ℝ) (attitude_prop : ℝ) : ℝ :=
  basic_knowledge * basic_prop + communication_skills * comm_prop + work_attitude * attitude_prop

/-- Theorem: The interview score for the given scores and proportions is 85 points -/
theorem interview_score_is_85 :
  interview_score 85 80 88 0.2 0.3 0.5 = 85 := by
  sorry

#eval interview_score 85 80 88 0.2 0.3 0.5

end NUMINAMATH_CALUDE_interview_score_is_85_l2429_242907


namespace NUMINAMATH_CALUDE_regression_line_not_necessarily_through_sample_point_l2429_242969

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a linear regression model -/
structure LinearRegression where
  a : ℝ  -- intercept
  b : ℝ  -- slope

/-- Calculates the y-value for a given x using the linear regression model -/
def predict (model : LinearRegression) (x : ℝ) : ℝ :=
  model.a + model.b * x

/-- Checks if a point lies on the regression line -/
def pointOnLine (model : LinearRegression) (p : Point) : Prop :=
  p.y = predict model p.x

/-- Theorem: The linear regression line does not necessarily pass through any sample point -/
theorem regression_line_not_necessarily_through_sample_point :
  ∃ (model : LinearRegression) (samples : List Point),
    samples.length > 0 ∧ ∀ p ∈ samples, ¬(pointOnLine model p) :=
by sorry

end NUMINAMATH_CALUDE_regression_line_not_necessarily_through_sample_point_l2429_242969


namespace NUMINAMATH_CALUDE_max_angle_sum_l2429_242976

/-- Represents a quadrilateral ABCD with specific properties -/
structure Quadrilateral :=
  (x : ℝ)
  (d : ℝ)
  (angle_sum : x + (x + 2*d) + (x + d) = 180)
  (angle_progression : x ≤ x + d ∧ x + d ≤ x + 2*d)
  (similarity : x + d = 60)

/-- The maximum sum of the largest angles in triangles ABC and ACD is 180° -/
theorem max_angle_sum (q : Quadrilateral) :
  ∃ (max_sum : ℝ), max_sum = 180 ∧
  ∀ (sum : ℝ), sum = (q.x + 2*q.d) + (q.x + 2*q.d) → sum ≤ max_sum :=
sorry

end NUMINAMATH_CALUDE_max_angle_sum_l2429_242976
