import Mathlib

namespace starting_lineup_count_l3405_340552

def team_size : Nat := 16
def lineup_size : Nat := 5
def twin_count : Nat := 2
def triplet_count : Nat := 3

theorem starting_lineup_count : 
  (triplet_count * Nat.choose (team_size - twin_count - triplet_count + 2) (lineup_size - twin_count - 1)) = 198 := by
  sorry

end starting_lineup_count_l3405_340552


namespace school_ratio_problem_l3405_340528

theorem school_ratio_problem (total_students : ℕ) (boys_percentage : ℚ) 
  (represented_students : ℕ) (h1 : total_students = 140) 
  (h2 : boys_percentage = 1/2) (h3 : represented_students = 98) : 
  (represented_students : ℚ) / (boys_percentage * total_students) = 7/5 := by
  sorry

end school_ratio_problem_l3405_340528


namespace milk_can_problem_l3405_340593

theorem milk_can_problem :
  ∃! (x y : ℕ), 10 * x + 17 * y = 206 :=
by sorry

end milk_can_problem_l3405_340593


namespace plush_bear_distribution_l3405_340558

theorem plush_bear_distribution (total_bears : ℕ) (kindergarten_bears : ℕ) (num_classes : ℕ) :
  total_bears = 48 →
  kindergarten_bears = 15 →
  num_classes = 3 →
  (total_bears - kindergarten_bears) / num_classes = 11 :=
by sorry

end plush_bear_distribution_l3405_340558


namespace fifteenth_odd_multiple_of_5_fifteenth_odd_multiple_of_5_is_145_l3405_340530

/-- The nth positive odd multiple of 5 -/
def nthOddMultipleOf5 (n : ℕ) : ℕ := 10 * n - 5

theorem fifteenth_odd_multiple_of_5 : nthOddMultipleOf5 15 = 145 := by
  sorry

/-- The 15th positive integer that is both odd and a multiple of 5 -/
def fifteenthOddMultipleOf5 : ℕ := nthOddMultipleOf5 15

theorem fifteenth_odd_multiple_of_5_is_145 : fifteenthOddMultipleOf5 = 145 := by
  sorry

end fifteenth_odd_multiple_of_5_fifteenth_odd_multiple_of_5_is_145_l3405_340530


namespace geometry_test_passing_l3405_340531

theorem geometry_test_passing (total_problems : ℕ) (passing_percentage : ℚ) 
  (hp : passing_percentage = 85 / 100) (ht : total_problems = 50) : 
  ∃ (max_missed : ℕ), 
    (((total_problems - max_missed : ℚ) / total_problems) ≥ passing_percentage ∧ 
     ∀ (n : ℕ), n > max_missed → 
       ((total_problems - n : ℚ) / total_problems) < passing_percentage) ∧
    max_missed = 7 :=
sorry

end geometry_test_passing_l3405_340531


namespace no_twin_prime_legs_in_right_triangle_l3405_340548

theorem no_twin_prime_legs_in_right_triangle :
  ∀ (p k : ℕ), 
    Prime p → 
    Prime (p + 2) → 
    (∃ (h : ℕ), h * h = p * p + (p + 2) * (p + 2)) → 
    False :=
by
  sorry

end no_twin_prime_legs_in_right_triangle_l3405_340548


namespace rectangle_cut_perimeter_l3405_340564

/-- Given a rectangle with perimeter 10, prove that when cut twice parallel to its
    length and width to form 9 smaller rectangles, the total perimeter of these
    9 rectangles is 30. -/
theorem rectangle_cut_perimeter (a b : ℝ) : 
  (2 * (a + b) = 10) →  -- Perimeter of original rectangle
  (∃ x y z w : ℝ, 
    x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧  -- Cuts are positive
    x + y + z = a ∧ w + y + z = b) →  -- Cuts divide length and width
  (2 * (a + b) + 4 * (a + b) = 30) :=  -- Total perimeter after cuts
by sorry

end rectangle_cut_perimeter_l3405_340564


namespace cube_root_equation_solution_l3405_340525

theorem cube_root_equation_solution :
  ∃ y : ℝ, y = 1/32 ∧ (5 - 1/y)^(1/3 : ℝ) = -3 :=
sorry

end cube_root_equation_solution_l3405_340525


namespace remaining_money_l3405_340582

def initial_amount : ℕ := 43
def pencil_cost : ℕ := 20
def candy_cost : ℕ := 5

theorem remaining_money :
  initial_amount - (pencil_cost + candy_cost) = 18 := by sorry

end remaining_money_l3405_340582


namespace smallest_base_for_perfect_fourth_power_l3405_340534

theorem smallest_base_for_perfect_fourth_power : 
  (∃ (b : ℕ), b > 0 ∧ ∃ (x : ℕ), 7 * b^2 + 7 * b + 7 = x^4) ∧ 
  (∀ (b : ℕ), b > 0 → (∃ (x : ℕ), 7 * b^2 + 7 * b + 7 = x^4) → b ≥ 18) :=
by sorry

end smallest_base_for_perfect_fourth_power_l3405_340534


namespace equilateral_triangle_third_vertex_y_coord_l3405_340559

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- An equilateral triangle with two vertices given -/
structure EquilateralTriangle where
  v1 : Point
  v2 : Point
  third_in_first_quadrant : Bool

/-- The y-coordinate of the third vertex of an equilateral triangle -/
def third_vertex_y_coord (t : EquilateralTriangle) : ℝ :=
  sorry

theorem equilateral_triangle_third_vertex_y_coord 
  (t : EquilateralTriangle) 
  (h1 : t.v1 = ⟨1, 3⟩) 
  (h2 : t.v2 = ⟨9, 3⟩) 
  (h3 : t.third_in_first_quadrant = true) : 
  third_vertex_y_coord t = 3 + 4 * Real.sqrt 3 :=
sorry

end equilateral_triangle_third_vertex_y_coord_l3405_340559


namespace gcd_factorial_eight_and_factorial_six_squared_l3405_340599

theorem gcd_factorial_eight_and_factorial_six_squared : 
  Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 1440 := by
  sorry

end gcd_factorial_eight_and_factorial_six_squared_l3405_340599


namespace tan_product_from_cos_sum_diff_l3405_340504

theorem tan_product_from_cos_sum_diff (α β : ℝ) 
  (h1 : Real.cos (α + β) = 2/3) 
  (h2 : Real.cos (α - β) = 1/3) : 
  Real.tan α * Real.tan β = -1/3 := by
  sorry

end tan_product_from_cos_sum_diff_l3405_340504


namespace cube_difference_simplification_l3405_340533

theorem cube_difference_simplification (a b : ℝ) (ha_pos : a > 0) (hb_neg : b < 0)
  (ha_sq : a^2 = 9/25) (hb_sq : b^2 = (3 + Real.sqrt 2)^2 / 14) :
  (a - b)^3 = 88 * Real.sqrt 2 / 12750 := by
  sorry

end cube_difference_simplification_l3405_340533


namespace matthews_crackers_l3405_340570

/-- The number of friends Matthew has -/
def num_friends : ℕ := 4

/-- The number of cakes Matthew had initially -/
def initial_cakes : ℕ := 8

/-- The number of cakes each person ate -/
def cakes_eaten_per_person : ℕ := 2

/-- The number of crackers Matthew had initially -/
def initial_crackers : ℕ := 8

theorem matthews_crackers :
  initial_crackers = 8 :=
by
  sorry

end matthews_crackers_l3405_340570


namespace impossibility_of_tiling_l3405_340596

/-- Represents a checkerboard -/
structure Checkerboard :=
  (rows : ℕ)
  (cols : ℕ)
  (missing_corner : Bool)

/-- Represents a trimino -/
structure Trimino :=
  (length : ℕ)
  (width : ℕ)

/-- Determines if a checkerboard can be tiled with triminos -/
def can_tile (board : Checkerboard) (tile : Trimino) : Prop :=
  ∃ (tiling : ℕ), 
    (board.rows * board.cols - if board.missing_corner then 1 else 0) = 
    tiling * (tile.length * tile.width)

theorem impossibility_of_tiling (board : Checkerboard) (tile : Trimino) : 
  (board.rows = 8 ∧ board.cols = 8 ∧ tile.length = 3 ∧ tile.width = 1) →
  (¬ can_tile board tile) ∧ 
  (¬ can_tile {rows := board.rows, cols := board.cols, missing_corner := true} tile) :=
sorry

end impossibility_of_tiling_l3405_340596


namespace candy_distribution_l3405_340519

theorem candy_distribution (num_students : ℕ) (pieces_per_student : ℕ) 
  (h1 : num_students = 43)
  (h2 : pieces_per_student = 8) :
  num_students * pieces_per_student = 344 := by
  sorry

end candy_distribution_l3405_340519


namespace sequence_converges_to_ones_l3405_340532

/-- The operation S applied to a sequence -/
def S (a : Fin (2^n) → Int) : Fin (2^n) → Int :=
  fun i => a i * a (i.succ)

/-- The result of applying S v times -/
def applyS (a : Fin (2^n) → Int) (v : Nat) : Fin (2^n) → Int :=
  match v with
  | 0 => a
  | v+1 => S (applyS a v)

theorem sequence_converges_to_ones 
  (n : Nat) (a : Fin (2^n) → Int) 
  (h : ∀ i, a i = 1 ∨ a i = -1) : 
  ∀ i, applyS a (2^n) i = 1 := by
  sorry

#check sequence_converges_to_ones

end sequence_converges_to_ones_l3405_340532


namespace reciprocal_of_repeating_three_l3405_340520

def repeating_decimal_to_fraction (d : ℚ) : ℚ :=
  if d ≥ 0 ∧ d < 1 then d / (1 - (10 * d - ⌊10 * d⌋)) else d

theorem reciprocal_of_repeating_three : 
  (repeating_decimal_to_fraction (1/3 : ℚ))⁻¹ = 3 := by
sorry

end reciprocal_of_repeating_three_l3405_340520


namespace unique_solution_for_star_equation_l3405_340527

-- Define the ⋆ operation
def star (x y : ℝ) : ℝ := 5*x - 4*y + 2*x*y

-- State the theorem
theorem unique_solution_for_star_equation :
  ∃! y : ℝ, star 4 y = 16 := by sorry

end unique_solution_for_star_equation_l3405_340527


namespace milk_students_l3405_340551

theorem milk_students (juice_students : ℕ) (juice_angle : ℝ) (total_angle : ℝ) :
  juice_students = 80 →
  juice_angle = 90 →
  total_angle = 360 →
  (juice_angle / total_angle) * (juice_students + (total_angle - juice_angle) / juice_angle * juice_students) = 240 :=
by sorry

end milk_students_l3405_340551


namespace cookie_sales_difference_l3405_340576

/-- The number of cookie boxes sold by Kim -/
def kim_boxes : ℕ := 54

/-- The number of cookie boxes sold by Jennifer -/
def jennifer_boxes : ℕ := 71

/-- Theorem stating the difference in cookie sales between Jennifer and Kim -/
theorem cookie_sales_difference :
  jennifer_boxes > kim_boxes ∧
  jennifer_boxes - kim_boxes = 17 :=
sorry

end cookie_sales_difference_l3405_340576


namespace divisibility_of_sixth_power_difference_l3405_340510

theorem divisibility_of_sixth_power_difference (a b : ℤ) 
  (ha : ¬ 3 ∣ a) (hb : ¬ 3 ∣ b) : 
  9 ∣ (a^6 - b^6) :=
by sorry

end divisibility_of_sixth_power_difference_l3405_340510


namespace number_manipulation_l3405_340507

theorem number_manipulation (x : ℝ) : (x - 5) / 7 = 7 → (x - 14) / 10 = 4 := by
  sorry

end number_manipulation_l3405_340507


namespace correct_yeast_experiment_methods_l3405_340502

/-- Represents the method used for counting yeast -/
inductive CountingMethod
| SamplingInspection
| Other

/-- Represents the action taken before extracting culture fluid -/
inductive PreExtractionAction
| GentlyShake
| Other

/-- Represents the measure taken when there are too many yeast cells -/
inductive OvercrowdingMeasure
| AppropriateDilution
| Other

/-- Represents the conditions of the yeast counting experiment -/
structure YeastExperiment where
  countingMethod : CountingMethod
  preExtractionAction : PreExtractionAction
  overcrowdingMeasure : OvercrowdingMeasure

/-- Theorem stating the correct methods and actions for the yeast counting experiment -/
theorem correct_yeast_experiment_methods :
  ∀ (experiment : YeastExperiment),
    experiment.countingMethod = CountingMethod.SamplingInspection ∧
    experiment.preExtractionAction = PreExtractionAction.GentlyShake ∧
    experiment.overcrowdingMeasure = OvercrowdingMeasure.AppropriateDilution :=
by sorry

end correct_yeast_experiment_methods_l3405_340502


namespace steve_gum_pieces_l3405_340526

theorem steve_gum_pieces (initial_gum : ℕ) (total_gum : ℕ) (h1 : initial_gum = 38) (h2 : total_gum = 54) :
  total_gum - initial_gum = 16 := by
  sorry

end steve_gum_pieces_l3405_340526


namespace quadratic_trinomial_square_l3405_340579

theorem quadratic_trinomial_square (a b c : ℝ) :
  (∀ x : ℤ, ∃ y : ℤ, a * x^2 + b * x + c = y^2) →
  (∃ m n k : ℤ, 2 * a = m ∧ 2 * b = n ∧ c = k^2) ∧
  (∃ p q r : ℤ, a = p ∧ b = q ∧ c = r^2) :=
by sorry

end quadratic_trinomial_square_l3405_340579


namespace second_term_is_five_l3405_340549

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℝ) (d : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => arithmetic_sequence a d n + d

-- Theorem statement
theorem second_term_is_five
  (a d : ℝ)
  (h : arithmetic_sequence a d 0 + arithmetic_sequence a d 2 = 10) :
  arithmetic_sequence a d 1 = 5 :=
by
  sorry

end second_term_is_five_l3405_340549


namespace king_arthur_table_seats_l3405_340508

/-- Represents a circular seating arrangement -/
structure CircularArrangement where
  size : ℕ
  opposite : ℕ → ℕ
  opposite_symmetric : ∀ n, n ≤ size → opposite (opposite n) = n

/-- The specific circular arrangement described in the problem -/
def kingArthurTable : CircularArrangement where
  size := 38
  opposite := fun n => (n + 19) % 38
  opposite_symmetric := sorry

theorem king_arthur_table_seats :
  ∃ (t : CircularArrangement), t.size = 38 ∧ t.opposite 10 = 29 := by
  use kingArthurTable
  constructor
  · rfl
  · rfl

#check king_arthur_table_seats

end king_arthur_table_seats_l3405_340508


namespace student_average_age_l3405_340572

theorem student_average_age 
  (num_students : ℕ) 
  (teacher_age : ℕ) 
  (total_average : ℕ) 
  (h1 : num_students = 30)
  (h2 : teacher_age = 46)
  (h3 : total_average = 16)
  : (((num_students + 1) * total_average - teacher_age) / num_students : ℚ) = 15 := by
  sorry

end student_average_age_l3405_340572


namespace product_minus_sum_probability_l3405_340598

def valid_pair (a b : ℕ) : Prop :=
  a ≤ 10 ∧ b ≤ 10 ∧ a * b - (a + b) > 4

def total_pairs : ℕ := 100

def valid_pairs : ℕ := 44

theorem product_minus_sum_probability :
  (valid_pairs : ℚ) / total_pairs = 11 / 25 := by sorry

end product_minus_sum_probability_l3405_340598


namespace absolute_value_inequality_solution_set_l3405_340562

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2 - x| ≥ 1} = {x : ℝ | x ≤ 1 ∨ x ≥ 3} := by sorry

end absolute_value_inequality_solution_set_l3405_340562


namespace arithmetic_equation_l3405_340587

theorem arithmetic_equation : 
  (5 / 6 : ℚ) - (-2 : ℚ) + (1 + 1 / 6 : ℚ) = 4 := by
  sorry

end arithmetic_equation_l3405_340587


namespace first_number_solution_l3405_340567

theorem first_number_solution (y : ℝ) (h : y = -4.5) :
  ∃ x : ℝ, x * y = 2 * x - 36 → x = 36 / 6.5 := by
  sorry

end first_number_solution_l3405_340567


namespace division_of_terms_l3405_340565

theorem division_of_terms (a b : ℝ) (h : b ≠ 0) : 3 * a^2 * b / b = 3 * a^2 := by
  sorry

end division_of_terms_l3405_340565


namespace arithmetic_sequence_count_100_l3405_340584

/-- The number of ways to select 3 different numbers from 1 to 100 
    that form an arithmetic sequence in their original order -/
def arithmeticSequenceCount : ℕ := 2450

/-- A function that counts the number of arithmetic sequences of length 3
    that can be formed from numbers 1 to n -/
def countArithmeticSequences (n : ℕ) : ℕ :=
  sorry

theorem arithmetic_sequence_count_100 : 
  countArithmeticSequences 100 = arithmeticSequenceCount := by sorry

end arithmetic_sequence_count_100_l3405_340584


namespace angle_properties_l3405_340571

theorem angle_properties (a θ : ℝ) (h : a > 0) 
  (h_point : ∃ (x y : ℝ), x = 3 * a ∧ y = 4 * a ∧ (Real.cos θ = x / Real.sqrt (x^2 + y^2)) ∧ (Real.sin θ = y / Real.sqrt (x^2 + y^2))) :
  (Real.sin θ = 4/5) ∧ 
  (Real.sin (3 * Real.pi / 2 - θ) + Real.cos (θ - Real.pi) = -6/5) := by
  sorry


end angle_properties_l3405_340571


namespace distinguishable_triangles_count_l3405_340563

/-- Represents the number of available colors for the triangles -/
def num_colors : ℕ := 8

/-- Represents the number of smaller triangles used to construct a large triangle -/
def triangles_per_large : ℕ := 4

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- Calculates the number of distinguishable large triangles -/
def num_distinguishable_triangles : ℕ :=
  let corner_same := num_colors -- All corners same color
  let corner_two_same := num_colors * (num_colors - 1) -- Two corners same, one different
  let corner_all_diff := choose num_colors 3 -- All corners different
  let total_corners := corner_same + corner_two_same + corner_all_diff
  total_corners * num_colors -- Multiply by choices for center triangle

theorem distinguishable_triangles_count :
  num_distinguishable_triangles = 960 :=
sorry

end distinguishable_triangles_count_l3405_340563


namespace fraction_to_decimal_l3405_340578

theorem fraction_to_decimal (h : 243 = 3^5) : 7 / 243 = 0.00224 := by
  sorry

end fraction_to_decimal_l3405_340578


namespace hyperbola_equation_l3405_340589

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∃ x y : ℝ, x^2/a^2 - y^2/b^2 = 1) →
  ((a - 4)^2 + b^2 = 16) →
  (a^2 + b^2 = 16) →
  (x^2/4 - y^2/12 = 1) := by
sorry

end hyperbola_equation_l3405_340589


namespace female_officers_count_l3405_340523

theorem female_officers_count (total_on_duty : ℕ) (female_duty_percent : ℚ) 
  (h1 : total_on_duty = 160)
  (h2 : female_duty_percent = 16 / 100) : 
  ∃ (total_female : ℕ), total_female = 1000 ∧ 
    (female_duty_percent * total_female : ℚ) = total_on_duty := by
  sorry

end female_officers_count_l3405_340523


namespace distance_to_pole_l3405_340516

def polar_distance (ρ : ℝ) (θ : ℝ) : ℝ := ρ

theorem distance_to_pole (A : ℝ × ℝ) (h : A = (3, -4)) :
  polar_distance A.1 A.2 = 3 := by
  sorry

end distance_to_pole_l3405_340516


namespace largest_common_divisor_l3405_340566

theorem largest_common_divisor :
  ∃ (n : ℕ), n = 30 ∧
  n ∣ 420 ∧
  n < 60 ∧
  n ∣ 90 ∧
  ∀ (m : ℕ), m ∣ 420 → m < 60 → m ∣ 90 → m ≤ n :=
by sorry

end largest_common_divisor_l3405_340566


namespace emily_candy_consumption_l3405_340529

/-- Emily's Halloween candy problem -/
theorem emily_candy_consumption (neighbor_candy : ℕ) (sister_candy : ℕ) (days : ℕ) 
  (h1 : neighbor_candy = 5)
  (h2 : sister_candy = 13)
  (h3 : days = 2) :
  (neighbor_candy + sister_candy) / days = 9 := by
  sorry

end emily_candy_consumption_l3405_340529


namespace arc_length_150_degrees_l3405_340547

/-- The arc length of a circle with radius 1 cm and central angle 150° is (5π/6) cm. -/
theorem arc_length_150_degrees : 
  let radius : ℝ := 1
  let central_angle_degrees : ℝ := 150
  let central_angle_radians : ℝ := central_angle_degrees * (π / 180)
  let arc_length : ℝ := radius * central_angle_radians
  arc_length = (5 * π) / 6 := by sorry

end arc_length_150_degrees_l3405_340547


namespace luncheon_cost_is_105_l3405_340560

/-- The cost of a luncheon consisting of one sandwich, one cup of coffee, and one piece of pie -/
def luncheon_cost (s c p : ℚ) : ℚ := s + c + p

/-- The cost of the first luncheon combination -/
def first_combination (s c p : ℚ) : ℚ := 3 * s + 7 * c + p

/-- The cost of the second luncheon combination -/
def second_combination (s c p : ℚ) : ℚ := 4 * s + 10 * c + p

theorem luncheon_cost_is_105 
  (s c p : ℚ) 
  (h1 : first_combination s c p = 315/100) 
  (h2 : second_combination s c p = 420/100) : 
  luncheon_cost s c p = 105/100 := by
  sorry

end luncheon_cost_is_105_l3405_340560


namespace interior_triangle_area_l3405_340513

theorem interior_triangle_area (a b c : ℝ) (ha : a^2 = 49) (hb : b^2 = 64) (hc : c^2 = 225) :
  (1/2 : ℝ) * a * b = 28 :=
sorry

end interior_triangle_area_l3405_340513


namespace house_price_ratio_l3405_340568

def total_price : ℕ := 600000
def first_house_price : ℕ := 200000

theorem house_price_ratio :
  (total_price - first_house_price) / first_house_price = 2 :=
by sorry

end house_price_ratio_l3405_340568


namespace rectangular_playground_vertical_length_l3405_340540

/-- The vertical length of a rectangular playground given specific conditions -/
theorem rectangular_playground_vertical_length :
  ∀ (square_side : ℝ) (rect_horizontal : ℝ) (rect_vertical : ℝ),
    square_side = 12 →
    rect_horizontal = 9 →
    4 * square_side = 2 * (rect_horizontal + rect_vertical) →
    rect_vertical = 15 :=
by sorry

end rectangular_playground_vertical_length_l3405_340540


namespace abies_chips_l3405_340546

theorem abies_chips (initial_bags : ℕ) (bought_bags : ℕ) (final_bags : ℕ) 
  (h1 : initial_bags = 20)
  (h2 : bought_bags = 6)
  (h3 : final_bags = 22) :
  initial_bags - (initial_bags - final_bags + bought_bags) = 4 :=
by sorry

end abies_chips_l3405_340546


namespace parabola_p_value_l3405_340592

/-- A parabola with equation y^2 = 2px and directrix x = -2 has p = 4 -/
theorem parabola_p_value (y x p : ℝ) : 
  (∀ y x, y^2 = 2*p*x) →  -- Condition 1: Parabola equation
  (x = -2)               -- Condition 2: Directrix equation
  → p = 4 :=             -- Conclusion: p = 4
by sorry

end parabola_p_value_l3405_340592


namespace sqrt_3_plus_2_times_sqrt_3_minus_2_l3405_340556

theorem sqrt_3_plus_2_times_sqrt_3_minus_2 : (Real.sqrt 3 + 2) * (Real.sqrt 3 - 2) = -1 := by
  sorry

end sqrt_3_plus_2_times_sqrt_3_minus_2_l3405_340556


namespace club_size_l3405_340545

/-- A club with committees satisfying specific conditions -/
structure Club where
  /-- The number of committees in the club -/
  num_committees : Nat
  /-- The number of members in the club -/
  num_members : Nat
  /-- Each member belongs to exactly two committees -/
  member_in_two_committees : True
  /-- Each pair of committees has exactly one member in common -/
  one_common_member : True

/-- Theorem stating that a club with 4 committees satisfying the given conditions has 6 members -/
theorem club_size (c : Club) : c.num_committees = 4 → c.num_members = 6 := by
  sorry

end club_size_l3405_340545


namespace intersection_A_B_l3405_340537

def A : Set ℝ := {-1, 1, 2, 4}
def B : Set ℝ := {x | |x - 1| ≤ 1}

theorem intersection_A_B : A ∩ B = {1, 2} := by sorry

end intersection_A_B_l3405_340537


namespace apps_difference_is_three_l3405_340524

/-- The difference between apps added and deleted -/
def appsDifference (initial final added : ℕ) : ℕ :=
  added - (initial + added - final)

/-- Proof that the difference between apps added and deleted is 3 -/
theorem apps_difference_is_three : appsDifference 21 24 89 = 3 := by
  sorry

end apps_difference_is_three_l3405_340524


namespace lemonade_water_requirement_l3405_340517

/-- The amount of water required for lemonade recipe -/
def water_required (water_parts : ℚ) (lemon_juice_parts : ℚ) (total_gallons : ℚ) (quarts_per_gallon : ℚ) (cups_per_quart : ℚ) : ℚ :=
  (water_parts / (water_parts + lemon_juice_parts)) * total_gallons * quarts_per_gallon * cups_per_quart

/-- Theorem stating the required amount of water for the lemonade recipe -/
theorem lemonade_water_requirement : 
  water_required 5 2 (3/2) 4 4 = 120/7 := by
  sorry

end lemonade_water_requirement_l3405_340517


namespace symmetric_point_and_line_l3405_340509

-- Define the line l₁
def l₁ (x y : ℝ) : Prop := 2 * x - 3 * y + 1 = 0

-- Define point A
def A : ℝ × ℝ := (-1, -2)

-- Define line m
def m (x y : ℝ) : Prop := 3 * x - 2 * y - 6 = 0

-- Define the symmetric point of a given point with respect to l₁
def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the symmetric line of a given line with respect to l₁
def symmetric_line (l : (ℝ → ℝ → Prop)) : (ℝ → ℝ → Prop) := sorry

theorem symmetric_point_and_line :
  (symmetric_point A = (-33/13, 4/13)) ∧
  (∀ x y, symmetric_line m x y ↔ 3 * x - 11 * y + 34 = 0) :=
sorry

end symmetric_point_and_line_l3405_340509


namespace min_value_of_f_l3405_340569

/-- The function to be minimized -/
def f (x y : ℝ) : ℝ := 2 * x^2 + 3 * x * y + 4 * y^2 - 8 * x - 6 * y

/-- The theorem stating that the minimum value of f is -14 -/
theorem min_value_of_f :
  ∃ (min : ℝ), min = -14 ∧ ∀ (x y : ℝ), f x y ≥ min :=
by sorry

end min_value_of_f_l3405_340569


namespace pens_left_after_giving_away_l3405_340550

/-- Given that a student's parents bought her 56 pens and she gave 22 pens to her friends,
    prove that the number of pens left for her to use is 34. -/
theorem pens_left_after_giving_away (total_pens : ℕ) (pens_given_away : ℕ) :
  total_pens = 56 → pens_given_away = 22 → total_pens - pens_given_away = 34 := by
  sorry

end pens_left_after_giving_away_l3405_340550


namespace arithmetic_sequence_75th_term_l3405_340583

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_75th_term
  (a : ℕ → ℚ)
  (h_arith : arithmetic_sequence a)
  (h_15 : a 15 = 8)
  (h_60 : a 60 = 20) :
  a 75 = 24 := by
  sorry

end arithmetic_sequence_75th_term_l3405_340583


namespace fixed_point_of_exponential_function_l3405_340500

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 4 + a^(x - 1)
  f 1 = 5 := by sorry

end fixed_point_of_exponential_function_l3405_340500


namespace auston_taller_than_emma_l3405_340591

def inch_to_cm (inches : ℝ) : ℝ := inches * 2.54

def height_difference_cm (auston_height_inch : ℝ) (emma_height_inch : ℝ) : ℝ :=
  inch_to_cm auston_height_inch - inch_to_cm emma_height_inch

theorem auston_taller_than_emma : 
  height_difference_cm 60 54 = 15.24 := by sorry

end auston_taller_than_emma_l3405_340591


namespace first_player_wins_l3405_340580

/-- Represents a chessboard with knights on opposite corners -/
structure Chessboard :=
  (squares : Finset (ℕ × ℕ))
  (knight1 : ℕ × ℕ)
  (knight2 : ℕ × ℕ)

/-- Represents a move in the game -/
def Move := ℕ × ℕ

/-- Checks if a knight can reach another position on the board -/
def can_reach (board : Chessboard) (start finish : ℕ × ℕ) : Prop :=
  sorry

/-- Represents the game state -/
structure GameState :=
  (board : Chessboard)
  (current_player : ℕ)

/-- Applies a move to the game state -/
def apply_move (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if a player has a winning strategy -/
def has_winning_strategy (player : ℕ) (state : GameState) : Prop :=
  sorry

/-- The main theorem stating that the first player has a winning strategy -/
theorem first_player_wins (initial_board : Chessboard) :
  has_winning_strategy 1 { board := initial_board, current_player := 1 } :=
sorry

end first_player_wins_l3405_340580


namespace blocks_differing_in_two_ways_l3405_340538

/-- Represents the properties of a block -/
structure Block where
  material : Fin 3
  size : Fin 3
  color : Fin 4
  shape : Fin 5

/-- The set of all blocks -/
def all_blocks : Finset Block := sorry

/-- The reference block (wood small blue hexagon) -/
def reference_block : Block := ⟨0, 0, 0, 1⟩

/-- The number of ways a block differs from the reference block -/
def diff_count (b : Block) : Nat := sorry

/-- Theorem: The number of blocks differing in exactly 2 ways from the reference block is 44 -/
theorem blocks_differing_in_two_ways :
  (all_blocks.filter (λ b => diff_count b = 2)).card = 44 := by sorry

end blocks_differing_in_two_ways_l3405_340538


namespace painted_cubes_4x4x4_l3405_340586

/-- The number of unit cubes with at least one face painted in a 4x4x4 cube -/
def painted_cubes (n : Nat) : Nat :=
  n^3 - (n - 2)^3

/-- The proposition that the number of painted cubes in a 4x4x4 cube is 41 -/
theorem painted_cubes_4x4x4 :
  painted_cubes 4 = 41 := by
  sorry

end painted_cubes_4x4x4_l3405_340586


namespace square_side_length_l3405_340518

theorem square_side_length (P A : ℝ) (h1 : P = 12) (h2 : A = 9) : ∃ s : ℝ, s > 0 ∧ P = 4 * s ∧ A = s ^ 2 ∧ s = 3 := by
  sorry

end square_side_length_l3405_340518


namespace balls_sold_l3405_340512

theorem balls_sold (selling_price : ℕ) (cost_price : ℕ) (loss : ℕ) : 
  selling_price = 720 → 
  cost_price = 60 → 
  loss = 5 * cost_price → 
  ∃ n : ℕ, n * cost_price - selling_price = loss ∧ n = 17 :=
by sorry

end balls_sold_l3405_340512


namespace intersection_x_coordinate_l3405_340511

-- Define the equations of the two lines
def line1 (x y : ℝ) : Prop := y = 3 * x - 17
def line2 (x y : ℝ) : Prop := 3 * x + y = 103

-- Theorem stating that the x-coordinate of the intersection is 20
theorem intersection_x_coordinate :
  ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ x = 20 := by
  sorry

end intersection_x_coordinate_l3405_340511


namespace paper_fold_distance_l3405_340543

theorem paper_fold_distance (area : ℝ) (h_area : area = 18) : ∃ (distance : ℝ), distance = 6 := by
  sorry

end paper_fold_distance_l3405_340543


namespace max_a_for_inequality_l3405_340539

theorem max_a_for_inequality : ∃ (a : ℝ), ∀ (x : ℝ), |x - 2| + |x - 8| ≥ a ∧ ∀ (b : ℝ), (∀ (y : ℝ), |y - 2| + |y - 8| ≥ b) → b ≤ a :=
by sorry

end max_a_for_inequality_l3405_340539


namespace equation_solution_l3405_340501

theorem equation_solution (x : ℝ) : (2*x - 1)^2 = 81 → x = 5 ∨ x = -4 := by
  sorry

end equation_solution_l3405_340501


namespace starting_number_proof_l3405_340514

def has_two_fives (n : ℕ) : Prop :=
  (n / 10 = 5 ∧ n % 10 = 5) ∨ (n / 100 = 5 ∧ n % 100 / 10 = 5)

theorem starting_number_proof :
  ∀ (start : ℕ),
    start ≤ 54 →
    (∃! n : ℕ, start ≤ n ∧ n ≤ 50 ∧ has_two_fives n) →
    start = 54 :=
by sorry

end starting_number_proof_l3405_340514


namespace max_puzzles_in_club_l3405_340522

/-- Represents a math club with members solving puzzles -/
structure MathClub where
  members : ℕ
  average_puzzles : ℕ
  min_puzzles : ℕ

/-- Calculates the maximum number of puzzles one member can solve -/
def max_puzzles_by_one (club : MathClub) : ℕ :=
  club.members * club.average_puzzles - (club.members - 1) * club.min_puzzles

/-- Theorem stating the maximum number of puzzles solved by one member in the given conditions -/
theorem max_puzzles_in_club (club : MathClub) 
  (h_members : club.members = 40)
  (h_average : club.average_puzzles = 6)
  (h_min : club.min_puzzles = 2) :
  max_puzzles_by_one club = 162 := by
  sorry

#eval max_puzzles_by_one ⟨40, 6, 2⟩

end max_puzzles_in_club_l3405_340522


namespace additional_rate_calculation_l3405_340535

/-- Telephone company charging model -/
structure TelephoneCharge where
  initial_rate : ℚ  -- Rate for the first 1/5 minute in cents
  additional_rate : ℚ  -- Rate for each additional 1/5 minute in cents

/-- Calculate the total charge for a given duration -/
def total_charge (model : TelephoneCharge) (duration : ℚ) : ℚ :=
  model.initial_rate + (duration * 5 - 1) * model.additional_rate

theorem additional_rate_calculation (model : TelephoneCharge) 
  (h1 : model.initial_rate = 310/100)  -- 3.10 cents for the first 1/5 minute
  (h2 : total_charge model (8 : ℚ) = 1870/100)  -- 18.70 cents for 8 minutes
  : model.additional_rate = 40/100 := by
  sorry

end additional_rate_calculation_l3405_340535


namespace equal_hire_probability_l3405_340581

/-- Represents the hiring process for a factory with n job openings and n applicants. -/
structure HiringProcess (n : ℕ) where
  (n_ge_3 : n ≥ 3)
  (job_openings : Fin n)
  (applicants : Fin n)
  (qualified : Fin n → Fin n → Prop)
  (qualified_condition : ∀ i j : Fin n, qualified i j ↔ i.val ≥ j.val)
  (arrival_order : Fin n → Fin n)
  (is_hired : Fin n → Prop)

/-- The probability of an applicant being hired. -/
def hire_probability (hp : HiringProcess n) (applicant : Fin n) : ℝ :=
  sorry

/-- Theorem stating that applicants n and n-1 have the same probability of being hired. -/
theorem equal_hire_probability (hp : HiringProcess n) :
  hire_probability hp ⟨n - 1, sorry⟩ = hire_probability hp ⟨n - 2, sorry⟩ :=
sorry

end equal_hire_probability_l3405_340581


namespace parabola_m_value_l3405_340544

/-- Theorem: For a parabola with equation x² = my, where m is a positive real number,
    if the distance from the vertex to the directrix is 1/2, then m = 2. -/
theorem parabola_m_value (m : ℝ) (h1 : m > 0) : 
  (∀ x y : ℝ, x^2 = m*y) →  -- Parabola equation
  (1/2 : ℝ) = (1/4 : ℝ) * m →  -- Distance from vertex to directrix is 1/2
  m = 2 := by
sorry

end parabola_m_value_l3405_340544


namespace odd_function_implies_a_zero_l3405_340577

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The function f(x) = (x^2+1)(x+a) -/
def f (a : ℝ) : ℝ → ℝ := fun x ↦ (x^2 + 1) * (x + a)

theorem odd_function_implies_a_zero (a : ℝ) : IsOdd (f a) → a = 0 := by
  sorry

end odd_function_implies_a_zero_l3405_340577


namespace z_value_theorem_l3405_340585

theorem z_value_theorem (x y z : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ y ≠ x) 
  (eq : 1 / x - 1 / y = 1 / z) : z = (x * y) / (y - x) := by
  sorry

end z_value_theorem_l3405_340585


namespace equality_of_positive_integers_l3405_340575

theorem equality_of_positive_integers (a b : ℕ+) (p : ℕ) (h_prime : Nat.Prime p) 
  (h_p_eq : p = a + b + 1) (h_divides : p ∣ 4 * a * b - 1) : a = b := by
  sorry

end equality_of_positive_integers_l3405_340575


namespace project_completion_time_l3405_340505

theorem project_completion_time 
  (initial_people : ℕ) 
  (initial_days : ℕ) 
  (additional_people : ℕ) 
  (h1 : initial_people = 12)
  (h2 : initial_days = 15)
  (h3 : additional_people = 8) : 
  (initial_days + (initial_people * initial_days * 2) / (initial_people + additional_people)) = 33 :=
by sorry

end project_completion_time_l3405_340505


namespace min_value_sum_reciprocals_l3405_340561

theorem min_value_sum_reciprocals (p q r s t u : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hu : u > 0)
  (sum_eq_8 : p + q + r + s + t + u = 8) : 
  (1/p + 4/q + 9/r + 16/s + 25/t + 49/u) ≥ 60.5 := by
  sorry

end min_value_sum_reciprocals_l3405_340561


namespace tea_cost_price_l3405_340506

/-- The cost price per kg of the 80 kg of tea -/
def x : ℝ := 15

/-- The theorem stating that the cost price per kg of the 80 kg of tea is 15 -/
theorem tea_cost_price : 
  ∀ (quantity_1 quantity_2 cost_2 profit sale_price : ℝ),
  quantity_1 = 80 →
  quantity_2 = 20 →
  cost_2 = 20 →
  profit = 0.2 →
  sale_price = 19.2 →
  x = 15 :=
by
  sorry

end tea_cost_price_l3405_340506


namespace equal_roots_quadratic_l3405_340521

/-- 
A quadratic equation x^2 + 3x - k = 0 has two equal real roots if and only if k = -9/4.
-/
theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 + 3*x - k = 0 ∧ (∀ y : ℝ, y^2 + 3*y - k = 0 → y = x)) ↔ k = -9/4 := by
  sorry

end equal_roots_quadratic_l3405_340521


namespace product_inequality_l3405_340595

theorem product_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (habc : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := by
  sorry

end product_inequality_l3405_340595


namespace zoo_feeding_days_l3405_340588

def num_lions : ℕ := 3
def num_tigers : ℕ := 2
def num_leopards : ℕ := 5
def num_hyenas : ℕ := 4

def lion_consumption : ℕ := 25
def tiger_consumption : ℕ := 20
def leopard_consumption : ℕ := 15
def hyena_consumption : ℕ := 10

def total_meat : ℕ := 1200

def daily_consumption : ℕ :=
  num_lions * lion_consumption +
  num_tigers * tiger_consumption +
  num_leopards * leopard_consumption +
  num_hyenas * hyena_consumption

theorem zoo_feeding_days :
  (total_meat / daily_consumption : ℕ) = 5 := by sorry

end zoo_feeding_days_l3405_340588


namespace existence_of_plane_only_properties_l3405_340536

-- Define abstract types for plane and solid geometry
def PlaneGeometry : Type := Unit
def SolidGeometry : Type := Unit

-- Define a property as a function that takes a geometry and returns a proposition
def GeometricProperty : Type := (PlaneGeometry ⊕ SolidGeometry) → Prop

-- Define a function to check if a property holds in plane geometry
def holdsInPlaneGeometry (prop : GeometricProperty) : Prop :=
  prop (Sum.inl ())

-- Define a function to check if a property holds in solid geometry
def holdsInSolidGeometry (prop : GeometricProperty) : Prop :=
  prop (Sum.inr ())

-- State the theorem
theorem existence_of_plane_only_properties :
  ∃ (prop : GeometricProperty),
    holdsInPlaneGeometry prop ∧ ¬holdsInSolidGeometry prop := by
  sorry

-- Examples of properties (these are just placeholders and not actual proofs)
def perpendicularLinesParallel : GeometricProperty := fun _ => True
def uniquePerpendicularLine : GeometricProperty := fun _ => True
def equalSidedQuadrilateralIsRhombus : GeometricProperty := fun _ => True

end existence_of_plane_only_properties_l3405_340536


namespace orangeade_price_day2_l3405_340554

/-- Represents the price and volume of orangeade on a given day -/
structure OrangeadeDay where
  juice : ℝ
  water : ℝ
  price : ℝ

/-- Proves that the price per glass on the second day is $0.40 given the conditions -/
theorem orangeade_price_day2 (day1 day2 : OrangeadeDay) :
  day1.juice = day2.juice ∧ 
  day1.water = day1.juice ∧ 
  day2.water = 2 * day1.water ∧ 
  day1.price = 0.60 ∧ 
  (day1.juice + day1.water) * day1.price = (day2.juice + day2.water) * day2.price
  → day2.price = 0.40 := by
  sorry

#check orangeade_price_day2

end orangeade_price_day2_l3405_340554


namespace sqrt_difference_sum_l3405_340553

theorem sqrt_difference_sum (x : ℝ) : 
  Real.sqrt (64 - x^2) - Real.sqrt (36 - x^2) = 4 →
  Real.sqrt (64 - x^2) + Real.sqrt (36 - x^2) = 7 := by
  sorry

end sqrt_difference_sum_l3405_340553


namespace infinite_series_sum_l3405_340542

/-- The sum of the infinite series ∑(n=1 to ∞) (n+1) / (n^2(n+2)) is equal to 3/8 + π^2/24 -/
theorem infinite_series_sum : 
  ∑' n : ℕ, (n + 1 : ℝ) / (n^2 * (n + 2)) = 3/8 + π^2/24 := by
  sorry

end infinite_series_sum_l3405_340542


namespace store_profit_loss_l3405_340597

theorem store_profit_loss (price : ℝ) (profit_margin loss_margin : ℝ) : 
  price = 168 ∧ profit_margin = 0.2 ∧ loss_margin = 0.2 →
  (price - price / (1 + profit_margin)) + (price - price / (1 - loss_margin)) = -14 := by
  sorry

end store_profit_loss_l3405_340597


namespace symmetric_line_equation_l3405_340557

-- Define the points P and Q
def P : ℝ × ℝ := (3, 2)
def Q : ℝ × ℝ := (1, 4)

-- Define the line l as a function ax + by + c = 0
def line_l (a b c : ℝ) (x y : ℝ) : Prop := a * x + b * y + c = 0

-- Define symmetry with respect to a line
def symmetric_wrt_line (P Q : ℝ × ℝ) (a b c : ℝ) : Prop :=
  let midpoint := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  line_l a b c midpoint.1 midpoint.2 ∧
  a * (Q.2 - P.2) = b * (P.1 - Q.1)

-- Theorem statement
theorem symmetric_line_equation :
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ symmetric_wrt_line P Q a b c ∧ line_l a b c = line_l 1 (-1) 1 :=
by sorry

end symmetric_line_equation_l3405_340557


namespace greatest_lower_bound_reciprocal_sum_l3405_340503

theorem greatest_lower_bound_reciprocal_sum (a b : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) : 
  (1 / a + 1 / b ≥ 4) ∧ ∀ m > 4, ∃ a b, 0 < a ∧ 0 < b ∧ a + b = 1 ∧ 1 / a + 1 / b < m :=
sorry

end greatest_lower_bound_reciprocal_sum_l3405_340503


namespace simplify_square_roots_l3405_340515

theorem simplify_square_roots : Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7 := by
  sorry

end simplify_square_roots_l3405_340515


namespace compute_expression_l3405_340574

theorem compute_expression : 7^2 + 4*5 - 2^3 = 61 := by
  sorry

end compute_expression_l3405_340574


namespace average_daily_sales_l3405_340541

/-- Given the sales of pens over a 13-day period, calculate the average daily sales. -/
theorem average_daily_sales (day1_sales : ℕ) (other_days_sales : ℕ) (num_other_days : ℕ) : 
  day1_sales = 96 →
  other_days_sales = 44 →
  num_other_days = 12 →
  (day1_sales + num_other_days * other_days_sales) / (num_other_days + 1) = 48 :=
by
  sorry

#check average_daily_sales

end average_daily_sales_l3405_340541


namespace quadratic_expression_value_l3405_340594

theorem quadratic_expression_value (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 7) 
  (h2 : 2 * x + 3 * y = 8) : 
  13 * x^2 + 24 * x * y + 13 * y^2 = 113 := by
  sorry

end quadratic_expression_value_l3405_340594


namespace sin_135_degrees_l3405_340590

theorem sin_135_degrees : Real.sin (135 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end sin_135_degrees_l3405_340590


namespace unique_integer_solution_l3405_340573

theorem unique_integer_solution : ∃! (x : ℤ), (45 + x / 89) * 89 = 4028 :=
by
  -- The proof goes here
  sorry

end unique_integer_solution_l3405_340573


namespace max_vector_difference_value_l3405_340555

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define the theorem
theorem max_vector_difference_value (a b : V) (ha : ‖a‖ = 2) (hb : ‖b‖ = 1) :
  ∃ (c : V), ∀ (x : V), ‖a - b‖ ≤ ‖x‖ ∧ ‖x‖ = 3 :=
sorry

end max_vector_difference_value_l3405_340555
