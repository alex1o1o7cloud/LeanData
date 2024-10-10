import Mathlib

namespace fraction_equals_zero_l2064_206435

theorem fraction_equals_zero (x : ℝ) : (x^2 - 4) / (x + 2) = 0 → x = 2 := by
  sorry

end fraction_equals_zero_l2064_206435


namespace sodium_bisulfite_moles_required_l2064_206444

/-- Represents the balanced chemical equation for the reaction --/
structure ChemicalEquation :=
  (NaHSO3 : ℕ)
  (HCl : ℕ)
  (NaCl : ℕ)
  (H2O : ℕ)
  (SO2 : ℕ)

/-- The balanced equation for the reaction --/
def balanced_equation : ChemicalEquation :=
  { NaHSO3 := 1, HCl := 1, NaCl := 1, H2O := 1, SO2 := 1 }

/-- Theorem stating the number of moles of Sodium bisulfite required --/
theorem sodium_bisulfite_moles_required 
  (NaCl_produced : ℕ) 
  (HCl_used : ℕ) 
  (h1 : NaCl_produced = 2) 
  (h2 : HCl_used = 2) 
  (h3 : balanced_equation.NaHSO3 = balanced_equation.HCl) 
  (h4 : balanced_equation.NaHSO3 = balanced_equation.NaCl) :
  NaCl_produced = 2 := by
  sorry

end sodium_bisulfite_moles_required_l2064_206444


namespace total_cost_theorem_l2064_206493

-- Define the cost of individual items
def eraser_cost : ℝ := sorry
def pen_cost : ℝ := sorry
def marker_cost : ℝ := sorry

-- Define the conditions
axiom condition1 : eraser_cost + 3 * pen_cost + 2 * marker_cost = 240
axiom condition2 : 2 * eraser_cost + 4 * marker_cost + 5 * pen_cost = 440

-- Define the theorem to prove
theorem total_cost_theorem :
  3 * eraser_cost + 4 * pen_cost + 6 * marker_cost = 520 := by
  sorry

end total_cost_theorem_l2064_206493


namespace fishing_rod_price_l2064_206400

theorem fishing_rod_price (initial_price : ℝ) (saturday_increase : ℝ) (sunday_discount : ℝ) :
  initial_price = 50 ∧ 
  saturday_increase = 0.2 ∧ 
  sunday_discount = 0.15 →
  initial_price * (1 + saturday_increase) * (1 - sunday_discount) = 51 := by
  sorry

end fishing_rod_price_l2064_206400


namespace min_chinese_score_l2064_206478

/-- Represents the scores of a student in three subjects -/
structure Scores where
  chinese : ℝ
  mathematics : ℝ
  english : ℝ

/-- The average score of the three subjects is 92 -/
def average_score (s : Scores) : Prop :=
  (s.chinese + s.mathematics + s.english) / 3 = 92

/-- Each subject has a maximum score of 100 points -/
def max_score (s : Scores) : Prop :=
  s.chinese ≤ 100 ∧ s.mathematics ≤ 100 ∧ s.english ≤ 100

/-- The Mathematics score is 4 points higher than the Chinese score -/
def math_chinese_relation (s : Scores) : Prop :=
  s.mathematics = s.chinese + 4

/-- The minimum possible score for Chinese is 86 points -/
theorem min_chinese_score (s : Scores) 
  (h1 : average_score s) 
  (h2 : max_score s) 
  (h3 : math_chinese_relation s) : 
  s.chinese ≥ 86 := by
  sorry

end min_chinese_score_l2064_206478


namespace x_equals_y_when_t_is_half_l2064_206437

theorem x_equals_y_when_t_is_half (t : ℚ) : 
  let x := 1 - 4 * t
  let y := 2 * t - 2
  x = y ↔ t = 1/2 := by
sorry

end x_equals_y_when_t_is_half_l2064_206437


namespace fifth_root_equation_solution_l2064_206446

theorem fifth_root_equation_solution :
  ∃ x : ℝ, (x^(1/2) : ℝ) = 3 ∧ x^(1/2) = (x * (x^3)^(1/2))^(1/5) := by
  sorry

end fifth_root_equation_solution_l2064_206446


namespace fallen_pages_count_l2064_206410

/-- Represents a page number as a triple of digits -/
structure PageNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- Converts a PageNumber to its numerical value -/
def PageNumber.toNat (p : PageNumber) : Nat :=
  p.hundreds * 100 + p.tens * 10 + p.ones

/-- Checks if a PageNumber is even -/
def PageNumber.isEven (p : PageNumber) : Prop :=
  p.toNat % 2 = 0

/-- Checks if a PageNumber is a permutation of another PageNumber -/
def PageNumber.isPermutationOf (p1 p2 : PageNumber) : Prop :=
  (p1.hundreds = p2.hundreds ∨ p1.hundreds = p2.tens ∨ p1.hundreds = p2.ones) ∧
  (p1.tens = p2.hundreds ∨ p1.tens = p2.tens ∨ p1.tens = p2.ones) ∧
  (p1.ones = p2.hundreds ∨ p1.ones = p2.tens ∨ p1.ones = p2.ones)

theorem fallen_pages_count 
  (first_page last_page : PageNumber)
  (h_first : first_page.toNat = 143)
  (h_perm : last_page.isPermutationOf first_page)
  (h_even : last_page.isEven)
  (h_greater : last_page.toNat > first_page.toNat) :
  last_page.toNat - first_page.toNat + 1 = 172 := by
  sorry

end fallen_pages_count_l2064_206410


namespace at_least_four_same_prob_l2064_206407

-- Define the number of dice and sides
def num_dice : ℕ := 5
def num_sides : ℕ := 8

-- Define the probability of a specific outcome for a single die
def single_prob : ℚ := 1 / num_sides

-- Define the probability of all five dice showing the same number
def all_same_prob : ℚ := single_prob ^ (num_dice - 1)

-- Define the probability of exactly four dice showing the same number
def four_same_prob : ℚ := 
  (num_dice : ℚ) * single_prob ^ (num_dice - 2) * (1 - single_prob)

-- State the theorem
theorem at_least_four_same_prob : 
  all_same_prob + four_same_prob = 9 / 1024 := by sorry

end at_least_four_same_prob_l2064_206407


namespace balcony_difference_l2064_206454

/-- Represents the number of tickets sold for each section of the theater. -/
structure TheaterSales where
  orchestra : ℕ
  balcony : ℕ
  vip : ℕ

/-- Calculates the total revenue from ticket sales. -/
def totalRevenue (sales : TheaterSales) : ℕ :=
  15 * sales.orchestra + 10 * sales.balcony + 20 * sales.vip

/-- Calculates the total number of tickets sold. -/
def totalTickets (sales : TheaterSales) : ℕ :=
  sales.orchestra + sales.balcony + sales.vip

/-- Theorem stating the difference between balcony tickets and the sum of orchestra and VIP tickets. -/
theorem balcony_difference (sales : TheaterSales) 
    (h1 : totalTickets sales = 550)
    (h2 : totalRevenue sales = 8000) :
    sales.balcony - (sales.orchestra + sales.vip) = 370 := by
  sorry

end balcony_difference_l2064_206454


namespace greatest_integer_x_l2064_206404

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

def f (x : ℤ) : ℚ := (x^2 + 4*x + 13) / (x - 4)

theorem greatest_integer_x : 
  (∀ x : ℤ, x > 49 → ¬ is_integer (f x)) ∧ 
  is_integer (f 49) := by
sorry

end greatest_integer_x_l2064_206404


namespace money_at_departure_l2064_206464

def money_at_arrival : ℕ := 87
def money_difference : ℕ := 71

theorem money_at_departure : 
  money_at_arrival - money_difference = 16 := by sorry

end money_at_departure_l2064_206464


namespace line_perp_plane_implies_planes_perp_l2064_206447

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (containedIn : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (planePerpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_plane_implies_planes_perp
  (m n : Line) (α β : Plane)
  (h1 : m ≠ n)
  (h2 : α ≠ β)
  (h3 : containedIn m α)
  (h4 : containedIn n β)
  (h5 : perpendicular n α) :
  planePerpendicular α β :=
sorry

end line_perp_plane_implies_planes_perp_l2064_206447


namespace camping_hike_distance_l2064_206457

/-- The total distance hiked by Irwin's family during their camping trip -/
theorem camping_hike_distance 
  (car_to_stream : ℝ) 
  (stream_to_meadow : ℝ) 
  (meadow_to_campsite : ℝ)
  (h1 : car_to_stream = 0.2)
  (h2 : stream_to_meadow = 0.4)
  (h3 : meadow_to_campsite = 0.1) :
  car_to_stream + stream_to_meadow + meadow_to_campsite = 0.7 := by
  sorry

end camping_hike_distance_l2064_206457


namespace volume_of_rectangular_prism_l2064_206422

/-- Represents a rectangular prism with dimensions a, d, and h -/
structure RectangularPrism where
  a : ℝ
  d : ℝ
  h : ℝ
  a_pos : 0 < a
  d_pos : 0 < d
  h_pos : 0 < h

/-- Calculates the volume of a rectangular prism -/
def volume (prism : RectangularPrism) : ℝ :=
  prism.a * prism.d * prism.h

/-- Theorem: The volume of a rectangular prism is equal to a * d * h -/
theorem volume_of_rectangular_prism (prism : RectangularPrism) :
  volume prism = prism.a * prism.d * prism.h :=
by sorry

end volume_of_rectangular_prism_l2064_206422


namespace square_sum_zero_implies_both_zero_l2064_206469

theorem square_sum_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end square_sum_zero_implies_both_zero_l2064_206469


namespace tan_theta_value_l2064_206472

theorem tan_theta_value (θ : ℝ) :
  let z : ℂ := Complex.mk (Real.sin θ - 3/5) (Real.cos θ - 4/5)
  (z.re = 0 ∧ z.im ≠ 0) → Real.tan θ = -3/4 := by
  sorry

end tan_theta_value_l2064_206472


namespace geometry_biology_overlap_l2064_206413

theorem geometry_biology_overlap (total : ℕ) (geometry : ℕ) (biology : ℕ)
  (h_total : total = 232)
  (h_geometry : geometry = 144)
  (h_biology : biology = 119) :
  min geometry biology - (geometry + biology - total) = 88 :=
by sorry

end geometry_biology_overlap_l2064_206413


namespace base4_21012_to_decimal_l2064_206492

def base4_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (4 ^ i)) 0

theorem base4_21012_to_decimal :
  base4_to_decimal [2, 1, 0, 1, 2] = 582 := by
  sorry

end base4_21012_to_decimal_l2064_206492


namespace divisible_by_ten_l2064_206459

theorem divisible_by_ten : ∃ k : ℤ, 43^43 - 17^17 = 10 * k := by
  sorry

end divisible_by_ten_l2064_206459


namespace total_cost_mangoes_l2064_206442

def prices : List Nat := [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
def boxes : Nat := 36

theorem total_cost_mangoes :
  (List.sum prices) * boxes = 3060 := by
  sorry

end total_cost_mangoes_l2064_206442


namespace equation_transformation_l2064_206491

theorem equation_transformation (x y : ℝ) (h : y = x + 1/x) :
  x^3 + x^2 - 6*x + 1 = 0 ↔ x*(x^2*y - 6) + 1 = 0 := by
sorry

end equation_transformation_l2064_206491


namespace min_value_of_expression_l2064_206428

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (x^2 / (x + 2) + y^2 / (y + 1)) ≥ 1/4 ∧ 
  ∃ x₀ y₀, x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧ x₀^2 / (x₀ + 2) + y₀^2 / (y₀ + 1) = 1/4 :=
by sorry

end min_value_of_expression_l2064_206428


namespace new_student_weight_l2064_206424

theorem new_student_weight (initial_count : ℕ) (replaced_weight : ℝ) (average_decrease : ℝ) :
  initial_count = 5 →
  replaced_weight = 72 →
  average_decrease = 12 →
  let new_weight := replaced_weight - initial_count * average_decrease
  new_weight = 12 := by
  sorry

end new_student_weight_l2064_206424


namespace x_eq_2_sufficient_not_necessary_l2064_206417

def a (x : ℝ) : Fin 2 → ℝ := ![1, x - 1]
def b (x : ℝ) : Fin 2 → ℝ := ![x + 1, 3]

def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ k : ℝ, ∀ i : Fin 2, v i = k * w i

theorem x_eq_2_sufficient_not_necessary :
  (∃ x : ℝ, x ≠ 2 ∧ parallel (a x) (b x)) ∧
  (∀ x : ℝ, x = 2 → parallel (a x) (b x)) :=
sorry

end x_eq_2_sufficient_not_necessary_l2064_206417


namespace first_triangular_year_21st_century_l2064_206477

/-- Triangular number function -/
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

/-- First triangular number year in the 21st century -/
theorem first_triangular_year_21st_century :
  ∃ n : ℕ, triangular n = 2016 ∧ 
  (∀ m : ℕ, triangular m ≥ 2000 → triangular n ≤ triangular m) := by
  sorry

end first_triangular_year_21st_century_l2064_206477


namespace roots_sum_squares_and_product_l2064_206466

theorem roots_sum_squares_and_product (α β : ℝ) : 
  (2 * α^2 - α - 4 = 0) → (2 * β^2 - β - 4 = 0) → α^2 + α*β + β^2 = 9/4 := by
  sorry

end roots_sum_squares_and_product_l2064_206466


namespace grace_pool_volume_l2064_206445

/-- The volume of water in Grace's pool -/
def pool_volume (first_hose_rate : ℝ) (first_hose_time : ℝ) (second_hose_rate : ℝ) (second_hose_time : ℝ) : ℝ :=
  first_hose_rate * first_hose_time + second_hose_rate * second_hose_time

/-- Theorem stating that Grace's pool contains 390 gallons of water -/
theorem grace_pool_volume :
  let first_hose_rate : ℝ := 50
  let first_hose_time : ℝ := 5
  let second_hose_rate : ℝ := 70
  let second_hose_time : ℝ := 2
  pool_volume first_hose_rate first_hose_time second_hose_rate second_hose_time = 390 :=
by
  sorry


end grace_pool_volume_l2064_206445


namespace some_base_value_l2064_206484

theorem some_base_value (x y some_base : ℝ) 
  (h1 : x * y = 1)
  (h2 : (some_base ^ ((x + y)^2)) / (some_base ^ ((x - y)^2)) = 256) :
  some_base = 4 := by
sorry

end some_base_value_l2064_206484


namespace three_equidistant_lines_l2064_206481

/-- A point in a plane represented by its coordinates -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A line in a plane represented by its equation ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Returns true if three points are not collinear -/
def nonCollinear (p1 p2 p3 : Point2D) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) ≠ (p3.x - p1.x) * (p2.y - p1.y)

/-- Returns true if a line is equidistant from three points -/
def equidistantLine (l : Line2D) (p1 p2 p3 : Point2D) : Prop :=
  let d1 := |l.a * p1.x + l.b * p1.y + l.c| / Real.sqrt (l.a^2 + l.b^2)
  let d2 := |l.a * p2.x + l.b * p2.y + l.c| / Real.sqrt (l.a^2 + l.b^2)
  let d3 := |l.a * p3.x + l.b * p3.y + l.c| / Real.sqrt (l.a^2 + l.b^2)
  d1 = d2 ∧ d2 = d3

/-- Main theorem: There are exactly three lines equidistant from three non-collinear points -/
theorem three_equidistant_lines (p1 p2 p3 : Point2D) 
  (h : nonCollinear p1 p2 p3) : 
  ∃! (s : Finset Line2D), s.card = 3 ∧ ∀ l ∈ s, equidistantLine l p1 p2 p3 :=
sorry

end three_equidistant_lines_l2064_206481


namespace equation_solution_l2064_206436

theorem equation_solution : ∃ x : ℝ, 5 * (x - 4) = 2 * (3 - 2 * x) + 10 ∧ x = 4 := by
  sorry

end equation_solution_l2064_206436


namespace eightiethDigitIsOne_l2064_206490

/-- The sequence of digits formed by concatenating consecutive integers from 60 to 1 in descending order -/
def descendingSequence : List Nat := sorry

/-- The 80th digit in the descendingSequence -/
def eightiethDigit : Nat := sorry

/-- Theorem stating that the 80th digit in the sequence is 1 -/
theorem eightiethDigitIsOne : eightiethDigit = 1 := by sorry

end eightiethDigitIsOne_l2064_206490


namespace complex_fraction_simplification_l2064_206414

theorem complex_fraction_simplification :
  let z₁ : ℂ := 2 + 4 * I
  let z₂ : ℂ := 2 - 4 * I
  z₁ / z₂ - z₂ / z₁ = -8/5 + 16/5 * I :=
by sorry

end complex_fraction_simplification_l2064_206414


namespace joeys_route_length_l2064_206405

/-- Given a round trip with total time 1 hour and average speed 3 miles/hour,
    prove that the one-way distance is 1.5 miles. -/
theorem joeys_route_length (total_time : ℝ) (avg_speed : ℝ) (one_way_distance : ℝ) :
  total_time = 1 →
  avg_speed = 3 →
  one_way_distance = avg_speed * total_time / 2 →
  one_way_distance = 1.5 := by
  sorry

#check joeys_route_length

end joeys_route_length_l2064_206405


namespace multiplication_subtraction_equality_l2064_206483

theorem multiplication_subtraction_equality : 120 * 2400 - 20 * 2400 - 100 * 2400 = 0 := by
  sorry

end multiplication_subtraction_equality_l2064_206483


namespace range_of_a_l2064_206497

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 6) * (x - 2*a - 5) > 0}
def B (a : ℝ) : Set ℝ := {x | (a^2 + 2 - x) * (2*a - x) < 0}

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, 
    a > 1/2 → 
    (∀ x : ℝ, x ∈ B a → x ∈ A a) → 
    (∃ x : ℝ, x ∈ A a ∧ x ∉ B a) → 
    a > 1/2 ∧ a ≤ 2 :=
by sorry

end range_of_a_l2064_206497


namespace absolute_value_reciprocal_graph_l2064_206488

theorem absolute_value_reciprocal_graph (x : ℝ) (x_nonzero : x ≠ 0) :
  (1 / |x|) = if x > 0 then 1 / x else -1 / x :=
by sorry

end absolute_value_reciprocal_graph_l2064_206488


namespace work_speed_ratio_l2064_206438

-- Define the work speeds
def A_work_speed : ℚ := 1 / 18
def B_work_speed : ℚ := 1 / 36

-- Define the combined work speed
def combined_work_speed : ℚ := 1 / 12

-- Theorem statement
theorem work_speed_ratio :
  (A_work_speed + B_work_speed = combined_work_speed) →
  (A_work_speed / B_work_speed = 2) :=
by
  sorry

end work_speed_ratio_l2064_206438


namespace quadratic_inequality_range_l2064_206421

theorem quadratic_inequality_range (a : ℝ) : 
  (a ≠ 0 ∧ ∀ x : ℝ, a * x^2 + 2 * a * x - 4 < 0) ↔ (-4 < a ∧ a < 0) :=
sorry

end quadratic_inequality_range_l2064_206421


namespace f_plus_a_over_e_positive_sum_less_than_two_l2064_206463

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := (a * x + 1) * Real.exp x

theorem f_plus_a_over_e_positive (a : ℝ) (h : a > 0) :
  ∀ x : ℝ, f a x + a / Real.exp 1 > 0 := by sorry

theorem sum_less_than_two (x₁ x₂ : ℝ) (h1 : x₁ ≠ x₂) (h2 : f (-1/2) x₁ = f (-1/2) x₂) :
  x₁ + x₂ < 2 := by sorry

end

end f_plus_a_over_e_positive_sum_less_than_two_l2064_206463


namespace sum_of_repeating_decimals_l2064_206471

/-- Definition of the repeating decimal 0.3333... -/
def repeating_3 : ℚ := 1/3

/-- Definition of the repeating decimal 0.0404... -/
def repeating_04 : ℚ := 4/99

/-- Definition of the repeating decimal 0.005005... -/
def repeating_005 : ℚ := 5/999

/-- Theorem stating that the sum of the three repeating decimals equals 1135/2997 -/
theorem sum_of_repeating_decimals : 
  repeating_3 + repeating_04 + repeating_005 = 1135/2997 := by sorry

end sum_of_repeating_decimals_l2064_206471


namespace sector_area_l2064_206402

theorem sector_area (perimeter : ℝ) (central_angle : ℝ) (area : ℝ) : 
  perimeter = 8 → central_angle = 2 → area = 4 := by
  sorry

end sector_area_l2064_206402


namespace total_area_calculation_l2064_206461

def original_length : ℝ := 13
def original_width : ℝ := 18
def increase : ℝ := 2
def num_equal_rooms : ℕ := 4
def num_double_rooms : ℕ := 1

def new_length : ℝ := original_length + increase
def new_width : ℝ := original_width + increase

def room_area : ℝ := new_length * new_width

theorem total_area_calculation :
  (num_equal_rooms : ℝ) * room_area + (num_double_rooms : ℝ) * 2 * room_area = 1800 := by
  sorry

end total_area_calculation_l2064_206461


namespace sum_even_implies_one_even_l2064_206439

theorem sum_even_implies_one_even (a b c : ℕ) :
  Even (a + b + c) → (Even a ∨ Even b ∨ Even c) :=
sorry

end sum_even_implies_one_even_l2064_206439


namespace zero_point_in_interval_l2064_206465

/-- The function f(x) = ln x - 3/x has a zero point in the interval (2, 3) -/
theorem zero_point_in_interval (f : ℝ → ℝ) :
  (∀ x > 0, f x = Real.log x - 3 / x) →
  (∀ x > 0, StrictMono f) →
  ∃ c ∈ Set.Ioo 2 3, f c = 0 := by
  sorry

end zero_point_in_interval_l2064_206465


namespace solve_equation_l2064_206431

theorem solve_equation : (45 : ℚ) / (9 - 3/7) = 21/4 := by sorry

end solve_equation_l2064_206431


namespace golden_ratio_cosine_l2064_206476

theorem golden_ratio_cosine (golden_ratio : ℝ) (h1 : golden_ratio = (Real.sqrt 5 - 1) / 2) 
  (h2 : golden_ratio = 2 * Real.sin (18 * π / 180)) : 
  Real.cos (36 * π / 180) = (Real.sqrt 5 + 1) / 4 := by
  sorry

end golden_ratio_cosine_l2064_206476


namespace exists_same_color_distance_exists_color_for_all_distances_l2064_206453

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function that assigns a color to each point
def colorAssignment : Point → Color := sorry

-- Define a function to calculate distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Statement for part (i)
theorem exists_same_color_distance (x : ℝ) :
  ∃ (c : Color) (p1 p2 : Point),
    colorAssignment p1 = c ∧
    colorAssignment p2 = c ∧
    distance p1 p2 = x :=
  sorry

-- Statement for part (ii)
theorem exists_color_for_all_distances :
  ∃ (c : Color), ∀ (x : ℝ),
    ∃ (p1 p2 : Point),
      colorAssignment p1 = c ∧
      colorAssignment p2 = c ∧
      distance p1 p2 = x :=
  sorry

end exists_same_color_distance_exists_color_for_all_distances_l2064_206453


namespace min_value_and_inequality_l2064_206494

/-- Given positive real numbers a, b, and c such that the minimum value of |x - a| + |x + b| + c is 1 -/
def min_condition (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (∀ x, |x - a| + |x + b| + c ≥ 1) ∧ (∃ x, |x - a| + |x + b| + c = 1)

theorem min_value_and_inequality (a b c : ℝ) (h : min_condition a b c) :
  (∀ x y z, 9*x^2 + 4*y^2 + (1/4)*z^2 ≥ 36/157) ∧
  (9*a^2 + 4*b^2 + (1/4)*c^2 = 36/157) ∧
  (Real.sqrt (a^2 + a*b + b^2) + Real.sqrt (b^2 + b*c + c^2) + Real.sqrt (c^2 + c*a + a^2) > 3/2) :=
by sorry

end min_value_and_inequality_l2064_206494


namespace farm_heads_count_l2064_206426

/-- Represents a farm with hens and cows -/
structure Farm where
  hens : ℕ
  cows : ℕ

/-- The total number of feet on the farm -/
def totalFeet (f : Farm) : ℕ := 2 * f.hens + 4 * f.cows

/-- The total number of heads on the farm -/
def totalHeads (f : Farm) : ℕ := f.hens + f.cows

/-- Theorem stating that a farm with 140 feet and 22 hens has 46 heads -/
theorem farm_heads_count (f : Farm) 
  (feet_count : totalFeet f = 140) 
  (hen_count : f.hens = 22) : 
  totalHeads f = 46 := by
  sorry

end farm_heads_count_l2064_206426


namespace completing_square_equivalence_l2064_206401

theorem completing_square_equivalence :
  ∀ x : ℝ, 4 * x^2 - 2 * x - 1 = 0 ↔ (x - 1/4)^2 = 5/16 := by
  sorry

end completing_square_equivalence_l2064_206401


namespace percent_of_x_is_z_l2064_206420

theorem percent_of_x_is_z (x y z : ℝ) 
  (h1 : 0.45 * z = 0.39 * y) 
  (h2 : y = 0.75 * x) : 
  z = 0.65 * x := by
  sorry

end percent_of_x_is_z_l2064_206420


namespace complex_number_coordinates_l2064_206451

theorem complex_number_coordinates : 
  let z : ℂ := Complex.I * (2 - Complex.I)
  (z.re = 1 ∧ z.im = 2) :=
by sorry

end complex_number_coordinates_l2064_206451


namespace p_necessary_not_sufficient_l2064_206495

-- Define the propositions p and q
def p (x y : ℝ) : Prop := x + y > 2 ∧ x * y > 1
def q (x y : ℝ) : Prop := x > 1 ∧ y > 1

-- Theorem statement
theorem p_necessary_not_sufficient :
  (∀ x y : ℝ, q x y → p x y) ∧
  ¬(∀ x y : ℝ, p x y → q x y) :=
sorry

end p_necessary_not_sufficient_l2064_206495


namespace complex_equation_solution_l2064_206443

theorem complex_equation_solution (x : ℝ) (y : ℂ) 
  (h1 : y.re = 0)  -- y is purely imaginary
  (h2 : (3 * x + 1 : ℂ) - 2 * Complex.I = y) : 
  x = -1/3 ∧ y = -2 * Complex.I := by
  sorry

end complex_equation_solution_l2064_206443


namespace sum_not_exceeding_eight_probability_most_probable_sum_probability_of_most_probable_sum_l2064_206458

def ball_count : ℕ := 8

def ball_labels : Finset ℕ := Finset.range ball_count

def sum_of_pair (i j : ℕ) : ℕ := i + j

def valid_pairs : Finset (ℕ × ℕ) :=
  (ball_labels.product ball_labels).filter (λ p => p.1 < p.2)

def pairs_with_sum (n : ℕ) : Finset (ℕ × ℕ) :=
  valid_pairs.filter (λ p => sum_of_pair p.1 p.2 = n)

def probability (favorable : Finset (ℕ × ℕ)) : ℚ :=
  favorable.card / valid_pairs.card

theorem sum_not_exceeding_eight_probability :
  probability (valid_pairs.filter (λ p => sum_of_pair p.1 p.2 ≤ 8)) = 3/7 := by sorry

theorem most_probable_sum :
  ∃ n : ℕ, n = 9 ∧ 
    ∀ m : ℕ, probability (pairs_with_sum n) ≥ probability (pairs_with_sum m) := by sorry

theorem probability_of_most_probable_sum :
  probability (pairs_with_sum 9) = 1/7 := by sorry

end sum_not_exceeding_eight_probability_most_probable_sum_probability_of_most_probable_sum_l2064_206458


namespace polynomial_simplification_l2064_206479

theorem polynomial_simplification (y : ℝ) :
  (2 * y^6 + 3 * y^5 + y^3 + 15) - (y^6 + 4 * y^5 - 2 * y^4 + 17) =
  y^6 - y^5 + 2 * y^4 + y^3 - 2 := by
  sorry

end polynomial_simplification_l2064_206479


namespace page_difference_l2064_206462

/-- The number of purple books Mirella read -/
def purple_books : ℕ := 8

/-- The number of orange books Mirella read -/
def orange_books : ℕ := 7

/-- The number of pages in each purple book -/
def purple_pages : ℕ := 320

/-- The number of pages in each orange book -/
def orange_pages : ℕ := 640

/-- The difference between the total number of orange pages and purple pages read by Mirella -/
theorem page_difference : 
  orange_books * orange_pages - purple_books * purple_pages = 1920 := by
  sorry

end page_difference_l2064_206462


namespace initial_charge_correct_l2064_206434

/-- The initial charge for renting a bike at Oceanside Bike Rental Shop -/
def initial_charge : ℝ := 17

/-- The hourly rate for renting a bike -/
def hourly_rate : ℝ := 7

/-- The number of hours Tom rented the bike -/
def rental_hours : ℝ := 9

/-- The total cost Tom paid for renting the bike -/
def total_cost : ℝ := 80

/-- Theorem stating that the initial charge is correct given the conditions -/
theorem initial_charge_correct : 
  initial_charge + hourly_rate * rental_hours = total_cost :=
by sorry

end initial_charge_correct_l2064_206434


namespace expression_simplification_l2064_206456

theorem expression_simplification (x : ℚ) (h : x = 3) : 
  (((x - 1) / (x + 2) + 1) / ((x - 1) / (x + 2) - 1)) = -7/3 := by
  sorry

end expression_simplification_l2064_206456


namespace trigonometric_expression_equals_two_l2064_206423

theorem trigonometric_expression_equals_two :
  (Real.cos (10 * π / 180) + Real.sqrt 3 * Real.sin (10 * π / 180)) /
  Real.sqrt (1 - Real.sin (50 * π / 180) ^ 2) = 2 := by
  sorry

end trigonometric_expression_equals_two_l2064_206423


namespace missing_fraction_problem_l2064_206415

theorem missing_fraction_problem (sum : ℚ) (f1 f2 f3 f4 f5 f6 f7 : ℚ) : 
  sum = 45/100 →
  f1 = 1/3 →
  f2 = 1/2 →
  f3 = -5/6 →
  f4 = 1/4 →
  f5 = -9/20 →
  f6 = -9/20 →
  f1 + f2 + f3 + f4 + f5 + f6 + f7 = sum →
  f7 = 11/10 := by
sorry

end missing_fraction_problem_l2064_206415


namespace cubic_meter_to_cubic_cm_l2064_206448

-- Define the conversion factor from meters to centimeters
def meters_to_cm : ℝ := 100

-- Theorem statement
theorem cubic_meter_to_cubic_cm :
  (1 : ℝ) * meters_to_cm^3 = 1000000 := by
  sorry

end cubic_meter_to_cubic_cm_l2064_206448


namespace paul_sold_63_books_l2064_206468

/-- The number of books Paul sold in a garage sale --/
def books_sold_in_garage_sale (initial_books donated_books exchanged_books given_to_friend remaining_books : ℕ) : ℕ :=
  initial_books - donated_books - given_to_friend - remaining_books

/-- Theorem stating that Paul sold 63 books in the garage sale --/
theorem paul_sold_63_books :
  books_sold_in_garage_sale 250 50 20 35 102 = 63 := by
  sorry

end paul_sold_63_books_l2064_206468


namespace xiao_ming_running_time_l2064_206473

theorem xiao_ming_running_time (track_length : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : track_length = 360)
  (h2 : speed1 = 5)
  (h3 : speed2 = 4) :
  let avg_speed := (speed1 + speed2) / 2
  let total_time := track_length / avg_speed
  let half_distance := track_length / 2
  let second_half_time := half_distance / speed2
  second_half_time = 44 := by
sorry

end xiao_ming_running_time_l2064_206473


namespace polynomial_root_problem_l2064_206418

/-- The polynomial h(x) -/
def h (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 2*x + 15

/-- The polynomial f(x) -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^4 - x^3 + b*x^2 + 120*x + c

/-- The theorem statement -/
theorem polynomial_root_problem (a b c : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    h a x = 0 ∧ h a y = 0 ∧ h a z = 0 ∧
    f b c x = 0 ∧ f b c y = 0 ∧ f b c z = 0) →
  f b c (-1) = -1995.25 := by
  sorry

end polynomial_root_problem_l2064_206418


namespace area_isosceles_right_triangle_radius_circumcircle_isosceles_right_triangle_l2064_206449

/-- A right triangle with two equal angles and hypotenuse 8√2 -/
structure IsoscelesRightTriangle where
  /-- The length of the hypotenuse -/
  hypotenuse : ℝ
  /-- The hypotenuse is 8√2 -/
  hypotenuse_eq : hypotenuse = 8 * Real.sqrt 2

/-- The area of an isosceles right triangle with hypotenuse 8√2 is 32 -/
theorem area_isosceles_right_triangle (t : IsoscelesRightTriangle) :
    (1 / 2 : ℝ) * t.hypotenuse^2 / 2 = 32 := by sorry

/-- The radius of the circumcircle of an isosceles right triangle with hypotenuse 8√2 is 4√2 -/
theorem radius_circumcircle_isosceles_right_triangle (t : IsoscelesRightTriangle) :
    t.hypotenuse / 2 = 4 * Real.sqrt 2 := by sorry

end area_isosceles_right_triangle_radius_circumcircle_isosceles_right_triangle_l2064_206449


namespace third_quadrant_trig_sum_l2064_206486

theorem third_quadrant_trig_sum (α : Real) : 
  π < α ∧ α < 3*π/2 → 
  |Real.sin (α/2)| / Real.sin (α/2) + |Real.cos (α/2)| / Real.cos (α/2) = 0 := by
sorry

end third_quadrant_trig_sum_l2064_206486


namespace barrel_capacity_l2064_206475

theorem barrel_capacity (original_amount : ℝ) (capacity : ℝ) : 
  (original_amount = 3 / 5 * capacity) →
  (original_amount - 18 = 0.6 * original_amount) →
  (capacity = 75) :=
by sorry

end barrel_capacity_l2064_206475


namespace no_three_digit_number_eight_times_smaller_l2064_206408

theorem no_three_digit_number_eight_times_smaller : ¬ ∃ (a b c : ℕ), 
  (1 ≤ a ∧ a ≤ 9) ∧ 
  (b ≤ 9) ∧ 
  (c ≤ 9) ∧ 
  (100 * a + 10 * b + c = 8 * (10 * b + c)) := by
sorry

end no_three_digit_number_eight_times_smaller_l2064_206408


namespace sum_and_product_of_radical_conjugates_l2064_206409

theorem sum_and_product_of_radical_conjugates (a b : ℝ) : 
  ((a + Real.sqrt b) + (a - Real.sqrt b) = -6) →
  ((a + Real.sqrt b) * (a - Real.sqrt b) = 9) →
  (a + b = -3) := by
  sorry

end sum_and_product_of_radical_conjugates_l2064_206409


namespace smallest_positive_angle_theorem_l2064_206433

theorem smallest_positive_angle_theorem (y : ℝ) : 
  (5 * Real.cos y * Real.sin y ^ 3 - 5 * Real.cos y ^ 3 * Real.sin y = 1 / 2) →
  y = (1 / 4) * Real.arcsin (2 / 5) ∧ y > 0 ∧ 
  ∀ z, z > 0 → (5 * Real.cos z * Real.sin z ^ 3 - 5 * Real.cos z ^ 3 * Real.sin z = 1 / 2) → y ≤ z :=
by sorry

end smallest_positive_angle_theorem_l2064_206433


namespace one_in_set_zero_one_l2064_206430

theorem one_in_set_zero_one : 1 ∈ ({0, 1} : Set ℕ) := by sorry

end one_in_set_zero_one_l2064_206430


namespace hyperbola_asymptote_l2064_206455

/-- Given a hyperbola with equation x²/4 - y²/m² = 1 where m > 0,
    and one of its asymptotes is 5x - 2y = 0, prove that m = 5. -/
theorem hyperbola_asymptote (m : ℝ) (h1 : m > 0) : 
  (∃ x y : ℝ, x^2/4 - y^2/m^2 = 1 ∧ 5*x - 2*y = 0) → m = 5 := by
sorry

end hyperbola_asymptote_l2064_206455


namespace turtle_race_ratio_l2064_206416

theorem turtle_race_ratio : 
  ∀ (greta_time george_time gloria_time : ℕ),
    greta_time = 6 →
    george_time = greta_time - 2 →
    gloria_time = 8 →
    (gloria_time : ℚ) / (george_time : ℚ) = 2 := by
  sorry

end turtle_race_ratio_l2064_206416


namespace ticket_difference_l2064_206440

theorem ticket_difference (initial_tickets : ℕ) (remaining_tickets : ℕ) : 
  initial_tickets = 48 → remaining_tickets = 32 → initial_tickets - remaining_tickets = 16 := by
  sorry

end ticket_difference_l2064_206440


namespace article_cost_is_60_l2064_206487

/-- Proves that the cost of an article is 60, given the specified conditions --/
theorem article_cost_is_60 (cost : ℝ) (selling_price : ℝ) : 
  (selling_price = 1.25 * cost) →
  (0.8 * cost + 0.3 * (0.8 * cost) = selling_price - 12.6) →
  cost = 60 := by
  sorry

end article_cost_is_60_l2064_206487


namespace moon_speed_conversion_l2064_206419

/-- Converts a speed from kilometers per second to kilometers per hour -/
def km_per_second_to_km_per_hour (speed_km_per_second : ℝ) : ℝ :=
  speed_km_per_second * 3600

/-- The moon's speed in kilometers per second -/
def moon_speed_km_per_second : ℝ := 0.9

theorem moon_speed_conversion :
  km_per_second_to_km_per_hour moon_speed_km_per_second = 3240 := by
  sorry

end moon_speed_conversion_l2064_206419


namespace largest_n_divisible_by_seven_largest_n_is_99999_l2064_206452

theorem largest_n_divisible_by_seven (n : ℕ) : n < 100000 →
  (10 * (n - 3)^5 - n^2 + 20 * n - 30) % 7 = 0 →
  n ≤ 99999 :=
by sorry

theorem largest_n_is_99999 :
  (10 * (99999 - 3)^5 - 99999^2 + 20 * 99999 - 30) % 7 = 0 ∧
  ∀ m : ℕ, m > 99999 → m < 100000 →
    (10 * (m - 3)^5 - m^2 + 20 * m - 30) % 7 ≠ 0 :=
by sorry

end largest_n_divisible_by_seven_largest_n_is_99999_l2064_206452


namespace geometric_progression_problem_l2064_206489

theorem geometric_progression_problem :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    (b / a = c / b) ∧
    (a + b + c = 65) ∧
    (a * b * c = 3375) ∧
    ((a = 5 ∧ b = 15 ∧ c = 45) ∨ (a = 45 ∧ b = 15 ∧ c = 5)) :=
by sorry

end geometric_progression_problem_l2064_206489


namespace complex_coordinate_l2064_206467

theorem complex_coordinate (z : ℂ) (h : Complex.I * z = 1 + 2 * Complex.I) : z = 2 - Complex.I := by
  sorry

end complex_coordinate_l2064_206467


namespace negation_of_symmetry_for_all_l2064_206498

-- Define a type for functions
variable {α : Type*} [LinearOrder α]

-- Define symmetry about y=x
def symmetric_about_y_eq_x (f : α → α) : Prop :=
  ∀ x y, f y = x ↔ f x = y

-- State the theorem
theorem negation_of_symmetry_for_all :
  (¬ ∀ f : α → α, symmetric_about_y_eq_x f) ↔
  (∃ f : α → α, ¬ symmetric_about_y_eq_x f) :=
sorry

end negation_of_symmetry_for_all_l2064_206498


namespace remainder_theorem_l2064_206411

theorem remainder_theorem : 
  (2^300 + 405) % (2^150 + 2^75 + 1) = 404 := by
  sorry

end remainder_theorem_l2064_206411


namespace nails_per_paw_is_four_l2064_206406

/-- The number of nails on one paw of a dog -/
def nails_per_paw : ℕ := sorry

/-- The total number of trimmed nails -/
def total_nails : ℕ := 164

/-- The number of dogs with three legs -/
def three_legged_dogs : ℕ := 3

/-- Theorem stating that the number of nails on one paw of a dog is 4 -/
theorem nails_per_paw_is_four : nails_per_paw = 4 := by sorry

end nails_per_paw_is_four_l2064_206406


namespace solution_correctness_l2064_206429

def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(0, 1, 1), (0, -1, -1), (1, 1, 0), (-1, -1, 0), (1, 0, 1), (-1, 0, -1),
   (Real.sqrt 3 / 3, Real.sqrt 3 / 3, Real.sqrt 3 / 3),
   (-Real.sqrt 3 / 3, -Real.sqrt 3 / 3, -Real.sqrt 3 / 3)}

def satisfies_conditions (a b c : ℝ) : Prop :=
  a^2*b + c = b^2*c + a ∧ 
  b^2*c + a = c^2*a + b ∧
  a*b + b*c + c*a = 1

theorem solution_correctness :
  ∀ (a b c : ℝ), satisfies_conditions a b c ↔ (a, b, c) ∈ solution_set :=
by sorry

end solution_correctness_l2064_206429


namespace quadratic_inequality_solution_l2064_206427

theorem quadratic_inequality_solution :
  {z : ℝ | z^2 - 40*z + 340 ≤ 4} = Set.Icc 12 28 := by sorry

end quadratic_inequality_solution_l2064_206427


namespace total_pizza_pieces_l2064_206425

/-- Given 10 children, each buying 20 pizzas, and each pizza containing 6 pieces,
    the total number of pizza pieces is 1200. -/
theorem total_pizza_pieces :
  let num_children : ℕ := 10
  let pizzas_per_child : ℕ := 20
  let pieces_per_pizza : ℕ := 6
  num_children * pizzas_per_child * pieces_per_pizza = 1200 :=
by
  sorry


end total_pizza_pieces_l2064_206425


namespace reflect_point_coords_l2064_206432

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a function to reflect a point across the xz-plane
def reflectAcrossXZPlane (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

-- Theorem statement
theorem reflect_point_coords :
  let original := Point3D.mk (-4) 3 5
  let reflected := reflectAcrossXZPlane original
  reflected.x = 4 ∧ reflected.y = -3 ∧ reflected.z = 5 := by
  sorry


end reflect_point_coords_l2064_206432


namespace honda_cars_count_l2064_206480

-- Define the total number of cars
def total_cars : ℕ := 9000

-- Define the percentage of red Honda cars
def red_honda_percentage : ℚ := 90 / 100

-- Define the percentage of total red cars
def total_red_percentage : ℚ := 60 / 100

-- Define the percentage of red non-Honda cars
def red_non_honda_percentage : ℚ := 225 / 1000

-- Theorem statement
theorem honda_cars_count (honda_cars : ℕ) :
  (honda_cars : ℚ) * red_honda_percentage + 
  ((total_cars - honda_cars) : ℚ) * red_non_honda_percentage = 
  (total_cars : ℚ) * total_red_percentage →
  honda_cars = 5000 := by
sorry

end honda_cars_count_l2064_206480


namespace shirley_cases_needed_l2064_206403

-- Define the number of boxes sold
def boxes_sold : ℕ := 54

-- Define the number of boxes per case
def boxes_per_case : ℕ := 6

-- Define the number of cases needed
def cases_needed : ℕ := boxes_sold / boxes_per_case

-- Theorem statement
theorem shirley_cases_needed : cases_needed = 9 := by
  sorry

end shirley_cases_needed_l2064_206403


namespace congruence_solution_l2064_206450

theorem congruence_solution : ∃ n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -2187 [ZMOD 10] ∧ n = 3 := by
  sorry

end congruence_solution_l2064_206450


namespace f_positive_when_x_positive_smallest_a_for_g_inequality_l2064_206485

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x) - (2 * x) / (x + 2)

noncomputable def g (x : ℝ) : ℝ := f x - 4 / (x + 2)

theorem f_positive_when_x_positive (x : ℝ) (hx : x > 0) : f x > 0 := by
  sorry

theorem smallest_a_for_g_inequality : 
  ∀ a : ℝ, (∀ x : ℝ, x > 0 → g x < x + a) ↔ a > -2 := by
  sorry

end f_positive_when_x_positive_smallest_a_for_g_inequality_l2064_206485


namespace remainder_of_large_number_l2064_206412

theorem remainder_of_large_number (p : Nat) (h_prime : Nat.Prime p) :
  123456789012 ≡ 71 [MOD p] :=
by
  sorry

end remainder_of_large_number_l2064_206412


namespace mythical_zoo_count_l2064_206499

theorem mythical_zoo_count (total_heads : ℕ) (total_legs : ℕ) : 
  total_heads = 300 → 
  total_legs = 798 → 
  ∃ (two_legged three_legged : ℕ), 
    two_legged + three_legged = total_heads ∧ 
    2 * two_legged + 3 * three_legged = total_legs ∧ 
    two_legged = 102 := by
sorry

end mythical_zoo_count_l2064_206499


namespace shirt_cost_problem_l2064_206474

theorem shirt_cost_problem (total_cost : ℕ) (num_shirts : ℕ) (num_known_shirts : ℕ) (known_shirt_cost : ℕ) (h1 : total_cost = 85) (h2 : num_shirts = 5) (h3 : num_known_shirts = 3) (h4 : known_shirt_cost = 15) :
  let remaining_shirts := num_shirts - num_known_shirts
  let remaining_cost := total_cost - (num_known_shirts * known_shirt_cost)
  remaining_cost / remaining_shirts = 20 := by
sorry

end shirt_cost_problem_l2064_206474


namespace sequence_length_is_602_l2064_206441

/-- The number of terms in an arithmetic sequence -/
def arithmetic_sequence_length (a₁ aₙ d : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

/-- Theorem: The number of terms in the specified arithmetic sequence is 602 -/
theorem sequence_length_is_602 :
  arithmetic_sequence_length 3 3008 5 = 602 := by
  sorry

end sequence_length_is_602_l2064_206441


namespace crabapple_recipients_count_l2064_206470

/-- The number of students in Mrs. Crabapple's class -/
def num_students : ℕ := 12

/-- The number of times the class meets in a week -/
def class_meetings_per_week : ℕ := 5

/-- The number of possible sequences of crabapple recipients in a week -/
def crabapple_sequences : ℕ := num_students ^ class_meetings_per_week

/-- Theorem stating the number of possible sequences of crabapple recipients -/
theorem crabapple_recipients_count : crabapple_sequences = 248832 := by
  sorry

end crabapple_recipients_count_l2064_206470


namespace squares_in_figure_50_l2064_206496

/-- The number of squares in figure n -/
def f (n : ℕ) : ℕ := 2 * n^2 + 4 * n + 2

/-- The sequence satisfies the given initial conditions -/
axiom initial_conditions :
  f 0 = 2 ∧ f 1 = 8 ∧ f 2 = 18 ∧ f 3 = 32

/-- The number of squares in figure 50 is 5202 -/
theorem squares_in_figure_50 : f 50 = 5202 := by
  sorry

end squares_in_figure_50_l2064_206496


namespace sphere_volume_from_surface_area_l2064_206482

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
    r > 0 → 
    4 * Real.pi * r^2 = 36 * Real.pi → 
    (4 / 3) * Real.pi * r^3 = 36 * Real.pi := by
  sorry

end sphere_volume_from_surface_area_l2064_206482


namespace total_free_sides_length_l2064_206460

/-- A rectangular table with one side against a wall -/
structure RectangularTable where
  /-- Length of the side opposite the wall -/
  opposite_side : ℝ
  /-- Length of each of the other two free sides -/
  adjacent_side : ℝ
  /-- The side opposite the wall is twice the length of each adjacent side -/
  opposite_twice_adjacent : opposite_side = 2 * adjacent_side
  /-- The area of the table is 128 square feet -/
  area_is_128 : opposite_side * adjacent_side = 128

/-- The total length of the table's free sides is 32 feet -/
theorem total_free_sides_length (table : RectangularTable) :
  table.opposite_side + 2 * table.adjacent_side = 32 := by
  sorry

end total_free_sides_length_l2064_206460
