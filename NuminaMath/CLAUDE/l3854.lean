import Mathlib

namespace negation_of_existence_proposition_l3854_385417

theorem negation_of_existence_proposition :
  (¬ ∃ x : ℝ, x > 0 ∧ Real.log x = x - 1) ↔ (∀ x : ℝ, x > 0 → Real.log x ≠ x - 1) := by
  sorry

end negation_of_existence_proposition_l3854_385417


namespace lizzy_final_amount_l3854_385434

/-- Calculates the total amount Lizzy will have after loans are returned with interest -/
def lizzys_total_amount (initial_amount : ℝ) (alice_loan : ℝ) (bob_loan : ℝ) 
  (alice_interest_rate : ℝ) (bob_interest_rate : ℝ) : ℝ :=
  initial_amount - alice_loan - bob_loan + 
  alice_loan * (1 + alice_interest_rate) + 
  bob_loan * (1 + bob_interest_rate)

/-- Theorem stating that Lizzy will have $52.75 after loans are returned -/
theorem lizzy_final_amount : 
  lizzys_total_amount 50 25 20 0.15 0.20 = 52.75 := by
  sorry

end lizzy_final_amount_l3854_385434


namespace binomial_9_5_l3854_385482

theorem binomial_9_5 : Nat.choose 9 5 = 126 := by
  sorry

end binomial_9_5_l3854_385482


namespace eight_people_seating_theorem_l3854_385402

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def seating_arrangements (total_people : ℕ) (restricted_people : ℕ) : ℕ :=
  factorial total_people - factorial (total_people - restricted_people + 1) * factorial restricted_people

theorem eight_people_seating_theorem :
  seating_arrangements 8 3 = 36000 :=
by sorry

end eight_people_seating_theorem_l3854_385402


namespace sqrt_product_equality_l3854_385466

theorem sqrt_product_equality : 2 * Real.sqrt 3 * (3 * Real.sqrt 2) = 6 * Real.sqrt 6 := by
  sorry

end sqrt_product_equality_l3854_385466


namespace ratio_AD_BC_l3854_385440

-- Define the points
variable (A B C D : ℝ × ℝ)

-- Define the conditions
def is_equilateral (A B C : ℝ × ℝ) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

def is_right_triangle (B C D : ℝ × ℝ) : Prop :=
  (C.1 - B.1) * (D.1 - B.1) + (C.2 - B.2) * (D.2 - B.2) = 0

def BC_twice_BD (B C D : ℝ × ℝ) : Prop :=
  dist B C = 2 * dist B D

-- Theorem statement
theorem ratio_AD_BC (A B C D : ℝ × ℝ) 
  (h1 : is_equilateral A B C)
  (h2 : is_right_triangle B C D)
  (h3 : BC_twice_BD B C D) :
  dist A D / dist B C = 3/2 :=
sorry

end ratio_AD_BC_l3854_385440


namespace circle_area_equality_l3854_385495

theorem circle_area_equality (r₁ r₂ : ℝ) (h₁ : r₁ = 35) (h₂ : r₂ = 25) :
  ∃ r₃ : ℝ, r₃ = 10 * Real.sqrt 6 ∧ π * r₃^2 = π * (r₁^2 - r₂^2) := by
  sorry

end circle_area_equality_l3854_385495


namespace greatest_divisor_four_consecutive_integers_l3854_385425

theorem greatest_divisor_four_consecutive_integers :
  ∀ n : ℕ, n > 0 →
  ∃ m : ℕ, m > 24 ∧ ¬(m ∣ (n * (n + 1) * (n + 2) * (n + 3))) ∧
  ∀ k : ℕ, k ≤ 24 → (k ∣ (n * (n + 1) * (n + 2) * (n + 3))) :=
by sorry

end greatest_divisor_four_consecutive_integers_l3854_385425


namespace solution_set_when_a_is_2_range_of_a_when_f_geq_4_l3854_385491

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| + 2 * |x - a|

-- Part 1
theorem solution_set_when_a_is_2 :
  {x : ℝ | f 2 x ≤ x + 4} = {x : ℝ | 1/2 ≤ x ∧ x ≤ 7/2} := by sorry

-- Part 2
theorem range_of_a_when_f_geq_4 :
  {a : ℝ | ∀ x, f a x ≥ 4} = {a : ℝ | a ≤ -5 ∨ a ≥ 3} := by sorry

end solution_set_when_a_is_2_range_of_a_when_f_geq_4_l3854_385491


namespace halfway_fraction_l3854_385474

theorem halfway_fraction : 
  let a := (3 : ℚ) / 4
  let b := (5 : ℚ) / 7
  let midpoint := (a + b) / 2
  midpoint = (41 : ℚ) / 56 := by sorry

end halfway_fraction_l3854_385474


namespace soda_price_increase_l3854_385452

/-- Proves that the percentage increase in the price of a can of soda is 50% given the specified conditions. -/
theorem soda_price_increase (initial_combined_price new_candy_price new_soda_price candy_increase : ℝ) 
  (h1 : initial_combined_price = 16)
  (h2 : new_candy_price = 15)
  (h3 : new_soda_price = 6)
  (h4 : candy_increase = 25)
  : (new_soda_price - (initial_combined_price - new_candy_price / (1 + candy_increase / 100))) / 
    (initial_combined_price - new_candy_price / (1 + candy_increase / 100)) * 100 = 50 := by
  sorry

#check soda_price_increase

end soda_price_increase_l3854_385452


namespace broccoli_carrot_calorie_ratio_l3854_385416

/-- The number of calories in a pound of carrots -/
def carrot_calories : ℕ := 51

/-- The number of pounds of carrots Tom eats -/
def carrot_pounds : ℕ := 1

/-- The number of pounds of broccoli Tom eats -/
def broccoli_pounds : ℕ := 2

/-- The total number of calories Tom ate -/
def total_calories : ℕ := 85

/-- The number of calories in a pound of broccoli -/
def broccoli_calories : ℚ := (total_calories - carrot_calories * carrot_pounds) / broccoli_pounds

theorem broccoli_carrot_calorie_ratio :
  broccoli_calories / carrot_calories = 1 / 3 := by
  sorry

end broccoli_carrot_calorie_ratio_l3854_385416


namespace fraction_to_decimal_l3854_385455

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l3854_385455


namespace vector_difference_magnitude_l3854_385405

/-- Given vectors a and b in R^2, prove that their difference has magnitude 5 -/
theorem vector_difference_magnitude (a b : ℝ × ℝ) : 
  a = (2, 1) → b = (-2, 4) → ‖a - b‖ = 5 := by
  sorry

end vector_difference_magnitude_l3854_385405


namespace parallel_vectors_imply_k_equals_five_l3854_385411

/-- Given vectors a, b, and c in ℝ², prove that if (a - c) is parallel to b, then k = 5. -/
theorem parallel_vectors_imply_k_equals_five :
  let a : Fin 2 → ℝ := ![3, 1]
  let b : Fin 2 → ℝ := ![1, 3]
  let c : Fin 2 → ℝ := ![k, 7]
  (∃ (t : ℝ), (a - c) = t • b) → k = 5 := by
  sorry

end parallel_vectors_imply_k_equals_five_l3854_385411


namespace determinant_2x2_matrix_l3854_385430

theorem determinant_2x2_matrix (x : ℝ) :
  Matrix.det !![5, x; -3, 9] = 45 + 3 * x := by sorry

end determinant_2x2_matrix_l3854_385430


namespace pineapples_theorem_l3854_385435

/-- Calculates the number of fresh pineapples left in a store. -/
def fresh_pineapples_left (initial : ℕ) (sold : ℕ) (rotten : ℕ) : ℕ :=
  initial - sold - rotten

/-- Proves that the number of fresh pineapples left is 29. -/
theorem pineapples_theorem :
  fresh_pineapples_left 86 48 9 = 29 := by
  sorry

end pineapples_theorem_l3854_385435


namespace perpendicular_parallel_relations_l3854_385462

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)

variable (α : Plane) (a b : Line)

-- State the theorem
theorem perpendicular_parallel_relations :
  (∀ a b : Line, ∀ α : Plane,
    (parallel a b ∧ perpendicular_line_plane a α → perpendicular_line_plane b α)) ∧
  (∀ a b : Line, ∀ α : Plane,
    (perpendicular_line_plane a α ∧ perpendicular_line_plane b α → parallel a b)) :=
sorry

end perpendicular_parallel_relations_l3854_385462


namespace expression_factorization_l3854_385470

theorem expression_factorization (x : ℝ) : 
  (15 * x^3 + 80 * x - 5) - (-4 * x^3 + 4 * x - 5) = 19 * x * (x^2 + 4) := by
  sorry

end expression_factorization_l3854_385470


namespace perfect_squares_l3854_385415

theorem perfect_squares (k : ℕ) (h1 : k > 0) (h2 : ∃ a : ℕ, k * (k + 1) = 3 * a^2) : 
  (∃ m : ℕ, k = 3 * m^2) ∧ (∃ n : ℕ, k + 1 = n^2) := by
  sorry

end perfect_squares_l3854_385415


namespace expression_equality_l3854_385499

theorem expression_equality : 5 * 401 + 4 * 401 + 3 * 401 + 400 = 5212 := by
  sorry

end expression_equality_l3854_385499


namespace imaginary_part_of_z_l3854_385432

theorem imaginary_part_of_z (i : ℂ) (z : ℂ) : 
  i * i = -1 → (1 : ℂ) + i = z * ((1 : ℂ) - i) → Complex.im z = 1 := by
  sorry

end imaginary_part_of_z_l3854_385432


namespace parabola_intersection_l3854_385443

/-- The parabola y = x^2 - 2x - 3 intersects the x-axis at (-1, 0) and (3, 0) -/
theorem parabola_intersection (x : ℝ) :
  let y := x^2 - 2*x - 3
  (y = 0 ∧ x = -1) ∨ (y = 0 ∧ x = 3) :=
by sorry

end parabola_intersection_l3854_385443


namespace negation_existence_gt_one_l3854_385424

theorem negation_existence_gt_one :
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) := by sorry

end negation_existence_gt_one_l3854_385424


namespace mary_overtime_rate_increase_l3854_385446

/-- Calculates the percentage increase in overtime rate compared to regular rate -/
def overtime_rate_increase (max_hours : ℕ) (regular_hours : ℕ) (regular_rate : ℚ) (max_earnings : ℚ) : ℚ :=
  let overtime_hours := max_hours - regular_hours
  let regular_earnings := regular_hours * regular_rate
  let overtime_earnings := max_earnings - regular_earnings
  let overtime_rate := overtime_earnings / overtime_hours
  ((overtime_rate - regular_rate) / regular_rate) * 100

/-- The percentage increase in overtime rate for Mary's work schedule -/
theorem mary_overtime_rate_increase :
  overtime_rate_increase 80 20 8 760 = 25 := by
  sorry

end mary_overtime_rate_increase_l3854_385446


namespace james_total_toys_l3854_385451

/-- The number of toy cars James buys -/
def toy_cars : ℕ := 20

/-- The number of toy soldiers James buys -/
def toy_soldiers : ℕ := 2 * toy_cars

/-- The total number of toys James buys -/
def total_toys : ℕ := toy_cars + toy_soldiers

theorem james_total_toys : total_toys = 60 := by sorry

end james_total_toys_l3854_385451


namespace staff_pizza_fraction_l3854_385409

theorem staff_pizza_fraction (teachers : ℕ) (staff : ℕ) (teacher_pizza_fraction : ℚ) (non_pizza_eaters : ℕ) :
  teachers = 30 →
  staff = 45 →
  teacher_pizza_fraction = 2/3 →
  non_pizza_eaters = 19 →
  (staff - (non_pizza_eaters - (teachers - teacher_pizza_fraction * teachers))) / staff = 4/5 :=
by sorry

end staff_pizza_fraction_l3854_385409


namespace vector_magnitude_problem_l3854_385467

def problem (b : ℝ × ℝ) : Prop :=
  let a : ℝ × ℝ := (2, 1)
  (a.1 + b.1)^2 + (a.2 + b.2)^2 = 4^2 ∧ 
  a.1 * b.1 + a.2 * b.2 = 1 →
  b.1^2 + b.2^2 = 3^2

theorem vector_magnitude_problem : ∀ b : ℝ × ℝ, problem b :=
  sorry

end vector_magnitude_problem_l3854_385467


namespace odd_sum_is_odd_l3854_385439

theorem odd_sum_is_odd (a b : ℤ) (ha : Odd a) (hb : Odd b) : Odd (a + 2*b + 1) := by
  sorry

end odd_sum_is_odd_l3854_385439


namespace radical_simplification_l3854_385403

theorem radical_simplification (y : ℝ) (h : y ≥ 0) :
  Real.sqrt (45 * y) * Real.sqrt (18 * y) * Real.sqrt (22 * y) = 18 * y * Real.sqrt (55 * y) := by
  sorry

end radical_simplification_l3854_385403


namespace max_pens_sold_is_226_l3854_385422

/-- Represents the store's promotional sale --/
structure PromotionalSale where
  penProfit : ℕ            -- Profit per pen in yuan
  teddyBearCost : ℕ        -- Cost of teddy bear in yuan
  pensPerPackage : ℕ       -- Number of pens in a promotional package
  totalProfit : ℕ          -- Total profit from the promotion in yuan

/-- Calculates the maximum number of pens sold during the promotional sale --/
def maxPensSold (sale : PromotionalSale) : ℕ :=
  sorry

/-- Theorem stating that for the given promotional sale conditions, 
    the maximum number of pens sold is 226 --/
theorem max_pens_sold_is_226 :
  let sale : PromotionalSale := {
    penProfit := 9
    teddyBearCost := 2
    pensPerPackage := 4
    totalProfit := 1922
  }
  maxPensSold sale = 226 := by
  sorry

end max_pens_sold_is_226_l3854_385422


namespace almond_butter_servings_l3854_385444

/-- Represents a mixed number as a whole number part and a fraction part -/
structure MixedNumber where
  whole : ℕ
  numerator : ℕ
  denominator : ℕ
  denominator_pos : denominator > 0

/-- Converts a mixed number to a rational number -/
def mixedNumberToRational (m : MixedNumber) : ℚ :=
  m.whole + (m.numerator : ℚ) / m.denominator

theorem almond_butter_servings 
  (container_amount : MixedNumber) 
  (serving_size : ℚ) 
  (h1 : container_amount = ⟨37, 2, 3, by norm_num⟩) 
  (h2 : serving_size = 3) : 
  ∃ (result : MixedNumber), 
    mixedNumberToRational result = 
      mixedNumberToRational container_amount / serving_size ∧
    result = ⟨12, 5, 9, by norm_num⟩ := by
  sorry

end almond_butter_servings_l3854_385444


namespace candle_equality_l3854_385472

/-- Represents the number of times each candle is used over n Sundays -/
def total_usage (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents the number of times each individual candle is used -/
def individual_usage (n : ℕ) : ℚ := (n + 1) / 2

/-- Theorem stating that for all candles to be of equal length after n Sundays,
    n must be a positive odd integer -/
theorem candle_equality (n : ℕ) (h : n > 0) :
  (∀ (i : ℕ), i ≤ n → (individual_usage n).num % (individual_usage n).den = 0) ↔
  n % 2 = 1 :=
sorry

end candle_equality_l3854_385472


namespace triangle_sum_zero_l3854_385494

theorem triangle_sum_zero (a b c : ℝ) 
  (ha : |a| ≥ |b + c|) 
  (hb : |b| ≥ |c + a|) 
  (hc : |c| ≥ |a + b|) : 
  a + b + c = 0 := by
sorry

end triangle_sum_zero_l3854_385494


namespace geometric_sequence_sum_l3854_385465

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 1 →  -- given condition
  4 * a 2 - 2 * a 3 = 2 * a 3 - a 4 →  -- arithmetic sequence condition
  a 2 + a 3 + a 4 = 14 := by
sorry

end geometric_sequence_sum_l3854_385465


namespace probability_theorem_l3854_385478

def total_containers : ℕ := 14
def dry_soil_containers : ℕ := 6
def selected_containers : ℕ := 5
def desired_dry_containers : ℕ := 3

def probability_dry_soil : ℚ :=
  (Nat.choose dry_soil_containers desired_dry_containers *
   Nat.choose (total_containers - dry_soil_containers) (selected_containers - desired_dry_containers)) /
  Nat.choose total_containers selected_containers

theorem probability_theorem :
  probability_dry_soil = 560 / 2002 :=
sorry

end probability_theorem_l3854_385478


namespace six_digit_number_puzzle_l3854_385448

theorem six_digit_number_puzzle : ∃! n : ℕ,
  100000 ≤ n ∧ n < 1000000 ∧
  n % 10 = 7 ∧
  7 * 100000 + n / 10 = 5 * n ∧
  n = 142857 := by
  sorry

end six_digit_number_puzzle_l3854_385448


namespace birdhouse_volume_difference_l3854_385426

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℚ := 12

/-- Volume of a rectangular prism -/
def volume (width height depth : ℚ) : ℚ := width * height * depth

/-- Sara's birdhouse dimensions in feet -/
def sara_width : ℚ := 1
def sara_height : ℚ := 2
def sara_depth : ℚ := 2

/-- Jake's birdhouse dimensions in inches -/
def jake_width : ℚ := 16
def jake_height : ℚ := 20
def jake_depth : ℚ := 18

/-- Theorem stating the difference in volume between Sara's and Jake's birdhouses -/
theorem birdhouse_volume_difference :
  volume (sara_width * feet_to_inches) (sara_height * feet_to_inches) (sara_depth * feet_to_inches) -
  volume jake_width jake_height jake_depth = 1152 := by
  sorry

end birdhouse_volume_difference_l3854_385426


namespace speed_ratio_l3854_385475

/-- The speed of object A -/
def v_A : ℝ := sorry

/-- The speed of object B -/
def v_B : ℝ := sorry

/-- The distance B is initially short of O -/
def initial_distance : ℝ := 600

/-- The time when A and B are first equidistant from O -/
def t1 : ℝ := 3

/-- The time when A and B are again equidistant from O -/
def t2 : ℝ := 12

/-- The theorem stating the ratio of speeds -/
theorem speed_ratio : v_A / v_B = 2 / 3 := by
  sorry

end speed_ratio_l3854_385475


namespace same_color_choices_l3854_385421

theorem same_color_choices (m : ℕ) : 
  let total_objects := 2 * m
  let red_objects := m
  let blue_objects := m
  (number_of_ways_to_choose_same_color : ℕ) = 2 :=
by
  sorry

end same_color_choices_l3854_385421


namespace expression_evaluation_l3854_385418

theorem expression_evaluation : 
  let expr := 125 - 25 * 4
  expr = 25 := by sorry

#check expression_evaluation

end expression_evaluation_l3854_385418


namespace cube_edge_length_in_pyramid_l3854_385476

/-- Represents a pyramid with an equilateral triangular base -/
structure EquilateralPyramid where
  base_side_length : ℝ
  apex_height : ℝ

/-- Represents a cube -/
structure Cube where
  edge_length : ℝ

/-- Theorem stating the edge length of the cube in the given pyramid configuration -/
theorem cube_edge_length_in_pyramid (p : EquilateralPyramid) (c : Cube) 
  (h1 : p.base_side_length = 3)
  (h2 : p.apex_height = 9)
  (h3 : c.edge_length * Real.sqrt 3 = p.apex_height) : 
  c.edge_length = 3 := by
  sorry


end cube_edge_length_in_pyramid_l3854_385476


namespace square_brush_ratio_l3854_385449

/-- A square with side length s and a brush of width w -/
structure SquareAndBrush where
  s : ℝ
  w : ℝ

/-- The painted area is one-third of the square's area -/
def paintedAreaIsOneThird (sb : SquareAndBrush) : Prop :=
  (1/2 * sb.w^2 + (sb.s - sb.w)^2 / 2) = sb.s^2 / 3

/-- The theorem to be proved -/
theorem square_brush_ratio (sb : SquareAndBrush) 
    (h : paintedAreaIsOneThird sb) : 
    sb.s / sb.w = 3 + Real.sqrt 3 := by
  sorry

end square_brush_ratio_l3854_385449


namespace kieras_envelopes_l3854_385433

theorem kieras_envelopes :
  ∀ (yellow : ℕ),
  let blue := 14
  let green := 3 * yellow
  yellow < blue →
  blue + yellow + green = 46 →
  blue - yellow = 8 :=
by
  sorry

end kieras_envelopes_l3854_385433


namespace opposite_of_negative_five_l3854_385420

theorem opposite_of_negative_five : 
  (-(- 5 : ℤ)) = (5 : ℤ) := by sorry

end opposite_of_negative_five_l3854_385420


namespace binary_110010_is_50_l3854_385487

def binary_to_decimal (b : List Bool) : ℕ :=
  (List.enumFrom 0 b).foldl (fun acc (i, x) => acc + if x then 2^i else 0) 0

def binary_110010 : List Bool := [false, true, false, false, true, true]

theorem binary_110010_is_50 : binary_to_decimal binary_110010 = 50 := by
  sorry

end binary_110010_is_50_l3854_385487


namespace sector_area_l3854_385456

/-- Given a sector with central angle 7/(2π) and arc length 7, its area is 7π. -/
theorem sector_area (central_angle : Real) (arc_length : Real) (area : Real) :
  central_angle = 7 / (2 * Real.pi) →
  arc_length = 7 →
  area = 7 * Real.pi :=
by
  sorry

#check sector_area

end sector_area_l3854_385456


namespace min_value_theorem_min_value_achieved_l3854_385488

theorem min_value_theorem (x : ℝ) (h : x > 4) :
  (x + 15) / Real.sqrt (x - 4) ≥ 2 * Real.sqrt 19 :=
by sorry

theorem min_value_achieved (x : ℝ) (h : x > 4) :
  ∃ x₀ > 4, (x₀ + 15) / Real.sqrt (x₀ - 4) = 2 * Real.sqrt 19 :=
by sorry

end min_value_theorem_min_value_achieved_l3854_385488


namespace james_net_income_l3854_385441

def regular_price : ℝ := 20
def discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.05
def maintenance_fee : ℝ := 35
def insurance_fee : ℝ := 15

def monday_hours : ℝ := 8
def wednesday_hours : ℝ := 8
def friday_hours : ℝ := 6
def sunday_hours : ℝ := 5

def total_hours : ℝ := monday_hours + wednesday_hours + friday_hours + sunday_hours
def rental_days : ℕ := 4

def discounted_rental : Bool := rental_days ≥ 3

theorem james_net_income :
  let total_rental_income := total_hours * regular_price
  let discounted_income := if discounted_rental then total_rental_income * (1 - discount_rate) else total_rental_income
  let income_with_tax := discounted_income * (1 + sales_tax_rate)
  let total_expenses := maintenance_fee + (insurance_fee * rental_days)
  let net_income := income_with_tax - total_expenses
  net_income = 415.30 := by sorry

end james_net_income_l3854_385441


namespace f_bound_iff_m_range_l3854_385427

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.exp x + x^2 / m^2 - x

theorem f_bound_iff_m_range (m : ℝ) (hm : m ≠ 0) :
  (∀ a b : ℝ, a ∈ Set.Icc (-1) 1 → b ∈ Set.Icc (-1) 1 → |f m a - f m b| ≤ Real.exp 1) ↔
  m ∈ Set.Iic (-Real.sqrt 2 / 2) ∪ Set.Ici (Real.sqrt 2 / 2) :=
sorry

end f_bound_iff_m_range_l3854_385427


namespace drum_capacity_ratio_l3854_385473

theorem drum_capacity_ratio (c_x c_y : ℝ) : 
  c_x > 0 → c_y > 0 →
  (1/2 * c_x + 1/2 * c_y = 3/4 * c_y) →
  c_y / c_x = 2 := by
sorry

end drum_capacity_ratio_l3854_385473


namespace white_surface_fraction_is_half_l3854_385436

/-- Represents a cube constructed from smaller cubes -/
structure CompositeCube where
  edge_length : ℕ
  total_small_cubes : ℕ
  white_cubes : ℕ
  black_cubes : ℕ

/-- Calculates the fraction of white surface area for a composite cube -/
def white_surface_fraction (c : CompositeCube) : ℚ :=
  sorry

/-- The specific composite cube from the problem -/
def problem_cube : CompositeCube :=
  { edge_length := 4
  , total_small_cubes := 64
  , white_cubes := 48
  , black_cubes := 16 }

theorem white_surface_fraction_is_half :
  white_surface_fraction problem_cube = 1/2 :=
sorry

end white_surface_fraction_is_half_l3854_385436


namespace rachel_assembly_time_l3854_385493

/-- Calculates the total time taken to assemble furniture -/
def total_assembly_time (num_chairs : ℕ) (num_tables : ℕ) (time_per_piece : ℕ) : ℕ :=
  (num_chairs + num_tables) * time_per_piece

/-- Proves that the total assembly time for Rachel's furniture is 40 minutes -/
theorem rachel_assembly_time :
  total_assembly_time 7 3 4 = 40 := by
  sorry

end rachel_assembly_time_l3854_385493


namespace bill_difference_is_18_l3854_385413

-- Define the tip percentages and amount
def mike_tip_percent : ℚ := 15 / 100
def joe_tip_percent : ℚ := 25 / 100
def anna_tip_percent : ℚ := 10 / 100
def tip_amount : ℚ := 3

-- Define the bills as functions of the tip percentage
def bill (tip_percent : ℚ) : ℚ := tip_amount / tip_percent

-- Theorem statement
theorem bill_difference_is_18 :
  let mike_bill := bill mike_tip_percent
  let joe_bill := bill joe_tip_percent
  let anna_bill := bill anna_tip_percent
  let max_bill := max mike_bill (max joe_bill anna_bill)
  let min_bill := min mike_bill (min joe_bill anna_bill)
  max_bill - min_bill = 18 := by sorry

end bill_difference_is_18_l3854_385413


namespace inequality_solution_set_l3854_385429

open Set Real

theorem inequality_solution_set : 
  let S := {x : ℝ | (π/2)^((x-1)^2) ≤ (2/π)^(x^2-5*x-5)}
  S = Icc (-1/2) 4 := by
sorry

end inequality_solution_set_l3854_385429


namespace median_of_100_numbers_l3854_385485

def is_median (s : Finset ℕ) (m : ℕ) : Prop :=
  2 * (s.filter (· < m)).card ≤ s.card ∧ 2 * (s.filter (· > m)).card ≤ s.card

theorem median_of_100_numbers (s : Finset ℕ) (h_card : s.card = 100) :
  (∃ x ∈ s, is_median (s.erase x) 78) →
  (∃ y ∈ s, y ≠ x → is_median (s.erase y) 66) →
  is_median s 72 :=
sorry

end median_of_100_numbers_l3854_385485


namespace polynomial_leading_coefficient_l3854_385445

/-- A polynomial g satisfying g(x + 1) - g(x) = 12x + 2 for all x has leading coefficient 6 -/
theorem polynomial_leading_coefficient (g : ℝ → ℝ) :
  (∀ x, g (x + 1) - g x = 12 * x + 2) →
  ∃ c, ∀ x, g x = 6 * x^2 - 4 * x + c :=
sorry

end polynomial_leading_coefficient_l3854_385445


namespace tan_45_degrees_l3854_385469

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end tan_45_degrees_l3854_385469


namespace beth_sold_coins_l3854_385459

-- Define the initial number of coins Beth had
def initial_coins : ℕ := 125

-- Define the number of coins Carl gave to Beth
def gifted_coins : ℕ := 35

-- Define the total number of coins Beth had after receiving the gift
def total_coins : ℕ := initial_coins + gifted_coins

-- Define the number of coins Beth sold (half of her total coins)
def sold_coins : ℕ := total_coins / 2

-- Theorem stating that the number of coins Beth sold is equal to 80
theorem beth_sold_coins : sold_coins = 80 := by
  sorry

end beth_sold_coins_l3854_385459


namespace sector_area_l3854_385400

theorem sector_area (α : Real) (r : Real) (h1 : α = 150 * π / 180) (h2 : r = Real.sqrt 3) :
  (α * r^2) / 2 = 5 * π / 4 := by
  sorry

end sector_area_l3854_385400


namespace a_more_stable_than_b_l3854_385423

/-- Represents a shooter with their shooting variance -/
structure Shooter where
  name : String
  variance : ℝ
  variance_nonneg : 0 ≤ variance

/-- Definition of more stable shooting performance -/
def more_stable (a b : Shooter) : Prop :=
  a.variance < b.variance

/-- Theorem stating that shooter A has more stable performance than B -/
theorem a_more_stable_than_b :
  let a : Shooter := ⟨"A", 0.12, by norm_num⟩
  let b : Shooter := ⟨"B", 0.6, by norm_num⟩
  more_stable a b := by
  sorry


end a_more_stable_than_b_l3854_385423


namespace intersection_difference_l3854_385408

def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 6
def parabola2 (x : ℝ) : ℝ := -2 * x^2 - 4 * x + 6

theorem intersection_difference :
  ∃ (a b c d : ℝ),
    (parabola1 a = parabola2 a) ∧
    (parabola1 c = parabola2 c) ∧
    (c ≥ a) ∧
    (c - a = 2/5) :=
by sorry

end intersection_difference_l3854_385408


namespace range_of_b_minus_a_l3854_385447

theorem range_of_b_minus_a (a b : ℝ) : 
  (a < b) →
  (∀ x : ℝ, (a ≤ x ∧ x ≤ b) → (x^2 + x - 2 ≤ 0)) →
  (∃ x : ℝ, (x^2 + x - 2 ≤ 0) ∧ ¬(a ≤ x ∧ x ≤ b)) →
  (0 < b - a) ∧ (b - a < 3) := by
sorry

end range_of_b_minus_a_l3854_385447


namespace quadratic_value_l3854_385489

-- Define the quadratic function
def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

-- State the theorem
theorem quadratic_value (p q : ℝ) :
  f p q 1 = 3 → f p q (-3) = 7 → f p q (-5) = 21 := by
  sorry

end quadratic_value_l3854_385489


namespace largest_integer_with_remainder_l3854_385460

theorem largest_integer_with_remainder : ∃ n : ℕ, n < 100 ∧ n % 6 = 4 ∧ ∀ m : ℕ, m < 100 ∧ m % 6 = 4 → m ≤ n := by
  sorry

end largest_integer_with_remainder_l3854_385460


namespace chord_length_l3854_385406

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def are_externally_tangent (c1 c2 : Circle) : Prop := sorry

def is_internally_tangent (c1 c2 : Circle) : Prop := sorry

def are_collinear (p1 p2 p3 : ℝ × ℝ) : Prop := sorry

def is_common_external_tangent (c1 c2 c3 : Circle) (chord : ℝ × ℝ → ℝ × ℝ) : Prop := sorry

-- Theorem statement
theorem chord_length 
  (c1 c2 c3 : Circle)
  (chord : ℝ × ℝ → ℝ × ℝ)
  (h1 : are_externally_tangent c1 c2)
  (h2 : is_internally_tangent c1 c3)
  (h3 : is_internally_tangent c2 c3)
  (h4 : c1.radius = 3)
  (h5 : c2.radius = 9)
  (h6 : are_collinear c1.center c2.center c3.center)
  (h7 : is_common_external_tangent c1 c2 c3 chord) :
  ∃ (a b : ℝ × ℝ), chord a = b ∧ Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 18 :=
sorry

end chord_length_l3854_385406


namespace range_equivalence_l3854_385407

/-- The set of real numbers satisfying the given conditions -/
def A : Set ℝ :=
  {a | ∀ x, (x^2 - 4*a*x + 3*a^2 < 0 → x^2 + 2*x - 8 > 0) ∧
    ∃ y, (y^2 - 4*a*y + 3*a^2 ≥ 0 ∧ y^2 + 2*y - 8 > 0)}

/-- The theorem stating the equivalence of set A and the expected range -/
theorem range_equivalence : A = {a : ℝ | a ≤ -4 ∨ a ≥ 2 ∨ a = 0} := by
  sorry

end range_equivalence_l3854_385407


namespace partial_fraction_decomposition_l3854_385468

theorem partial_fraction_decomposition (M₁ M₂ : ℝ) : 
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 → (27 * x - 19) / (x^2 - 5*x + 6) = M₁ / (x - 2) + M₂ / (x - 3)) →
  M₁ * M₂ = -2170 := by
sorry

end partial_fraction_decomposition_l3854_385468


namespace a_minus_b_value_l3854_385442

theorem a_minus_b_value (a b : ℝ) (ha : |a| = 4) (hb : |b| = 2) (hab : |a + b| = a + b) :
  a - b = 2 ∨ a - b = 6 := by
sorry

end a_minus_b_value_l3854_385442


namespace candy_mix_cost_l3854_385410

/-- Prove the cost of candy B given the mixture conditions -/
theorem candy_mix_cost (total_weight : ℝ) (mix_cost_per_pound : ℝ) 
  (candy_a_cost : ℝ) (candy_a_weight : ℝ) :
  total_weight = 5 →
  mix_cost_per_pound = 2 →
  candy_a_cost = 3.2 →
  candy_a_weight = 1 →
  ∃ (candy_b_cost : ℝ),
    candy_b_cost = 1.7 ∧
    total_weight * mix_cost_per_pound = 
      candy_a_weight * candy_a_cost + (total_weight - candy_a_weight) * candy_b_cost :=
by
  sorry


end candy_mix_cost_l3854_385410


namespace triangle_perimeter_upper_bound_l3854_385497

theorem triangle_perimeter_upper_bound :
  ∀ a b c : ℝ,
  a = 5 →
  b = 19 →
  a + b > c →
  a + c > b →
  b + c > a →
  a + b + c < 48 :=
by
  sorry

end triangle_perimeter_upper_bound_l3854_385497


namespace inequality_implication_l3854_385453

theorem inequality_implication (a b x y : ℝ) (h1 : 1 < a) (h2 : a < b)
  (h3 : a^x + b^y ≤ a^(-x) + b^(-y)) : x + y ≤ 0 := by
  sorry

end inequality_implication_l3854_385453


namespace triangle_height_l3854_385412

theorem triangle_height (base : ℝ) (area : ℝ) (height : ℝ) : 
  base = 4 → area = 10 → area = (base * height) / 2 → height = 5 := by
  sorry

end triangle_height_l3854_385412


namespace tangency_condition_l3854_385483

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = x^2 + 6

/-- The hyperbola equation -/
def hyperbola (m x y : ℝ) : Prop := y^2 - m*x^2 = 6

/-- The tangency condition -/
def are_tangent (m : ℝ) : Prop :=
  ∃ x y : ℝ, parabola x y ∧ hyperbola m x y ∧
  ∀ x' y' : ℝ, parabola x' y' ∧ hyperbola m x' y' → (x = x' ∧ y = y')

/-- The theorem stating the condition for tangency -/
theorem tangency_condition :
  ∀ m : ℝ, are_tangent m ↔ (m = 12 + 10 * Real.sqrt 6 ∨ m = 12 - 10 * Real.sqrt 6) :=
sorry

end tangency_condition_l3854_385483


namespace right_triangle_with_specific_perimeter_l3854_385486

theorem right_triangle_with_specific_perimeter :
  ∃ (b c : ℤ), 
    b = 7 ∧ 
    c = 5 ∧ 
    (b : ℝ)^2 + (b + c : ℝ)^2 = (b + 2*c : ℝ)^2 ∧ 
    b + (b + c) + (b + 2*c) = 36 :=
by sorry

end right_triangle_with_specific_perimeter_l3854_385486


namespace special_function_property_l3854_385419

/-- A function that is even, has period 2, and is monotonically decreasing on [-3, -2] -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ 
  (∀ x, f (x + 2) = f x) ∧
  (∀ x y, -3 ≤ x ∧ x < y ∧ y ≤ -2 → f y < f x)

/-- Acute angle in a triangle -/
def is_acute_angle (θ : ℝ) : Prop :=
  0 < θ ∧ θ < Real.pi / 2

theorem special_function_property 
  (f : ℝ → ℝ) 
  (h_f : special_function f) 
  (α β : ℝ) 
  (h_α : is_acute_angle α) 
  (h_β : is_acute_angle β) : 
  f (Real.sin α) > f (Real.cos β) := by
    sorry

end special_function_property_l3854_385419


namespace line_slope_intercept_product_l3854_385496

/-- Given a line passing through points (1, 4) and (-2, -2), prove that the product of its slope and y-intercept is 4. -/
theorem line_slope_intercept_product (m b : ℝ) : 
  (4 = m * 1 + b) → 
  (-2 = m * (-2) + b) → 
  m * b = 4 := by sorry

end line_slope_intercept_product_l3854_385496


namespace vacation_duration_l3854_385404

-- Define the parameters
def miles_per_day : ℕ := 250
def total_miles : ℕ := 1250

-- Theorem statement
theorem vacation_duration :
  total_miles / miles_per_day = 5 :=
sorry

end vacation_duration_l3854_385404


namespace smallest_third_altitude_nine_is_achievable_l3854_385477

/-- Represents a triangle with altitudes --/
structure TriangleWithAltitudes where
  /-- The lengths of the three altitudes --/
  altitudes : Fin 3 → ℝ
  /-- At least two altitudes are positive --/
  two_positive : ∃ (i j : Fin 3), i ≠ j ∧ altitudes i > 0 ∧ altitudes j > 0

/-- The proposition to be proved --/
theorem smallest_third_altitude 
  (t : TriangleWithAltitudes) 
  (h1 : t.altitudes 0 = 6) 
  (h2 : t.altitudes 1 = 18) 
  (h3 : ∃ (n : ℕ), t.altitudes 2 = n) :
  t.altitudes 2 ≥ 9 := by
sorry

/-- The proposition that 9 is achievable --/
theorem nine_is_achievable : 
  ∃ (t : TriangleWithAltitudes), 
    t.altitudes 0 = 6 ∧ 
    t.altitudes 1 = 18 ∧ 
    t.altitudes 2 = 9 := by
sorry

end smallest_third_altitude_nine_is_achievable_l3854_385477


namespace fraction_equivalence_l3854_385479

theorem fraction_equivalence (a b k : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hk : k ≠ 0) :
  (k * a) / (k * b) = a / b :=
sorry

end fraction_equivalence_l3854_385479


namespace quadratic_function_satisfies_conditions_l3854_385498

def f (x : ℝ) : ℝ := -2 * x^2 + 12 * x - 10

theorem quadratic_function_satisfies_conditions :
  (f 1 = 0) ∧ (f 5 = 0) ∧ (f 3 = 8) := by
  sorry

end quadratic_function_satisfies_conditions_l3854_385498


namespace zero_point_implies_a_range_l3854_385463

/-- Given a function y = x³ - ax where x ∈ ℝ and y has a zero point at (1, 2),
    prove that a ∈ (1, 4) -/
theorem zero_point_implies_a_range (a : ℝ) :
  (∃ x ∈ Set.Ioo 1 2, x^3 - a*x = 0) →
  a ∈ Set.Ioo 1 4 := by
  sorry

end zero_point_implies_a_range_l3854_385463


namespace train_length_l3854_385414

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (h1 : speed_kmh = 72) (h2 : time_s = 5.999520038396929) :
  ∃ (length_m : ℝ), abs (length_m - 119.99) < 0.01 :=
by
  sorry

end train_length_l3854_385414


namespace pet_ownership_l3854_385438

theorem pet_ownership (total_students : ℕ) 
  (dog_owners cat_owners other_pet_owners : ℕ)
  (no_pet_owners : ℕ)
  (only_dog_owners only_cat_owners only_other_pet_owners : ℕ) :
  total_students = 40 →
  dog_owners = total_students / 2 →
  cat_owners = total_students / 4 →
  other_pet_owners = 8 →
  no_pet_owners = 5 →
  only_dog_owners = 15 →
  only_cat_owners = 4 →
  only_other_pet_owners = 5 →
  ∃ (all_three_pets : ℕ),
    all_three_pets = 1 ∧
    dog_owners = only_dog_owners + (other_pet_owners - only_other_pet_owners) + 
                 (cat_owners - only_cat_owners) - all_three_pets + all_three_pets ∧
    cat_owners = only_cat_owners + (other_pet_owners - only_other_pet_owners) + 
                 (dog_owners - only_dog_owners) - all_three_pets + all_three_pets ∧
    other_pet_owners = only_other_pet_owners + (dog_owners - only_dog_owners) + 
                       (cat_owners - only_cat_owners) - all_three_pets + all_three_pets ∧
    total_students = dog_owners + cat_owners + other_pet_owners - 
                     (dog_owners - only_dog_owners) - (cat_owners - only_cat_owners) - 
                     (other_pet_owners - only_other_pet_owners) + all_three_pets + no_pet_owners :=
by
  sorry

end pet_ownership_l3854_385438


namespace committee_probability_l3854_385431

def total_members : ℕ := 30
def num_boys : ℕ := 12
def num_girls : ℕ := 18
def committee_size : ℕ := 6

theorem committee_probability :
  let total_combinations := Nat.choose total_members committee_size
  let all_boys_combinations := Nat.choose num_boys committee_size
  let all_girls_combinations := Nat.choose num_girls committee_size
  let prob_at_least_one_each := 1 - (all_boys_combinations + all_girls_combinations : ℚ) / total_combinations
  prob_at_least_one_each = 574287 / 593775 := by
  sorry

end committee_probability_l3854_385431


namespace john_twice_james_age_john_twice_james_age_proof_l3854_385490

/-- Proves that John will be twice as old as James in 15 years -/
theorem john_twice_james_age : ℕ → Prop :=
  fun years_until_twice_age : ℕ =>
    let john_current_age : ℕ := 39
    let james_brother_age : ℕ := 16
    let age_difference_james_brother : ℕ := 4
    let james_current_age : ℕ := james_brother_age - age_difference_james_brother
    let john_age_3_years_ago : ℕ := john_current_age - 3
    let james_age_in_future : ℕ → ℕ := fun x => james_current_age + x
    ∃ x : ℕ, john_age_3_years_ago = 2 * (james_age_in_future x) →
    (john_current_age + years_until_twice_age = 2 * (james_current_age + years_until_twice_age)) →
    years_until_twice_age = 15

/-- Proof of the theorem -/
theorem john_twice_james_age_proof : john_twice_james_age 15 := by
  sorry

end john_twice_james_age_john_twice_james_age_proof_l3854_385490


namespace circular_fields_area_comparison_l3854_385458

theorem circular_fields_area_comparison :
  ∀ (r1 r2 : ℝ),
  r1 > 0 → r2 > 0 →
  r2 / r1 = 10 / 4 →
  (π * r2^2 - π * r1^2) / (π * r1^2) * 100 = 525 :=
by
  sorry

end circular_fields_area_comparison_l3854_385458


namespace maya_lift_increase_l3854_385428

/-- Given America's peak lift and Maya's relative lift capacities, calculate the increase in Maya's lift capacity. -/
theorem maya_lift_increase (america_peak : ℝ) (maya_initial_ratio : ℝ) (maya_peak_ratio : ℝ) 
  (h1 : america_peak = 300)
  (h2 : maya_initial_ratio = 1/4)
  (h3 : maya_peak_ratio = 1/2) :
  maya_peak_ratio * america_peak - maya_initial_ratio * america_peak = 75 := by
  sorry

end maya_lift_increase_l3854_385428


namespace degree_to_radian_conversion_l3854_385471

theorem degree_to_radian_conversion :
  ((-300 : ℝ) * (π / 180)) = -(5 / 3 : ℝ) * π :=
by sorry

end degree_to_radian_conversion_l3854_385471


namespace composite_sum_of_powers_l3854_385437

theorem composite_sum_of_powers (a b c d m n : ℕ) 
  (h_pos : 0 < d ∧ d < c ∧ c < b ∧ b < a)
  (h_div : (a + b - c + d) ∣ (a * c + b * d))
  (h_m_pos : 0 < m)
  (h_n_odd : n % 2 = 1) :
  ∃ (k : ℕ), k > 1 ∧ k ∣ (a^n * b^m + c^m * d^n) :=
sorry

end composite_sum_of_powers_l3854_385437


namespace solution_set_f_positive_solution_set_f_leq_g_l3854_385457

-- Define the functions f and g
def f (m : ℝ) (x : ℝ) : ℝ := 3 * x^2 + (4 - m) * x - 6 * m
def g (m : ℝ) (x : ℝ) : ℝ := 2 * x^2 - x - m

-- Part 1: Solution set of f(x) > 0 when m = 1
theorem solution_set_f_positive (x : ℝ) :
  f 1 x > 0 ↔ x < -2 ∨ x > 1 := by sorry

-- Part 2: Solution set of f(x) ≤ g(x) when m > 0
theorem solution_set_f_leq_g (m : ℝ) (x : ℝ) (h : m > 0) :
  f m x ≤ g m x ↔ -5 ≤ x ∧ x ≤ m := by sorry

end solution_set_f_positive_solution_set_f_leq_g_l3854_385457


namespace shelly_money_theorem_l3854_385454

/-- Calculates the total amount of money Shelly has given the number of $10 and $5 bills -/
def total_money (ten_dollar_bills : ℕ) (five_dollar_bills : ℕ) : ℕ :=
  10 * ten_dollar_bills + 5 * five_dollar_bills

/-- Proves that Shelly has $130 in total -/
theorem shelly_money_theorem :
  let ten_dollar_bills : ℕ := 10
  let five_dollar_bills : ℕ := ten_dollar_bills - 4
  total_money ten_dollar_bills five_dollar_bills = 130 := by
  sorry

#check shelly_money_theorem

end shelly_money_theorem_l3854_385454


namespace cube_root_equation_solution_l3854_385401

theorem cube_root_equation_solution :
  ∃ y : ℝ, (5 + 2/y)^(1/3 : ℝ) = 3 ↔ y = 1/11 := by
  sorry

end cube_root_equation_solution_l3854_385401


namespace road_sign_difference_l3854_385450

/-- Represents the number of road signs at each intersection -/
structure RoadSigns where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- The conditions of the road sign problem -/
def roadSignProblem (rs : RoadSigns) : Prop :=
  rs.first = 40 ∧
  rs.second = rs.first + rs.first / 4 ∧
  rs.third = 2 * rs.second ∧
  rs.fourth < rs.third ∧
  rs.first + rs.second + rs.third + rs.fourth = 270

theorem road_sign_difference (rs : RoadSigns) 
  (h : roadSignProblem rs) : rs.third - rs.fourth = 20 := by
  sorry

end road_sign_difference_l3854_385450


namespace locus_of_constant_sum_distances_l3854_385464

-- Define a type for lines in a plane
structure Line where
  -- Add necessary fields to represent a line

-- Define a type for points in a plane
structure Point where
  -- Add necessary fields to represent a point

-- Define a function to calculate the distance between a point and a line
def distance (p : Point) (l : Line) : ℝ :=
  sorry

-- Define a function to check if two lines are parallel
def are_parallel (l1 l2 : Line) : Prop :=
  sorry

-- Define a type for the locus
inductive Locus
  | Region
  | Parallelogram
  | Octagon

-- State the theorem
theorem locus_of_constant_sum_distances 
  (l1 l2 m1 m2 : Line) 
  (h_parallel1 : are_parallel l1 l2) 
  (h_parallel2 : are_parallel m1 m2) 
  (sum : ℝ) :
  ∃ (locus : Locus),
    ∀ (p : Point),
      distance p l1 + distance p l2 + distance p m1 + distance p m2 = sum →
      (((are_parallel l1 m1) ∧ (locus = Locus.Region)) ∨
       ((¬are_parallel l1 m1) ∧ ((locus = Locus.Parallelogram) ∨ (locus = Locus.Octagon)))) :=
by sorry

end locus_of_constant_sum_distances_l3854_385464


namespace vectors_not_coplanar_l3854_385492

def a : ℝ × ℝ × ℝ := (1, -1, 4)
def b : ℝ × ℝ × ℝ := (1, 0, 3)
def c : ℝ × ℝ × ℝ := (1, -3, 8)

def scalar_triple_product (v1 v2 v3 : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := v1
  let (x2, y2, z2) := v2
  let (x3, y3, z3) := v3
  x1 * (y2 * z3 - y3 * z2) - y1 * (x2 * z3 - x3 * z2) + z1 * (x2 * y3 - x3 * y2)

theorem vectors_not_coplanar : scalar_triple_product a b c ≠ 0 := by
  sorry

end vectors_not_coplanar_l3854_385492


namespace sum_even_integers_102_to_200_l3854_385461

/-- The sum of even integers from 102 to 200 inclusive -/
def sum_even_102_to_200 : ℕ := 7550

/-- The sum of the first 50 positive even integers -/
def sum_first_50_even : ℕ := 2550

/-- The number of even integers from 102 to 200 inclusive -/
def num_even_102_to_200 : ℕ := 50

theorem sum_even_integers_102_to_200 :
  sum_even_102_to_200 = (num_even_102_to_200 / 2) * (102 + 200) :=
by sorry

end sum_even_integers_102_to_200_l3854_385461


namespace unique_solution_for_rational_equation_l3854_385481

theorem unique_solution_for_rational_equation :
  let k : ℚ := -3/4
  let f (x : ℚ) := (x + 3)/(k*x - 2) - x
  ∃! x, f x = 0 :=
by sorry

end unique_solution_for_rational_equation_l3854_385481


namespace trajectory_of_Q_l3854_385480

-- Define the line L that point P moves on
def L : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1 - p.2 + 3 = 0}

-- Define the fixed point M
def M : ℝ × ℝ := (-1, 2)

-- Define the property that Q is on the extension of PM and |PM| = |MQ|
def Q_property (P Q : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 1 ∧ Q = (t • (P - M) + M)

-- State the theorem
theorem trajectory_of_Q :
  ∀ Q : ℝ × ℝ, (∃ P ∈ L, Q_property P Q) → 2 * Q.1 - Q.2 + 5 = 0 :=
by sorry

end trajectory_of_Q_l3854_385480


namespace ratio_sum_last_number_l3854_385484

theorem ratio_sum_last_number (a b c : ℕ) : 
  a + b + c = 1000 → 
  5 * b = a → 
  4 * b = c → 
  c = 400 := by
sorry

end ratio_sum_last_number_l3854_385484
