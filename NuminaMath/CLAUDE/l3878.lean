import Mathlib

namespace min_tiles_needed_l3878_387832

/-- Represents the dimensions of a rectangular object -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- Calculates the area of a rectangle in square inches -/
def areaInSquareInches (rect : Rectangle) : ℕ := rect.length * rect.width

/-- Calculates the number of small rectangles needed to cover a larger rectangle -/
def tilesNeeded (smallRect : Rectangle) (largeRect : Rectangle) : ℕ :=
  (areaInSquareInches largeRect) / (areaInSquareInches smallRect)

theorem min_tiles_needed :
  let tile := Rectangle.mk 2 3
  let room := Rectangle.mk (feetToInches 3) (feetToInches 6)
  tilesNeeded tile room = 432 := by sorry

end min_tiles_needed_l3878_387832


namespace fraction_multiplication_cube_l3878_387839

theorem fraction_multiplication_cube : (1 / 2 : ℚ)^3 * (1 / 7 : ℚ) = 1 / 56 := by
  sorry

end fraction_multiplication_cube_l3878_387839


namespace baker_sales_difference_l3878_387835

/-- Represents the baker's sales data --/
structure BakerSales where
  usual_pastries : ℕ
  usual_bread : ℕ
  today_pastries : ℕ
  today_bread : ℕ
  pastry_price : ℕ
  bread_price : ℕ

/-- Calculates the difference between today's sales and average daily sales --/
def sales_difference (s : BakerSales) : ℕ :=
  let usual_total := s.usual_pastries * s.pastry_price + s.usual_bread * s.bread_price
  let today_total := s.today_pastries * s.pastry_price + s.today_bread * s.bread_price
  today_total - usual_total

/-- Theorem stating the difference in sales --/
theorem baker_sales_difference :
  ∃ (s : BakerSales),
    s.usual_pastries = 20 ∧
    s.usual_bread = 10 ∧
    s.today_pastries = 14 ∧
    s.today_bread = 25 ∧
    s.pastry_price = 2 ∧
    s.bread_price = 4 ∧
    sales_difference s = 48 := by
  sorry

end baker_sales_difference_l3878_387835


namespace area_of_smaller_circle_l3878_387850

/-- Given two externally tangent circles with common tangent lines,
    where the tangent segment length is 6 and the radius of the larger circle
    is 3 times that of the smaller circle, prove that the area of the
    smaller circle is 12π/5. -/
theorem area_of_smaller_circle (r : ℝ) : 
  r > 0 →  -- radius of smaller circle is positive
  6^2 + r^2 = (4*r)^2 →  -- Pythagorean theorem applied to the tangent-radius triangle
  π * r^2 = 12*π/5 :=
by sorry

end area_of_smaller_circle_l3878_387850


namespace height_difference_l3878_387897

def pine_height : ℚ := 12 + 4/5
def birch_height : ℚ := 18 + 1/2
def maple_height : ℚ := 14 + 3/5

def tallest_height : ℚ := max (max pine_height birch_height) maple_height
def shortest_height : ℚ := min (min pine_height birch_height) maple_height

theorem height_difference :
  tallest_height - shortest_height = 7 + 7/10 := by sorry

end height_difference_l3878_387897


namespace rotation_150_degrees_l3878_387864

-- Define the shapes
inductive Shape
  | Square
  | Triangle
  | Pentagon

-- Define the positions
inductive Position
  | Top
  | Right
  | Bottom

-- Define the circular arrangement
structure CircularArrangement :=
  (top : Shape)
  (right : Shape)
  (bottom : Shape)

-- Define the rotation function
def rotate150 (arr : CircularArrangement) : CircularArrangement :=
  { top := arr.right
  , right := arr.bottom
  , bottom := arr.top }

-- Theorem statement
theorem rotation_150_degrees (initial : CircularArrangement) 
  (h1 : initial.top = Shape.Square)
  (h2 : initial.right = Shape.Triangle)
  (h3 : initial.bottom = Shape.Pentagon) :
  let final := rotate150 initial
  final.top = Shape.Pentagon ∧ 
  final.right = Shape.Square ∧ 
  final.bottom = Shape.Triangle := by
  sorry

end rotation_150_degrees_l3878_387864


namespace complement_union_of_sets_l3878_387813

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

theorem complement_union_of_sets : 
  (A ∪ B)ᶜ = {2, 4} :=
by sorry

end complement_union_of_sets_l3878_387813


namespace multiplicand_difference_l3878_387831

theorem multiplicand_difference (a b : ℕ) : 
  a * b = 100100 → 
  a < b → 
  a % 10 = 2 → 
  b % 10 = 6 → 
  b - a = 564 := by
sorry

end multiplicand_difference_l3878_387831


namespace x_value_when_z_is_64_l3878_387842

/-- Given that x is inversely proportional to y², y is directly proportional to √z,
    and x = 4 when z = 16, prove that x = 1 when z = 64 -/
theorem x_value_when_z_is_64 
  (x y z : ℝ) 
  (h1 : ∃ (k : ℝ), x * y^2 = k) 
  (h2 : ∃ (m : ℝ), y = m * Real.sqrt z) 
  (h3 : x = 4 ∧ z = 16) : 
  z = 64 → x = 1 := by
sorry

end x_value_when_z_is_64_l3878_387842


namespace complex_modulus_l3878_387866

theorem complex_modulus (z : ℂ) (h : (1 + 2*I)*z = 3 - 4*I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_modulus_l3878_387866


namespace add_three_tenths_to_57_7_l3878_387884

theorem add_three_tenths_to_57_7 : (57.7 : ℝ) + (3 / 10 : ℝ) = 58 := by
  sorry

end add_three_tenths_to_57_7_l3878_387884


namespace greatest_b_value_l3878_387872

theorem greatest_b_value (a b : ℤ) (h : a * b + 7 * a + 6 * b = -6) : 
  ∀ c : ℤ, (∃ d : ℤ, d * c + 7 * d + 6 * c = -6) → c ≤ -1 :=
by sorry

end greatest_b_value_l3878_387872


namespace parallel_vectors_m_value_l3878_387800

/-- Two vectors in R² are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ,
  let a : ℝ × ℝ := (1, m)
  let b : ℝ × ℝ := (m, 2)
  parallel a b → m = -Real.sqrt 2 ∨ m = Real.sqrt 2 := by
sorry


end parallel_vectors_m_value_l3878_387800


namespace larger_solid_volume_is_seven_halves_l3878_387856

-- Define the rectangular prism
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a plane in 3D space
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define the function to calculate the volume of the larger solid
def largerSolidVolume (prism : RectangularPrism) (plane : Plane3D) : ℝ := sorry

-- Theorem statement
theorem larger_solid_volume_is_seven_halves :
  let prism := RectangularPrism.mk 2 3 1
  let A := Point3D.mk 0 0 0
  let B := Point3D.mk 3 0 0
  let E := Point3D.mk 0 3 0
  let F := Point3D.mk 0 3 1
  let G := Point3D.mk 3 3 1
  let P := Point3D.mk 1.5 (3/2) (1/2)
  let Q := Point3D.mk 0 (3/2) (1/2)
  let plane := Plane3D.mk 1 1 1 0  -- Placeholder plane equation
  largerSolidVolume prism plane = 7/2 := by
  sorry


end larger_solid_volume_is_seven_halves_l3878_387856


namespace probability_ratio_l3878_387811

def num_balls : ℕ := 25
def num_bins : ℕ := 6

def probability_config_1 : ℚ :=
  (Nat.choose num_bins 2 * (Nat.factorial num_balls / (Nat.factorial 3 * Nat.factorial 3 * (Nat.factorial 5)^4))) /
  (num_bins^num_balls : ℚ)

def probability_config_2 : ℚ :=
  (Nat.choose num_bins 2 * Nat.choose (num_bins - 2) 2 * (Nat.factorial num_balls / (Nat.factorial 3 * Nat.factorial 3 * (Nat.factorial 4)^2 * (Nat.factorial 5)^2))) /
  (num_bins^num_balls : ℚ)

theorem probability_ratio :
  probability_config_1 / probability_config_2 = 625 / 6 := by
  sorry

end probability_ratio_l3878_387811


namespace sqrt_sum_reciprocal_implies_fraction_l3878_387848

theorem sqrt_sum_reciprocal_implies_fraction (x : ℝ) (h : Real.sqrt x + 1 / Real.sqrt x = 3) :
  x / (x^2 + 2018*x + 1) = 1 / 2025 := by
  sorry

end sqrt_sum_reciprocal_implies_fraction_l3878_387848


namespace unique_three_digit_factorial_sum_l3878_387849

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def digit_factorial_sum (n : ℕ) : ℕ :=
  (n.digits 10).map factorial |>.sum

def does_not_contain_five (n : ℕ) : Prop :=
  5 ∉ n.digits 10

theorem unique_three_digit_factorial_sum :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ does_not_contain_five n ∧ n = digit_factorial_sum n :=
  by sorry

end unique_three_digit_factorial_sum_l3878_387849


namespace repeating_decimal_37_l3878_387836

/-- The repeating decimal 0.373737... expressed as a rational number -/
theorem repeating_decimal_37 : ∃ (x : ℚ), x = 37 / 99 ∧ 
  ∀ (n : ℕ), (100 * x - ⌊100 * x⌋ : ℚ) * 10^n = (37 * 10^n : ℚ) % 100 / 100 := by
  sorry

end repeating_decimal_37_l3878_387836


namespace triangle_angle_proof_l3878_387809

theorem triangle_angle_proof (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π → -- Triangle condition
  a * Real.cos B = 3 * b * Real.cos A → -- Given equation
  B = A - π / 6 → -- Given relation between A and B
  B = π / 6 := by sorry

end triangle_angle_proof_l3878_387809


namespace probability_yellow_ball_l3878_387873

/-- Probability of choosing a yellow ball from a bag -/
theorem probability_yellow_ball (red yellow blue : ℕ) (h : red = 2 ∧ yellow = 5 ∧ blue = 4) :
  (yellow : ℚ) / (red + yellow + blue : ℚ) = 5 / 11 := by
  sorry

end probability_yellow_ball_l3878_387873


namespace dean_has_30_insects_l3878_387815

-- Define the number of insects for each person
def angela_insects : ℕ := 75
def jacob_insects : ℕ := 2 * angela_insects
def dean_insects : ℕ := jacob_insects / 5

-- Theorem to prove
theorem dean_has_30_insects : dean_insects = 30 := by
  sorry

end dean_has_30_insects_l3878_387815


namespace change_after_purchase_l3878_387878

/-- Calculates the change after a purchase given initial amount, number of items, and cost per item. -/
def calculate_change (initial_amount : ℕ) (num_items : ℕ) (cost_per_item : ℕ) : ℕ :=
  initial_amount - (num_items * cost_per_item)

/-- Theorem stating that given $20 initially, buying 3 items at $2 each results in $14 change. -/
theorem change_after_purchase :
  calculate_change 20 3 2 = 14 := by
  sorry

end change_after_purchase_l3878_387878


namespace exchange_process_duration_l3878_387812

/-- Represents the maximum number of exchanges possible in the described process -/
def max_exchanges (n : ℕ) : ℕ := n - 1

/-- The number of children in the line -/
def total_children : ℕ := 20

/-- The theorem stating that the exchange process cannot continue for more than an hour -/
theorem exchange_process_duration : max_exchanges total_children < 60 := by
  sorry


end exchange_process_duration_l3878_387812


namespace total_bread_and_treats_l3878_387859

/-- The number of treats Jane brings -/
def jane_treats : ℕ := sorry

/-- The number of pieces of bread Jane brings -/
def jane_bread : ℕ := sorry

/-- The number of treats Wanda brings -/
def wanda_treats : ℕ := sorry

/-- The number of pieces of bread Wanda brings -/
def wanda_bread : ℕ := 90

theorem total_bread_and_treats :
  (jane_treats : ℚ) * (3 / 4) = jane_bread ∧
  (jane_treats : ℚ) / 2 = wanda_treats ∧
  3 * wanda_treats = wanda_bread ∧
  jane_treats + jane_bread + wanda_treats + wanda_bread = 225 := by sorry

end total_bread_and_treats_l3878_387859


namespace negation_of_proposition_l3878_387875

theorem negation_of_proposition (f : ℝ → ℝ) :
  (¬ ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0) ↔ 
  (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) :=
by sorry

end negation_of_proposition_l3878_387875


namespace candy_container_problem_l3878_387829

theorem candy_container_problem (V₁ V₂ n₁ : ℝ) (h₁ : V₁ = 72) (h₂ : V₂ = 216) (h₃ : n₁ = 30) :
  let n₂ := (n₁ / V₁) * V₂
  n₂ = 90 := by
sorry

end candy_container_problem_l3878_387829


namespace unique_linear_function_l3878_387879

/-- A linear function passing through two given points -/
def linear_function_through_points (k b : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  k ≠ 0 ∧ y₁ = k * x₁ + b ∧ y₂ = k * x₂ + b

theorem unique_linear_function :
  ∃! k b : ℝ, linear_function_through_points k b 1 3 0 (-2) ∧ 
  ∀ x : ℝ, k * x + b = 5 * x - 2 := by
  sorry

end unique_linear_function_l3878_387879


namespace mistaken_quotient_l3878_387869

theorem mistaken_quotient (D : ℕ) (h1 : D % 21 = 0) (h2 : D / 21 = 32) : D / 12 = 56 := by
  sorry

end mistaken_quotient_l3878_387869


namespace sum_first_six_primes_l3878_387847

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- The sum of the first n prime numbers -/
def sumFirstNPrimes (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of the first 6 prime numbers is 41 -/
theorem sum_first_six_primes : sumFirstNPrimes 6 = 41 := by sorry

end sum_first_six_primes_l3878_387847


namespace pizza_slices_left_l3878_387834

def large_pizza_slices : ℕ := 12
def small_pizza_slices : ℕ := 8
def num_large_pizzas : ℕ := 2
def num_small_pizzas : ℕ := 1

def dean_eaten : ℕ := large_pizza_slices / 2
def frank_eaten : ℕ := 3
def sammy_eaten : ℕ := large_pizza_slices / 3
def nancy_cheese_eaten : ℕ := 2
def nancy_pepperoni_eaten : ℕ := 1
def olivia_eaten : ℕ := 2

def total_slices : ℕ := num_large_pizzas * large_pizza_slices + num_small_pizzas * small_pizza_slices

def total_eaten : ℕ := dean_eaten + frank_eaten + sammy_eaten + nancy_cheese_eaten + nancy_pepperoni_eaten + olivia_eaten

theorem pizza_slices_left : total_slices - total_eaten = 14 := by
  sorry

end pizza_slices_left_l3878_387834


namespace advanced_ticket_price_is_14_50_l3878_387818

/-- The price of an advanced ticket for the Rhapsody Theater -/
def advanced_ticket_price (total_tickets : ℕ) (door_price : ℚ) (total_revenue : ℚ) (door_tickets : ℕ) : ℚ :=
  (total_revenue - door_price * door_tickets) / (total_tickets - door_tickets)

/-- Theorem stating that the advanced ticket price is $14.50 given the specific conditions -/
theorem advanced_ticket_price_is_14_50 :
  advanced_ticket_price 800 22 16640 672 = 14.5 := by
  sorry

#eval advanced_ticket_price 800 22 16640 672

end advanced_ticket_price_is_14_50_l3878_387818


namespace unique_positive_solution_l3878_387806

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ (2 * x^2 - 7)^2 = 49 := by
  sorry

end unique_positive_solution_l3878_387806


namespace eva_math_score_difference_l3878_387810

/-- Represents Eva's scores in a semester -/
structure SemesterScores where
  maths : ℕ
  arts : ℕ
  science : ℕ

/-- Calculates the total score for a semester -/
def totalScore (scores : SemesterScores) : ℕ :=
  scores.maths + scores.arts + scores.science

/-- Represents Eva's scores for the year -/
structure YearScores where
  first : SemesterScores
  second : SemesterScores

/-- The problem statement -/
theorem eva_math_score_difference 
  (year : YearScores)
  (h1 : year.second.maths = 80)
  (h2 : year.second.arts = 90)
  (h3 : year.second.science = 90)
  (h4 : year.first.arts = year.second.arts - 15)
  (h5 : year.first.science = year.second.science - year.second.science / 3)
  (h6 : totalScore year.first + totalScore year.second = 485)
  : year.first.maths = year.second.maths + 10 := by
  sorry

end eva_math_score_difference_l3878_387810


namespace magical_gate_diameter_l3878_387891

theorem magical_gate_diameter :
  let circle_equation := fun (x y : ℝ) => x^2 + y^2 + 2*x - 4*y + 3 = 0
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ x y, circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    2 * radius = 2 * Real.sqrt 2 :=
by sorry

end magical_gate_diameter_l3878_387891


namespace franks_money_l3878_387814

/-- Frank's initial amount of money -/
def initial_money : ℝ := 11

/-- Amount Frank spent on a game -/
def game_cost : ℝ := 3

/-- Frank's allowance -/
def allowance : ℝ := 14

/-- Frank's final amount of money -/
def final_money : ℝ := 22

theorem franks_money :
  initial_money - game_cost + allowance = final_money :=
by sorry

end franks_money_l3878_387814


namespace julia_played_with_12_on_monday_l3878_387894

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := 19 - 7

/-- The total number of kids Julia played with -/
def total_kids : ℕ := 19

/-- The number of kids Julia played with on Tuesday -/
def tuesday_kids : ℕ := 7

theorem julia_played_with_12_on_monday :
  monday_kids = 12 :=
sorry

end julia_played_with_12_on_monday_l3878_387894


namespace crystal_barrettes_count_l3878_387895

/-- The cost of one set of barrettes in dollars -/
def barrette_cost : ℕ := 3

/-- The cost of one comb in dollars -/
def comb_cost : ℕ := 1

/-- The total amount spent by both girls in dollars -/
def total_spent : ℕ := 14

/-- The number of sets of barrettes Kristine bought -/
def kristine_barrettes : ℕ := 1

/-- The number of combs Kristine bought -/
def kristine_combs : ℕ := 1

/-- The number of combs Crystal bought -/
def crystal_combs : ℕ := 1

/-- 
Given the costs of barrettes and combs, and the purchasing information for Kristine and Crystal,
prove that Crystal bought 3 sets of barrettes.
-/
theorem crystal_barrettes_count : 
  ∃ (x : ℕ), 
    barrette_cost * (kristine_barrettes + x) + 
    comb_cost * (kristine_combs + crystal_combs) = 
    total_spent ∧ x = 3 := by
  sorry


end crystal_barrettes_count_l3878_387895


namespace baseball_card_money_ratio_l3878_387838

/-- Proves the ratio of Lisa's money to Charlotte's money given the conditions of the baseball card purchase problem -/
theorem baseball_card_money_ratio :
  let card_cost : ℕ := 100
  let patricia_money : ℕ := 6
  let lisa_money : ℕ := 5 * patricia_money
  let additional_money_needed : ℕ := 49
  let total_money : ℕ := card_cost - additional_money_needed
  let charlotte_money : ℕ := total_money - lisa_money - patricia_money
  (lisa_money : ℚ) / (charlotte_money : ℚ) = 2 / 1 :=
by sorry

end baseball_card_money_ratio_l3878_387838


namespace sum_of_facing_angles_l3878_387855

-- Define an isosceles triangle
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  vertex_angle : ℝ
  is_isosceles : vertex_angle > 0 ∧ vertex_angle < 180

-- Define the configuration of two isosceles triangles
structure TwoTrianglesConfig where
  triangle1 : IsoscelesTriangle
  triangle2 : IsoscelesTriangle
  distance : ℝ
  same_base_line : Bool
  facing_equal_sides : Bool

-- Theorem statement
theorem sum_of_facing_angles (config : TwoTrianglesConfig) :
  config.triangle1 = config.triangle2 →
  config.triangle1.vertex_angle = 40 →
  config.distance = 4 →
  config.same_base_line = true →
  config.facing_equal_sides = true →
  (180 - config.triangle1.vertex_angle) + (180 - config.triangle2.vertex_angle) = 80 := by
  sorry


end sum_of_facing_angles_l3878_387855


namespace bianca_carrots_l3878_387828

/-- The number of carrots Bianca picked the next day -/
def carrots_picked_next_day (initial_carrots thrown_out_carrots final_total : ℕ) : ℕ :=
  final_total - (initial_carrots - thrown_out_carrots)

/-- Theorem stating that Bianca picked 47 carrots the next day -/
theorem bianca_carrots : carrots_picked_next_day 23 10 60 = 47 := by
  sorry

end bianca_carrots_l3878_387828


namespace rectangular_field_area_l3878_387880

/-- Proves that a rectangular field with sides in ratio 3:4 and fencing cost of 91 rupees at 0.25 rupees per meter has an area of 8112 square meters -/
theorem rectangular_field_area (x : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  x > 0 →
  cost_per_meter = 0.25 →
  total_cost = 91 →
  (14 * x * cost_per_meter = total_cost) →
  (3 * x) * (4 * x) = 8112 :=
by sorry

end rectangular_field_area_l3878_387880


namespace equation_solution_l3878_387852

theorem equation_solution :
  let f (x : ℝ) := (x^3 - x^2 - 4*x) / (x^2 + 5*x + 6) + x
  ∀ x : ℝ, f x = -4 ↔ x = (3 + Real.sqrt 105) / 4 ∨ x = (3 - Real.sqrt 105) / 4 := by
  sorry

end equation_solution_l3878_387852


namespace tangent_line_equation_l3878_387840

/-- The equation of the tangent line to the curve y = x sin x at the point (π, 0) is y = -πx + π² -/
theorem tangent_line_equation (x y : ℝ) : 
  (y = x * Real.sin x) → -- Curve equation
  (∃ (m b : ℝ), (y = m * x + b) ∧ -- Tangent line equation
                (0 = m * π + b) ∧ -- Point (π, 0) satisfies the tangent line equation
                (m = Real.sin π + π * Real.cos π)) → -- Slope of the tangent line
  (y = -π * x + π^2) -- Resulting tangent line equation
:= by sorry

end tangent_line_equation_l3878_387840


namespace student_average_greater_than_true_average_l3878_387820

theorem student_average_greater_than_true_average 
  (x y w : ℤ) (h : x < w ∧ w < y) : 
  (x + w + 2 * y) / 4 > (x + y + w) / 3 := by
  sorry

end student_average_greater_than_true_average_l3878_387820


namespace vaccine_comparison_l3878_387837

/-- Represents a vaccine trial result -/
structure VaccineTrial where
  vaccinated : Nat
  infected : Nat

/-- Determines if a vaccine is considered effective based on trial results and population infection rate -/
def is_effective (trial : VaccineTrial) (population_rate : Real) : Prop :=
  (trial.infected : Real) / trial.vaccinated < population_rate

/-- Compares the effectiveness of two vaccines -/
def more_effective (trial1 trial2 : VaccineTrial) (population_rate : Real) : Prop :=
  is_effective trial1 population_rate ∧ is_effective trial2 population_rate ∧
  (trial1.infected : Real) / trial1.vaccinated < (trial2.infected : Real) / trial2.vaccinated

theorem vaccine_comparison :
  let population_rate : Real := 0.2
  let vaccine_I : VaccineTrial := ⟨8, 0⟩
  let vaccine_II : VaccineTrial := ⟨25, 1⟩
  more_effective vaccine_II vaccine_I population_rate :=
by
  sorry

end vaccine_comparison_l3878_387837


namespace odd_power_sum_divisible_l3878_387821

theorem odd_power_sum_divisible (x y : ℤ) :
  ∀ n : ℕ, Odd n → (x + y) ∣ (x^n + y^n) := by
  sorry

end odd_power_sum_divisible_l3878_387821


namespace prob_even_sum_is_14_27_l3878_387804

/-- Represents an unfair die where even numbers are twice as likely as odd numbers -/
structure UnfairDie where
  /-- Probability of rolling an odd number -/
  odd_prob : ℝ
  /-- Probability of rolling an even number -/
  even_prob : ℝ
  /-- Ensures probabilities sum to 1 -/
  prob_sum : odd_prob + even_prob = 1
  /-- Ensures even numbers are twice as likely as odd numbers -/
  even_twice_odd : even_prob = 2 * odd_prob

/-- Represents the result of rolling the die three times -/
def ThreeRolls := Fin 3 → Bool

/-- The probability of getting an even sum when rolling the unfair die three times -/
def prob_even_sum (d : UnfairDie) : ℝ :=
  (d.even_prob^3) + 3 * (d.even_prob * d.odd_prob^2)

theorem prob_even_sum_is_14_27 (d : UnfairDie) : prob_even_sum d = 14/27 := by
  sorry

end prob_even_sum_is_14_27_l3878_387804


namespace servant_service_duration_l3878_387801

/-- Represents the servant's employment contract and actual service --/
structure ServantContract where
  yearlyPayment : ℕ  -- Payment in Rupees for a full year of service
  uniformPrice : ℕ   -- Price of the uniform in Rupees
  actualPayment : ℕ  -- Actual payment received in Rupees
  actualUniform : Bool -- Whether the servant received the uniform

/-- Calculates the number of months served based on the contract and actual payment --/
def monthsServed (contract : ServantContract) : ℚ :=
  let totalYearlyValue := contract.yearlyPayment + contract.uniformPrice
  let actualTotalReceived := contract.actualPayment + 
    (if contract.actualUniform then contract.uniformPrice else 0)
  let fractionWorked := (totalYearlyValue - actualTotalReceived) / contract.yearlyPayment
  12 * (1 - fractionWorked)

/-- Theorem stating that given the problem conditions, the servant served for approximately 3 months --/
theorem servant_service_duration (contract : ServantContract) 
  (h1 : contract.yearlyPayment = 900)
  (h2 : contract.uniformPrice = 100)
  (h3 : contract.actualPayment = 650)
  (h4 : contract.actualUniform = true) :
  ∃ (m : ℕ), m = 3 ∧ abs (monthsServed contract - m) < 1 := by
  sorry

end servant_service_duration_l3878_387801


namespace hawks_score_l3878_387805

theorem hawks_score (total_score : ℕ) (eagles_margin : ℕ) (eagles_three_pointers : ℕ) 
  (h1 : total_score = 82)
  (h2 : eagles_margin = 18)
  (h3 : eagles_three_pointers = 6) : 
  total_score / 2 - eagles_margin / 2 = 32 := by
  sorry

#check hawks_score

end hawks_score_l3878_387805


namespace root_product_rational_l3878_387833

-- Define the polynomial f(z)
def f (a b c d e : ℤ) (z : ℂ) : ℂ := a * z^4 + b * z^3 + c * z^2 + d * z + e

-- Define the roots r1, r2, r3, r4
variable (r1 r2 r3 r4 : ℂ)

-- State the theorem
theorem root_product_rational
  (a b c d e : ℤ)
  (h_a_nonzero : a ≠ 0)
  (h_f_factored : ∀ z, f a b c d e z = a * (z - r1) * (z - r2) * (z - r3) * (z - r4))
  (h_sum_rational : ∃ q : ℚ, (r1 + r2 : ℂ) = q)
  (h_sum_distinct : r1 + r2 ≠ r3 + r4) :
  ∃ q : ℚ, (r1 * r2 : ℂ) = q :=
sorry

end root_product_rational_l3878_387833


namespace complement_of_union_equals_zero_five_l3878_387882

def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2, 4}

theorem complement_of_union_equals_zero_five :
  (U \ (A ∪ B)) = {0, 5} := by sorry

end complement_of_union_equals_zero_five_l3878_387882


namespace car_value_decrease_l3878_387853

theorem car_value_decrease (original_value current_value : ℝ) 
  (h1 : original_value = 4000)
  (h2 : current_value = 2800) :
  (original_value - current_value) / original_value * 100 = 30 := by
sorry

end car_value_decrease_l3878_387853


namespace right_triangle_area_l3878_387886

theorem right_triangle_area (a b : ℝ) (h1 : a^2 - 7*a + 12 = 0) (h2 : b^2 - 7*b + 12 = 0) (h3 : a ≠ b) :
  ∃ (area : ℝ), (area = 6 ∨ area = (3 * Real.sqrt 7) / 2) ∧
  ((area = a * b / 2) ∨ (area = a * Real.sqrt (b^2 - a^2) / 2) ∨ (area = b * Real.sqrt (a^2 - b^2) / 2)) :=
by sorry

end right_triangle_area_l3878_387886


namespace interest_period_is_two_years_l3878_387881

/-- Given simple interest rate of 20% per annum and simple interest of $400,
    and compound interest of $440 for the same period and rate,
    prove that the time period is 2 years. -/
theorem interest_period_is_two_years 
  (simple_interest : ℝ) 
  (compound_interest : ℝ) 
  (rate : ℝ) :
  simple_interest = 400 →
  compound_interest = 440 →
  rate = 0.20 →
  ∃ t : ℝ, t = 2 ∧ (1 + rate)^t = (rate * simple_interest * t + simple_interest) / simple_interest :=
by sorry

end interest_period_is_two_years_l3878_387881


namespace arrangements_count_l3878_387854

def number_of_people : ℕ := 7
def number_of_gaps : ℕ := number_of_people - 1

theorem arrangements_count :
  (number_of_people - 2).factorial * number_of_gaps.choose 2 = 3600 :=
by sorry

end arrangements_count_l3878_387854


namespace chocolateProblemSolution_l3878_387823

def chocolateProblem (totalBoxes : Float) (piecesPerBox : Float) (remainingPieces : Nat) : Float :=
  let totalPieces := totalBoxes * piecesPerBox
  let givenPieces := totalPieces - remainingPieces.toFloat
  givenPieces / piecesPerBox

theorem chocolateProblemSolution :
  chocolateProblem 14.0 6.0 42 = 7.0 := by
  sorry

end chocolateProblemSolution_l3878_387823


namespace emily_minimum_grade_to_beat_ahmed_l3878_387862

/-- Represents a student's grade -/
structure StudentGrade where
  current_grade : ℕ
  final_grade : ℕ

/-- Calculates the final average grade given current grade and final assignment grade -/
def finalAverageGrade (s : StudentGrade) : ℚ :=
  (9 * s.current_grade + s.final_grade) / 10

theorem emily_minimum_grade_to_beat_ahmed :
  ∀ (ahmed emily : StudentGrade),
    ahmed.current_grade = 91 →
    emily.current_grade = 92 →
    ahmed.final_grade = 100 →
    (∀ g : ℕ, g < 92 → finalAverageGrade emily < finalAverageGrade ahmed) ∧
    finalAverageGrade { current_grade := 92, final_grade := 92 } > finalAverageGrade ahmed :=
by sorry

end emily_minimum_grade_to_beat_ahmed_l3878_387862


namespace prob_divisible_by_3_and_8_prob_divisible_by_3_and_8_value_l3878_387816

/-- The probability of a randomly selected three-digit number being divisible by both 3 and 8 -/
theorem prob_divisible_by_3_and_8 : ℚ :=
  let three_digit_numbers := Finset.Icc 100 999
  let divisible_by_24 := three_digit_numbers.filter (λ n => n % 24 = 0)
  (divisible_by_24.card : ℚ) / three_digit_numbers.card

/-- The probability of a randomly selected three-digit number being divisible by both 3 and 8 is 37/900 -/
theorem prob_divisible_by_3_and_8_value :
  prob_divisible_by_3_and_8 = 37 / 900 := by
  sorry


end prob_divisible_by_3_and_8_prob_divisible_by_3_and_8_value_l3878_387816


namespace sum_a_b_equals_three_l3878_387858

theorem sum_a_b_equals_three (a b : ℝ) (h : |a - 4| + (b + 1)^2 = 0) : a + b = 3 := by
  sorry

end sum_a_b_equals_three_l3878_387858


namespace value_calculation_l3878_387892

theorem value_calculation (initial_number : ℕ) (h : initial_number = 26) : 
  ((((initial_number + 20) * 2) / 2) - 2) * 2 = 88 := by
  sorry

end value_calculation_l3878_387892


namespace surface_area_unchanged_l3878_387863

/-- Represents the dimensions of a rectangular solid -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a cube -/
structure Cube where
  side : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surfaceArea (r : RectangularSolid) : ℝ :=
  2 * (r.length * r.width + r.length * r.height + r.width * r.height)

/-- Calculates the exposed surface area of a cube when it touches two faces of the solid -/
def exposedCubeArea (c : Cube) : ℝ :=
  2 * c.side * c.side

/-- Theorem: The surface area remains unchanged after cube removal -/
theorem surface_area_unchanged 
  (original : RectangularSolid)
  (removed : Cube)
  (h1 : original.length = 5)
  (h2 : original.width = 3)
  (h3 : original.height = 4)
  (h4 : removed.side = 2)
  (h5 : exposedCubeArea removed = exposedCubeArea removed) :
  surfaceArea original = surfaceArea original :=
by sorry

end surface_area_unchanged_l3878_387863


namespace julia_play_difference_l3878_387870

/-- The number of kids Julia played tag with on Monday -/
def monday_tag : ℕ := 28

/-- The number of kids Julia played hide & seek with on Monday -/
def monday_hide_seek : ℕ := 15

/-- The number of kids Julia played tag with on Tuesday -/
def tuesday_tag : ℕ := 33

/-- The number of kids Julia played hide & seek with on Tuesday -/
def tuesday_hide_seek : ℕ := 21

/-- The difference in the total number of kids Julia played with on Tuesday compared to Monday -/
theorem julia_play_difference : 
  (tuesday_tag + tuesday_hide_seek) - (monday_tag + monday_hide_seek) = 11 := by
  sorry

end julia_play_difference_l3878_387870


namespace negative_one_and_half_equality_l3878_387861

theorem negative_one_and_half_equality : -1 - (1/2 : ℚ) = -(3/2 : ℚ) := by
  sorry

end negative_one_and_half_equality_l3878_387861


namespace power_product_cube_l3878_387846

theorem power_product_cube (a b : ℝ) : (a * b) ^ 3 = a ^ 3 * b ^ 3 := by
  sorry

end power_product_cube_l3878_387846


namespace min_dot_product_on_ellipse_l3878_387893

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the center and right focus
def O : ℝ × ℝ := (0, 0)
def F : ℝ × ℝ := (1, 0)

-- Define the dot product of vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.1) + (v1.2 * v2.2)

-- State the theorem
theorem min_dot_product_on_ellipse :
  ∃ (min : ℝ), min = 1/2 ∧
  ∀ (P : ℝ × ℝ), is_on_ellipse P.1 P.2 →
    dot_product (P.1 - O.1, P.2 - O.2) (P.1 - F.1, P.2 - F.2) ≥ min :=
sorry

end min_dot_product_on_ellipse_l3878_387893


namespace largest_three_digit_number_with_1_hundreds_l3878_387826

def digits : List Nat := [1, 5, 6, 9]

def isValidNumber (n : Nat) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  (n / 100 = 1) ∧
  (∀ d, d ∈ digits → (n / 10 % 10 = d ∨ n % 10 = d))

theorem largest_three_digit_number_with_1_hundreds :
  ∀ n : Nat, isValidNumber n → n ≤ 196 :=
sorry

end largest_three_digit_number_with_1_hundreds_l3878_387826


namespace missing_interior_angle_l3878_387825

theorem missing_interior_angle (n : ℕ) (sum_without_one : ℝ) (missing_angle : ℝ) :
  n = 18 →
  sum_without_one = 2750 →
  (n - 2) * 180 = sum_without_one + missing_angle →
  missing_angle = 130 :=
by sorry

end missing_interior_angle_l3878_387825


namespace pyramid_cone_properties_l3878_387868

/-- Represents a square pyramid with a cone resting on its base --/
structure PyramidWithCone where
  pyramid_height : ℝ
  cone_base_radius : ℝ
  -- The cone is tangent to the other four faces of the pyramid
  is_tangent : Bool

/-- Calculates the edge length of the pyramid's base --/
def calculate_edge_length (p : PyramidWithCone) : ℝ := sorry

/-- Calculates the surface area of the cone not in contact with the pyramid --/
def calculate_cone_surface_area (p : PyramidWithCone) : ℝ := sorry

/-- Theorem stating the properties of the specific pyramid and cone configuration --/
theorem pyramid_cone_properties :
  let p : PyramidWithCone := {
    pyramid_height := 9,
    cone_base_radius := 3,
    is_tangent := true
  }
  calculate_edge_length p = 9 ∧
  calculate_cone_surface_area p = 30 * Real.pi := by sorry

end pyramid_cone_properties_l3878_387868


namespace perpendicular_line_through_point_l3878_387887

-- Define the given line
def given_line (x y : ℝ) : Prop := x - 2*y + 1 = 0

-- Define the point that the desired line passes through
def point : ℝ × ℝ := (-1, 3)

-- Define the desired line
def desired_line (x y : ℝ) : Prop := y + 2*x - 1 = 0

-- Theorem statement
theorem perpendicular_line_through_point :
  (∀ x y, given_line x y → desired_line x y → (x - point.1) * (x - point.1) + (y - point.2) * (y - point.2) = 0) ∧
  (∀ x₁ y₁ x₂ y₂, given_line x₁ y₁ → given_line x₂ y₂ → desired_line x₁ y₁ → desired_line x₂ y₂ →
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    ((x₂ - x₁) * (y₂ - y₁)) / ((x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁)) = -1/2) :=
sorry

end perpendicular_line_through_point_l3878_387887


namespace subset_sum_equals_A_l3878_387808

theorem subset_sum_equals_A (A : ℕ) (a : List ℕ) : 
  (∀ n ∈ Finset.range 9, A % (n + 1) = 0) →
  (∀ x ∈ a, x < 10) →
  (2 * A = a.sum) →
  ∃ s : List ℕ, s.toFinset ⊆ a.toFinset ∧ s.sum = A := by
  sorry

end subset_sum_equals_A_l3878_387808


namespace intersection_condition_l3878_387899

-- Define the quadratic function
def f (a x : ℝ) := a * x^2 - 4 * a * x - 2

-- Define the solution set of the inequality
def solution_set (a : ℝ) := {x : ℝ | f a x > 0}

-- Define the given set
def given_set := {x : ℝ | 3 < x ∧ x < 4}

-- Theorem statement
theorem intersection_condition (a : ℝ) : 
  (∃ x, x ∈ solution_set a ∧ x ∈ given_set) ↔ a < -2/3 :=
sorry

end intersection_condition_l3878_387899


namespace candy_bar_count_l3878_387865

theorem candy_bar_count (num_bags : ℕ) (bars_per_bag : ℕ) (h1 : num_bags = 5) (h2 : bars_per_bag = 3) :
  num_bags * bars_per_bag = 15 := by
sorry

end candy_bar_count_l3878_387865


namespace monomial_properties_l3878_387817

/-- Represents a monomial in two variables -/
structure Monomial (α : Type*) [Ring α] where
  coefficient : α
  exponent_a : ℕ
  exponent_b : ℕ

/-- Calculate the degree of a monomial -/
def Monomial.degree {α : Type*} [Ring α] (m : Monomial α) : ℕ :=
  m.exponent_a + m.exponent_b

/-- The monomial -2a²b -/
def example_monomial : Monomial ℤ :=
  { coefficient := -2
    exponent_a := 2
    exponent_b := 1 }

theorem monomial_properties :
  (example_monomial.coefficient = -2) ∧
  (example_monomial.degree = 3) := by
  sorry

end monomial_properties_l3878_387817


namespace ellipse_standard_equation_l3878_387830

/-- Definition of an ellipse with given major axis length and eccentricity -/
structure Ellipse where
  major_axis : ℝ
  eccentricity : ℝ

/-- The standard equation of an ellipse -/
def standard_equation (e : Ellipse) : Prop :=
  (∀ x y : ℝ, x^2 / 16 + y^2 / 7 = 1) ∨ (∀ x y : ℝ, x^2 / 7 + y^2 / 16 = 1)

/-- Theorem stating that an ellipse with major axis 8 and eccentricity 3/4 satisfies the standard equation -/
theorem ellipse_standard_equation (e : Ellipse) (h1 : e.major_axis = 8) (h2 : e.eccentricity = 3/4) :
  standard_equation e := by
  sorry

end ellipse_standard_equation_l3878_387830


namespace reflected_ray_slope_l3878_387885

/-- A light ray is emitted from a point, reflects off the y-axis, and is tangent to a circle. -/
theorem reflected_ray_slope (emissionPoint : ℝ × ℝ) (circleCenter : ℝ × ℝ) (circleRadius : ℝ) :
  emissionPoint = (-2, -3) →
  circleCenter = (-3, 2) →
  circleRadius = 1 →
  ∃ (k : ℝ), (k = -4/3 ∨ k = -3/4) ∧
    (∀ (x y : ℝ), (x + 3)^2 + (y - 2)^2 = 1 →
      (k * x - y - 2 * k - 3 = 0 →
        ((3 * k + 2 + 2 * k + 3)^2 / (k^2 + 1) = 1))) :=
by sorry

end reflected_ray_slope_l3878_387885


namespace certain_number_equation_l3878_387888

theorem certain_number_equation : ∃ x : ℤ, 9548 + 7314 = x + 13500 ∧ x = 3362 := by
  sorry

end certain_number_equation_l3878_387888


namespace circle_bisector_properties_l3878_387843

-- Define the circle
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define point A
def A : ℝ × ℝ := (6, 0)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define a point P on the circle
def P (x y : ℝ) : Prop := Circle x y

-- Define point M on the bisector of ∠POA and on PA
def M (x y : ℝ) (px py : ℝ) : Prop :=
  P px py ∧ 
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ 
  x = t * px + (1 - t) * A.1 ∧
  y = t * py + (1 - t) * A.2 ∧
  (x - O.1) * (A.1 - O.1) + (y - O.2) * (A.2 - O.2) = 
  (px - O.1) * (A.1 - O.1) + (py - O.2) * (A.2 - O.2)

-- Theorem statement
theorem circle_bisector_properties 
  (x y px py : ℝ) 
  (h_m : M x y px py) :
  (∃ (ma pm : ℝ), ma / pm = 3 ∧ 
    ma^2 = (x - A.1)^2 + (y - A.2)^2 ∧
    pm^2 = (x - px)^2 + (y - py)^2) ∧
  (x - 2/3)^2 + y^2 = 9/4 :=
sorry

end circle_bisector_properties_l3878_387843


namespace sum_even_minus_odd_product_equals_6401_l3878_387824

def sum_of_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

def product_of_odd_integers (a b : ℕ) : ℕ :=
  if ∃ n ∈ Finset.range (b - a + 1), Even (a + n) then 0 else 1

theorem sum_even_minus_odd_product_equals_6401 :
  sum_of_integers 100 150 + count_even_integers 100 150 - product_of_odd_integers 100 150 = 6401 := by
  sorry

end sum_even_minus_odd_product_equals_6401_l3878_387824


namespace unique_divisible_number_l3878_387845

/-- A function that constructs a five-digit number of the form 6n272 -/
def construct_number (n : Nat) : Nat :=
  60000 + n * 1000 + 272

/-- Proposition: 63272 is the only number of the form 6n272 (where n is a single digit) 
    that is divisible by both 11 and 5 -/
theorem unique_divisible_number : 
  ∃! n : Nat, n < 10 ∧ 
  (construct_number n).mod 11 = 0 ∧ 
  (construct_number n).mod 5 = 0 ∧
  construct_number n = 63272 := by
  sorry

end unique_divisible_number_l3878_387845


namespace return_probability_is_one_sixth_l3878_387871

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron :=
  (vertices : Finset (Fin 4))
  (edges : Finset (Fin 4 × Fin 4))
  (adjacent : Fin 4 → Finset (Fin 4))
  (adjacent_sym : ∀ v₁ v₂, v₂ ∈ adjacent v₁ ↔ v₁ ∈ adjacent v₂)
  (adjacent_card : ∀ v, (adjacent v).card = 3)

/-- The probability of returning to the starting vertex in two moves -/
def return_probability (t : RegularTetrahedron) : ℚ :=
  1 / 6

/-- Theorem: The probability of returning to the starting vertex in two moves is 1/6 -/
theorem return_probability_is_one_sixth (t : RegularTetrahedron) :
  return_probability t = 1 / 6 := by
  sorry

end return_probability_is_one_sixth_l3878_387871


namespace team_arrangement_solution_l3878_387822

/-- Represents the arrangement of team members in rows. -/
structure TeamArrangement where
  totalMembers : ℕ
  numRows : ℕ
  firstRowMembers : ℕ
  h1 : totalMembers = (numRows * (2 * firstRowMembers + numRows - 1)) / 2
  h2 : numRows > 16

/-- The solution to the team arrangement problem. -/
theorem team_arrangement_solution :
  ∃ (arr : TeamArrangement),
    arr.totalMembers = 1000 ∧
    arr.numRows = 25 ∧
    arr.firstRowMembers = 28 :=
  sorry


end team_arrangement_solution_l3878_387822


namespace fox_can_catch_mole_l3878_387807

/-- Represents a mound in the line of 100 mounds. -/
def Mound := Fin 100

/-- Represents the state of the game at any given time. -/
structure GameState where
  molePosition : Mound
  foxPosition : Mound

/-- Represents a strategy for the fox. -/
def FoxStrategy := GameState → Mound

/-- Represents the result of a single move in the game. -/
inductive MoveResult
  | Caught
  | Continue (newState : GameState)

/-- Simulates a single move in the game. -/
def makeMove (state : GameState) (strategy : FoxStrategy) : MoveResult :=
  sorry

/-- Simulates the game for a given number of moves. -/
def playGame (initialState : GameState) (strategy : FoxStrategy) (moves : Nat) : Bool :=
  sorry

/-- The main theorem stating that there exists a strategy for the fox to catch the mole. -/
theorem fox_can_catch_mole :
  ∃ (strategy : FoxStrategy), ∀ (initialState : GameState),
    playGame initialState strategy 200 = true :=
  sorry

end fox_can_catch_mole_l3878_387807


namespace circle_condition_l3878_387860

/-- A quadratic equation in two variables represents a circle if and only if
    D^2 + E^2 - 4F > 0, where the equation is in the form x^2 + y^2 + Dx + Ey + F = 0 -/
def is_circle (D E F : ℝ) : Prop := D^2 + E^2 - 4*F > 0

/-- The equation x^2 + y^2 + x + y + k = 0 represents a circle -/
def represents_circle (k : ℝ) : Prop := is_circle 1 1 k

/-- If x^2 + y^2 + x + y + k = 0 represents a circle, then k < 1/2 -/
theorem circle_condition (k : ℝ) : represents_circle k → k < 1/2 := by
  sorry

end circle_condition_l3878_387860


namespace bucket_capacity_l3878_387851

theorem bucket_capacity (tank_volume : ℝ) (bucket_count1 bucket_count2 : ℕ) 
  (bucket_capacity2 : ℝ) (h1 : bucket_count1 * bucket_capacity1 = tank_volume) 
  (h2 : bucket_count2 * bucket_capacity2 = tank_volume) 
  (h3 : bucket_count1 = 26) (h4 : bucket_count2 = 39) (h5 : bucket_capacity2 = 9) : 
  bucket_capacity1 = 13.5 :=
by
  sorry

end bucket_capacity_l3878_387851


namespace parabola_equation_l3878_387857

/-- A parabola with focus at (5,0) has the standard equation y^2 = 20x -/
theorem parabola_equation (x y : ℝ) : 
  (∃ (F : ℝ × ℝ), F = (5, 0) ∧ 
   ∀ (P : ℝ × ℝ), P.1 = x ∧ P.2 = y → 
   (P.1 - F.1)^2 + P.2^2 = (P.1 - 2.5)^2) → 
  y^2 = 20 * x := by
sorry

end parabola_equation_l3878_387857


namespace union_equality_implies_m_value_l3878_387889

def A (m : ℝ) : Set ℝ := {1, 3, Real.sqrt m}
def B (m : ℝ) : Set ℝ := {1, m}

theorem union_equality_implies_m_value (m : ℝ) :
  A m ∪ B m = A m → m = 0 ∨ m = 3 := by
  sorry

end union_equality_implies_m_value_l3878_387889


namespace total_spent_is_20_27_l3878_387819

/-- Calculates the total amount spent on items with discount and tax --/
def totalSpent (initialAmount : ℚ) (candyPrice : ℚ) (chocolatePrice : ℚ) (gumPrice : ℚ) 
  (chipsPrice : ℚ) (discountRate : ℚ) (taxRate : ℚ) : ℚ :=
  let discountedCandyPrice := candyPrice * (1 - discountRate)
  let subtotal := discountedCandyPrice + chocolatePrice + gumPrice + chipsPrice
  let tax := subtotal * taxRate
  subtotal + tax

/-- Theorem stating that the total amount spent is $20.27 --/
theorem total_spent_is_20_27 : 
  totalSpent 50 7 6 3 4 (10/100) (5/100) = 2027/100 := by
  sorry

end total_spent_is_20_27_l3878_387819


namespace exists_nonperiodic_with_repeating_subsequence_l3878_387827

/-- A sequence of natural numbers -/
def Sequence := ℕ → ℕ

/-- Property: For any index k, there exists a t such that the sequence repeats at multiples of t -/
def HasRepeatingSubsequence (a : Sequence) : Prop :=
  ∀ k : ℕ, ∃ t : ℕ, ∀ n : ℕ, a k = a (k + n * t)

/-- Property: A sequence is periodic -/
def IsPeriodic (a : Sequence) : Prop :=
  ∃ T : ℕ, ∀ k : ℕ, a k = a (k + T)

/-- Theorem: There exists a sequence that has repeating subsequences but is not periodic -/
theorem exists_nonperiodic_with_repeating_subsequence :
  ∃ a : Sequence, HasRepeatingSubsequence a ∧ ¬IsPeriodic a :=
sorry

end exists_nonperiodic_with_repeating_subsequence_l3878_387827


namespace equation_solution_l3878_387898

theorem equation_solution (x : ℝ) : 1 / x + x / 80 = 7 / 30 → x = 12 ∨ x = 20 / 3 := by
  sorry

end equation_solution_l3878_387898


namespace symmetric_derivative_minimum_value_l3878_387867

-- Define the function f
def f (b c x : ℝ) : ℝ := x^3 + b*x^2 + c*x

-- Define the derivative of f
def f' (b c x : ℝ) : ℝ := 3*x^2 + 2*b*x + c

-- State the theorem
theorem symmetric_derivative_minimum_value (b c : ℝ) :
  (∀ x : ℝ, f' b c (4 - x) = f' b c x) →  -- f' is symmetric about x = 2
  (∃ t : ℝ, ∀ x : ℝ, f b c t ≤ f b c x) →  -- f has a minimum value
  (b = -6) ∧  -- Part 1: value of b
  (∃ g : ℝ → ℝ, (∀ t > 2, g t = f b c t) ∧  -- Part 2: domain of g
                (∀ y : ℝ, (∃ t > 2, g t = y) ↔ y < 8))  -- Part 3: range of g
  := by sorry

end symmetric_derivative_minimum_value_l3878_387867


namespace parallelogram_base_l3878_387874

/-- 
Given a parallelogram with area 320 cm² and height 16 cm, 
prove that its base is 20 cm.
-/
theorem parallelogram_base (area : ℝ) (height : ℝ) (base : ℝ) :
  area = 320 ∧ height = 16 ∧ area = base * height → base = 20 := by
  sorry

end parallelogram_base_l3878_387874


namespace composite_number_l3878_387896

theorem composite_number (n : ℕ+) : ∃ (p : ℕ), Prime p ∧ p ∣ (19 * 8^n.val + 17) ∧ 1 < p ∧ p < 19 * 8^n.val + 17 := by
  sorry

end composite_number_l3878_387896


namespace total_pears_picked_l3878_387877

def alyssa_pears : ℕ := 42
def nancy_pears : ℕ := 17

theorem total_pears_picked :
  alyssa_pears + nancy_pears = 59 := by sorry

end total_pears_picked_l3878_387877


namespace smallest_upper_bound_l3878_387803

theorem smallest_upper_bound (x : ℤ) 
  (h1 : 3 < x ∧ x < 10)
  (h2 : 5 < x ∧ x < 18)
  (h3 : x > -2)
  (h4 : 0 < x ∧ x < 8)
  (h5 : x + 1 < 9) :
  ∃ (upper_bound : ℤ), 
    (∀ y : ℤ, (3 < y ∧ y < 10) → 
               (5 < y ∧ y < 18) → 
               (y > -2) → 
               (0 < y ∧ y < 8) → 
               (y + 1 < 9) → 
               y ≤ upper_bound) ∧
    (upper_bound = 8) :=
sorry

end smallest_upper_bound_l3878_387803


namespace quadratic_range_at_minus_two_l3878_387841

/-- A quadratic function passing through the origin -/
structure QuadraticThroughOrigin where
  a : ℝ
  b : ℝ
  a_nonzero : a ≠ 0

/-- The quadratic function f(x) = ax² + bx -/
def f (q : QuadraticThroughOrigin) (x : ℝ) : ℝ :=
  q.a * x^2 + q.b * x

/-- Theorem: For a quadratic function f(x) = ax² + bx (a ≠ 0) passing through the origin,
    if 1 ≤ f(-1) ≤ 2 and 2 ≤ f(1) ≤ 4, then 5 ≤ f(-2) ≤ 10 -/
theorem quadratic_range_at_minus_two (q : QuadraticThroughOrigin) 
    (h1 : 1 ≤ f q (-1)) (h2 : f q (-1) ≤ 2)
    (h3 : 2 ≤ f q 1) (h4 : f q 1 ≤ 4) :
    5 ≤ f q (-2) ∧ f q (-2) ≤ 10 := by
  sorry

end quadratic_range_at_minus_two_l3878_387841


namespace lcm_of_20_45_75_l3878_387876

theorem lcm_of_20_45_75 : Nat.lcm 20 (Nat.lcm 45 75) = 900 := by
  sorry

end lcm_of_20_45_75_l3878_387876


namespace composite_expression_l3878_387890

theorem composite_expression (a b : ℕ) : 
  ∃ (p q : ℕ), p > 1 ∧ q > 1 ∧ 4*a^2 + 4*a*b + 4*a + 2*b + 1 = p * q :=
by sorry

end composite_expression_l3878_387890


namespace campers_rowing_total_l3878_387883

/-- The total number of campers who went rowing throughout the day -/
def total_campers (morning afternoon evening : ℕ) : ℕ :=
  morning + afternoon + evening

/-- Theorem stating that the total number of campers who went rowing is 764 -/
theorem campers_rowing_total :
  total_campers 235 387 142 = 764 := by
  sorry

end campers_rowing_total_l3878_387883


namespace prob_white_both_urns_l3878_387844

/-- Represents an urn with a certain number of black and white balls -/
structure Urn :=
  (black : ℕ)
  (white : ℕ)

/-- Calculates the probability of drawing a white ball from an urn -/
def prob_white (u : Urn) : ℚ :=
  u.white / (u.black + u.white)

/-- The probability of drawing white balls from both urns is 7/30 -/
theorem prob_white_both_urns (urn1 urn2 : Urn)
  (h1 : urn1 = Urn.mk 6 4)
  (h2 : urn2 = Urn.mk 5 7) :
  prob_white urn1 * prob_white urn2 = 7 / 30 := by
  sorry

end prob_white_both_urns_l3878_387844


namespace hyperbola_conditions_exclusive_or_conditions_l3878_387802

-- Define proposition p
def p (k : ℝ) : Prop := k^2 - 8*k - 20 ≤ 0

-- Define proposition q
def q (k : ℝ) : Prop := ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 4 - k > 0 ∧ 1 - k < 0 ∧
  ∀ (x y : ℝ), x^2 / (4-k) + y^2 / (1-k) = 1 ↔ (x/a)^2 - (y/b)^2 = 1

theorem hyperbola_conditions (k : ℝ) : q k ↔ 1 < k ∧ k < 4 := by sorry

theorem exclusive_or_conditions (k : ℝ) : (p k ∨ q k) ∧ ¬(p k ∧ q k) ↔ 
  (-2 ≤ k ∧ k ≤ 1) ∨ (4 ≤ k ∧ k ≤ 10) := by sorry

end hyperbola_conditions_exclusive_or_conditions_l3878_387802
