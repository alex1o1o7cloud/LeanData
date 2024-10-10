import Mathlib

namespace book_pyramid_theorem_l1960_196001

/-- Represents a book pyramid with a given number of levels -/
structure BookPyramid where
  levels : ℕ
  top_level_books : ℕ
  ratio : ℚ
  total_books : ℕ

/-- Calculates the total number of books in the pyramid -/
def calculate_total (p : BookPyramid) : ℚ :=
  p.top_level_books * (1 - p.ratio ^ p.levels) / (1 - p.ratio)

/-- Theorem stating the properties of the specific book pyramid -/
theorem book_pyramid_theorem (p : BookPyramid) 
  (h1 : p.levels = 4)
  (h2 : p.ratio = 4/5)
  (h3 : p.total_books = 369) :
  p.top_level_books = 64 := by
  sorry


end book_pyramid_theorem_l1960_196001


namespace vegetarians_count_l1960_196087

/-- Represents the eating habits of a family -/
structure FamilyDiet where
  only_veg : ℕ
  only_non_veg : ℕ
  both : ℕ

/-- Calculates the total number of people who eat vegetarian food -/
def total_vegetarians (fd : FamilyDiet) : ℕ :=
  fd.only_veg + fd.both

/-- Theorem stating that the number of vegetarians in the given family is 20 -/
theorem vegetarians_count (fd : FamilyDiet) 
  (h1 : fd.only_veg = 11)
  (h2 : fd.only_non_veg = 6)
  (h3 : fd.both = 9) :
  total_vegetarians fd = 20 := by
  sorry

#eval total_vegetarians ⟨11, 6, 9⟩

end vegetarians_count_l1960_196087


namespace smallest_max_sum_l1960_196002

theorem smallest_max_sum (p q r s t : ℕ+) 
  (sum_constraint : p + q + r + s + t = 2015) : 
  (∃ (N : ℕ), 
    N = max (p + q) (max (q + r) (max (r + s) (s + t))) ∧ 
    N = 1005 ∧
    ∀ (M : ℕ), (M = max (p + q) (max (q + r) (max (r + s) (s + t))) → M ≥ N)) :=
by sorry

end smallest_max_sum_l1960_196002


namespace sqrt_inequality_l1960_196043

theorem sqrt_inequality (C : ℝ) (h : C > 1) :
  Real.sqrt (C + 1) - Real.sqrt C < Real.sqrt C - Real.sqrt (C - 1) := by
  sorry

end sqrt_inequality_l1960_196043


namespace xy_value_l1960_196075

theorem xy_value (x y : ℝ) 
  (h1 : (8 : ℝ)^x / (4 : ℝ)^(x + y) = 16)
  (h2 : (27 : ℝ)^(x + y) / (9 : ℝ)^(5 * y) = 729) :
  x * y = 96 := by
sorry

end xy_value_l1960_196075


namespace smallest_prime_with_digit_sum_23_l1960_196054

def digit_sum (n : ℕ) : ℕ := sorry

def is_prime (n : ℕ) : Prop := sorry

theorem smallest_prime_with_digit_sum_23 :
  (∀ p : ℕ, is_prime p ∧ digit_sum p = 23 → p ≥ 599) ∧
  is_prime 599 ∧
  digit_sum 599 = 23 := by sorry

end smallest_prime_with_digit_sum_23_l1960_196054


namespace complex_fraction_simplification_l1960_196059

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (3 - 2 * i) / (2 + 5 * i) = (-4 : ℝ) / 29 - (19 : ℝ) / 29 * i :=
by sorry

end complex_fraction_simplification_l1960_196059


namespace line_through_point_l1960_196076

/-- Given a line with equation 3 - 2kx = -4y that contains the point (5, -2),
    prove that the value of k is -0.5 -/
theorem line_through_point (k : ℝ) : 
  (3 - 2 * k * 5 = -4 * (-2)) → k = -0.5 := by
  sorry

end line_through_point_l1960_196076


namespace macey_savings_l1960_196034

/-- The amount Macey has already saved is equal to the cost of the shirt minus the amount she will save in the next 3 weeks. -/
theorem macey_savings (shirt_cost : ℝ) (weeks_left : ℕ) (weekly_savings : ℝ) 
  (h1 : shirt_cost = 3)
  (h2 : weeks_left = 3)
  (h3 : weekly_savings = 0.5) :
  shirt_cost - (weeks_left : ℝ) * weekly_savings = 1.5 := by
  sorry

end macey_savings_l1960_196034


namespace number_difference_l1960_196036

theorem number_difference (L S : ℕ) : L = 1495 → L = 5 * S + 4 → L - S = 1197 := by
  sorry

end number_difference_l1960_196036


namespace black_balls_count_l1960_196093

theorem black_balls_count (red white : ℕ) (p : ℚ) (black : ℕ) : 
  red = 3 → 
  white = 5 → 
  p = 1/4 → 
  (white : ℚ) / ((red : ℚ) + (white : ℚ) + (black : ℚ)) = p → 
  black = 12 := by
sorry

end black_balls_count_l1960_196093


namespace point_in_third_quadrant_implies_a_negative_l1960_196044

/-- A point is in the third quadrant if both its x and y coordinates are negative -/
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- If point A(a, a-1) is in the third quadrant, then a < 0 -/
theorem point_in_third_quadrant_implies_a_negative (a : ℝ) : 
  in_third_quadrant a (a - 1) → a < 0 := by
  sorry

end point_in_third_quadrant_implies_a_negative_l1960_196044


namespace snow_cone_price_is_0_875_l1960_196008

/-- Calculates the price of a snow cone given the conditions of Todd's snow-cone stand. -/
def snow_cone_price (borrowed : ℚ) (repay : ℚ) (ingredients_cost : ℚ) (num_sold : ℕ) (leftover : ℚ) : ℚ :=
  (repay + leftover) / num_sold

/-- Proves that the price of each snow cone is $0.875 under the given conditions. -/
theorem snow_cone_price_is_0_875 :
  snow_cone_price 100 110 75 200 65 = 0.875 := by
  sorry

end snow_cone_price_is_0_875_l1960_196008


namespace equation_solution_l1960_196029

theorem equation_solution (a b c : ℤ) : 
  (∀ x : ℝ, (x - a) * (x - 10) + 5 = (x + b) * (x + c)) → (a = 4 ∨ a = 16) :=
by sorry

end equation_solution_l1960_196029


namespace koolaid_percentage_is_four_percent_l1960_196086

def calculate_koolaid_percentage (initial_powder : ℚ) (initial_water : ℚ) (evaporated_water : ℚ) (water_multiplier : ℚ) : ℚ :=
  let remaining_water := initial_water - evaporated_water
  let final_water := remaining_water * water_multiplier
  let total_liquid := initial_powder + final_water
  (initial_powder / total_liquid) * 100

theorem koolaid_percentage_is_four_percent :
  calculate_koolaid_percentage 2 16 4 4 = 4 := by
  sorry

end koolaid_percentage_is_four_percent_l1960_196086


namespace exponential_simplification_l1960_196011

theorem exponential_simplification : 3 * ((-5)^2)^(3/4) = Real.sqrt 5 := by
  sorry

end exponential_simplification_l1960_196011


namespace enemy_plane_hit_probability_l1960_196047

/-- The probability of A hitting the enemy plane -/
def prob_A_hit : ℝ := 0.6

/-- The probability of B hitting the enemy plane -/
def prob_B_hit : ℝ := 0.5

/-- The probability that the enemy plane is hit by at least one of A or B -/
def prob_plane_hit : ℝ := 1 - (1 - prob_A_hit) * (1 - prob_B_hit)

theorem enemy_plane_hit_probability :
  prob_plane_hit = 0.8 :=
sorry

end enemy_plane_hit_probability_l1960_196047


namespace hotel_room_allocation_l1960_196061

theorem hotel_room_allocation (total_people : ℕ) (small_room_capacity : ℕ) 
  (num_small_rooms : ℕ) (h1 : total_people = 26) (h2 : small_room_capacity = 2) 
  (h3 : num_small_rooms = 1) :
  ∃ (large_room_capacity : ℕ),
    large_room_capacity = 12 ∧
    large_room_capacity > 0 ∧
    (total_people - num_small_rooms * small_room_capacity) % large_room_capacity = 0 ∧
    ∀ (x : ℕ), x > large_room_capacity → 
      (total_people - num_small_rooms * small_room_capacity) % x ≠ 0 :=
by sorry

end hotel_room_allocation_l1960_196061


namespace quadratic_inequality_solution_sets_l1960_196000

theorem quadratic_inequality_solution_sets 
  (a b : ℝ) 
  (h : Set.Icc (-2 : ℝ) 1 = {x : ℝ | a * x^2 - x + b ≥ 0}) : 
  Set.Icc (-1/2 : ℝ) 1 = {x : ℝ | b * x^2 - x + a ≤ 0} := by
sorry

end quadratic_inequality_solution_sets_l1960_196000


namespace temperature_calculation_l1960_196060

theorem temperature_calculation (T₁ T₂ : ℝ) : 
  2.24 * T₁ = 1.1 * 2 * 298 ∧ 1.76 * T₂ = 1.1 * 2 * 298 → 
  T₁ = 292.7 ∧ T₂ = 372.5 := by
  sorry

end temperature_calculation_l1960_196060


namespace quadratic_inequality_solution_l1960_196039

theorem quadratic_inequality_solution (a : ℝ) :
  let solution_set := {x : ℝ | 12 * x^2 - a * x - a^2 < 0}
  if a > 0 then
    solution_set = {x : ℝ | -a/4 < x ∧ x < a/3}
  else if a = 0 then
    solution_set = ∅
  else
    solution_set = {x : ℝ | a/3 < x ∧ x < -a/4} :=
by sorry

end quadratic_inequality_solution_l1960_196039


namespace quadratic_discriminant_l1960_196057

theorem quadratic_discriminant (a b c : ℝ) (x₁ x₂ : ℝ) :
  (∀ x, a * x^2 + b * x + c = a * (x - x₁) * (x - x₂)) →
  |x₂ - x₁| = 2 →
  b^2 - 4*a*c = 4 := by
sorry

end quadratic_discriminant_l1960_196057


namespace student_distribution_count_l1960_196015

/-- The number of ways to distribute 5 students into three groups -/
def distribute_students : ℕ :=
  -- The actual distribution logic would go here
  80

/-- The conditions for the distribution -/
def valid_distribution (a b c : ℕ) : Prop :=
  a + b + c = 5 ∧ a ≥ 2 ∧ b ≥ 1 ∧ c ≥ 1

theorem student_distribution_count :
  ∃ (a b c : ℕ), valid_distribution a b c ∧
  (∀ (x y z : ℕ), valid_distribution x y z → x + y + z = 5) ∧
  distribute_students = 80 :=
sorry

end student_distribution_count_l1960_196015


namespace complement_intersection_theorem_l1960_196062

def U : Set Nat := {1, 2, 3, 4}
def A : Set Nat := {1, 3, 4}
def B : Set Nat := {2, 4}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {2} := by sorry

end complement_intersection_theorem_l1960_196062


namespace fraction_problem_l1960_196097

theorem fraction_problem (N : ℝ) (F : ℝ) : 
  F * (1/4 * N) = 15 ∧ (3/10) * N = 54 → F = 1/3 :=
by sorry

end fraction_problem_l1960_196097


namespace gridiron_club_members_l1960_196022

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 6

/-- The cost of a T-shirt in dollars -/
def tshirt_cost : ℕ := sock_cost + 7

/-- The cost of a helmet in dollars -/
def helmet_cost : ℕ := 2 * tshirt_cost

/-- The cost of equipment for one member in dollars -/
def member_cost : ℕ := sock_cost + tshirt_cost + helmet_cost

/-- The total expenditure for all members in dollars -/
def total_expenditure : ℕ := 4680

/-- The number of members in the club -/
def club_members : ℕ := total_expenditure / member_cost

theorem gridiron_club_members :
  club_members = 104 :=
sorry

end gridiron_club_members_l1960_196022


namespace problem_solution_l1960_196070

theorem problem_solution : 
  let expr := (1 / (1 + 24 / 4) - 5 / 9) * (3 / (2 + 5 / 7)) / (2 / (3 + 3 / 4)) + 2.25
  ∀ A : ℝ, expr = 4 → (1 / (1 + 24 / A) - 5 / 9 = 1 / (1 + 24 / 4) - 5 / 9) → A = 4 := by
  sorry

end problem_solution_l1960_196070


namespace not_necessarily_p_or_q_l1960_196092

theorem not_necessarily_p_or_q (h1 : ¬p) (h2 : ¬(p ∧ q)) : 
  ¬∀ (p q : Prop), p ∨ q := by
sorry

end not_necessarily_p_or_q_l1960_196092


namespace first_pass_bubble_sort_l1960_196058

def bubbleSortPass (list : List Int) : List Int :=
  list.zipWith (λ a b => if a > b then b else a) (list.drop 1 ++ [0])

theorem first_pass_bubble_sort :
  bubbleSortPass [8, 23, 12, 14, 39, 11] = [8, 12, 14, 23, 11, 39] := by
  sorry

end first_pass_bubble_sort_l1960_196058


namespace want_is_correct_choice_l1960_196066

/-- Represents the possible word choices for the sentence --/
inductive WordChoice
  | hope
  | search
  | want
  | charge

/-- Represents the context of the situation --/
structure Situation where
  duration : Nat
  location : String
  isSnowstorm : Bool
  lackOfSupplies : Bool

/-- Defines the correct word choice given a situation --/
def correctWordChoice (s : Situation) : WordChoice :=
  if s.duration ≥ 5 && s.location = "station" && s.isSnowstorm && s.lackOfSupplies then
    WordChoice.want
  else
    WordChoice.hope  -- Default choice, not relevant for this problem

/-- Theorem stating that 'want' is the correct word choice for the given situation --/
theorem want_is_correct_choice (s : Situation) 
  (h1 : s.duration = 5)
  (h2 : s.location = "station")
  (h3 : s.isSnowstorm = true)
  (h4 : s.lackOfSupplies = true) :
  correctWordChoice s = WordChoice.want := by
  sorry


end want_is_correct_choice_l1960_196066


namespace total_cost_is_36_l1960_196016

-- Define the cost per dose for each antibiotic
def cost_a : ℚ := 3
def cost_b : ℚ := 4.5

-- Define the number of doses per week for each antibiotic
def doses_a : ℕ := 3 * 2  -- 3 days, twice a day
def doses_b : ℕ := 4 * 1  -- 4 days, once a day

-- Define the discount rate and the number of doses required for the discount
def discount_rate : ℚ := 0.2
def discount_doses : ℕ := 10

-- Define the total cost function
def total_cost : ℚ :=
  min (doses_a * cost_a) (discount_doses * cost_a * (1 - discount_rate)) +
  doses_b * cost_b

-- Theorem statement
theorem total_cost_is_36 : total_cost = 36 := by
  sorry

end total_cost_is_36_l1960_196016


namespace cube_root_27_times_sixth_root_64_times_sqrt_9_l1960_196067

theorem cube_root_27_times_sixth_root_64_times_sqrt_9 :
  (27 : ℝ) ^ (1/3) * (64 : ℝ) ^ (1/6) * (9 : ℝ) ^ (1/2) = 18 := by
  sorry

end cube_root_27_times_sixth_root_64_times_sqrt_9_l1960_196067


namespace product_lower_bound_l1960_196010

theorem product_lower_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b = a + b + 3) :
  a * b ≥ (7 * Real.sqrt 3) / 2 := by
  sorry

end product_lower_bound_l1960_196010


namespace gala_hat_count_l1960_196065

theorem gala_hat_count (total_attendees : ℕ) 
  (women_fraction : ℚ) (women_hat_percent : ℚ) (men_hat_percent : ℚ)
  (h1 : total_attendees = 2400)
  (h2 : women_fraction = 2/3)
  (h3 : women_hat_percent = 30/100)
  (h4 : men_hat_percent = 12/100) : 
  ↑⌊women_fraction * total_attendees * women_hat_percent⌋ + 
  ↑⌊(1 - women_fraction) * total_attendees * men_hat_percent⌋ = 576 :=
by sorry

end gala_hat_count_l1960_196065


namespace bubble_radius_l1960_196072

/-- Given a hemisphere with radius 4∛2 cm that has the same volume as a spherical bubble,
    the radius of the original bubble is 4 cm. -/
theorem bubble_radius (r : ℝ) (R : ℝ) : 
  r = 4 * Real.rpow 2 (1/3) → -- radius of hemisphere
  (2/3) * Real.pi * r^3 = (4/3) * Real.pi * R^3 → -- volume equality
  R = 4 := by
sorry

end bubble_radius_l1960_196072


namespace rectangle_triangle_area_l1960_196056

/-- The area of a geometric figure formed by a rectangle and an additional triangle -/
theorem rectangle_triangle_area (a b : ℝ) (h : 0 < a ∧ a < b) :
  let diagonal := Real.sqrt (a^2 + b^2)
  let triangle_area := (b * diagonal) / 4
  let total_area := a * b + triangle_area
  total_area = a * b + (b * Real.sqrt (a^2 + b^2)) / 4 := by
  sorry

end rectangle_triangle_area_l1960_196056


namespace greater_number_proof_l1960_196053

theorem greater_number_proof (x y : ℝ) (h1 : x > y) (h2 : x > 0) (h3 : y > 0)
  (h4 : x * y = 2688) (h5 : (x + y) - (x - y) = 64) : x = 84 := by
  sorry

end greater_number_proof_l1960_196053


namespace smallest_integer_in_special_set_l1960_196009

theorem smallest_integer_in_special_set : ∃ (n : ℤ),
  (n + 6 > 2 * ((7 * n + 21) / 7)) ∧
  (∀ (m : ℤ), m < n → ¬(m + 6 > 2 * ((7 * m + 21) / 7))) →
  n = -1 := by sorry

end smallest_integer_in_special_set_l1960_196009


namespace special_line_equation_l1960_196073

/-- A line passing through a point with x-axis and y-axis intercepts that are opposite numbers -/
structure SpecialLine where
  -- The point through which the line passes
  point : ℝ × ℝ
  -- The equation of the line, represented as a function ℝ² → ℝ
  equation : ℝ → ℝ → ℝ
  -- Condition: The line passes through the given point
  passes_through_point : equation point.1 point.2 = 0
  -- Condition: The line has intercepts on x-axis and y-axis that are opposite numbers
  opposite_intercepts : ∃ (a : ℝ), (equation a 0 = 0 ∧ equation 0 (-a) = 0) ∨ 
                                   (equation (-a) 0 = 0 ∧ equation 0 a = 0)

/-- Theorem: The equation of the special line is either x - y - 7 = 0 or 2x + 5y = 0 -/
theorem special_line_equation (l : SpecialLine) (h : l.point = (5, -2)) :
  (l.equation = fun x y => x - y - 7) ∨ (l.equation = fun x y => 2*x + 5*y) := by
  sorry

end special_line_equation_l1960_196073


namespace complex_fraction_equality_l1960_196095

theorem complex_fraction_equality (z : ℂ) (h : z + Complex.I = 4 - Complex.I) :
  z / (4 + 2 * Complex.I) = (3 - 4 * Complex.I) / 5 := by
  sorry

end complex_fraction_equality_l1960_196095


namespace complex_subtraction_l1960_196091

theorem complex_subtraction (a b : ℂ) (h1 : a = 5 - 3*I) (h2 : b = 4 + 3*I) :
  a - 3*b = -7 - 12*I := by sorry

end complex_subtraction_l1960_196091


namespace eight_sided_dice_divisible_by_four_probability_l1960_196084

theorem eight_sided_dice_divisible_by_four_probability : 
  let dice_outcomes : Finset ℕ := Finset.range 8
  let divisible_by_four : Finset ℕ := {4, 8}
  let total_outcomes : ℕ := dice_outcomes.card * dice_outcomes.card
  let favorable_outcomes : ℕ := divisible_by_four.card * divisible_by_four.card
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 16 :=
by sorry

end eight_sided_dice_divisible_by_four_probability_l1960_196084


namespace guppy_count_theorem_l1960_196042

/-- Calculates the total number of guppies given initial count and two batches of baby guppies -/
def total_guppies (initial : ℕ) (first_batch : ℕ) (second_batch : ℕ) : ℕ :=
  initial + first_batch + second_batch

/-- Converts dozens to individual count -/
def dozens_to_count (dozens : ℕ) : ℕ :=
  dozens * 12

theorem guppy_count_theorem (initial : ℕ) (first_batch_dozens : ℕ) (second_batch : ℕ) 
  (h1 : initial = 7)
  (h2 : first_batch_dozens = 3)
  (h3 : second_batch = 9) :
  total_guppies initial (dozens_to_count first_batch_dozens) second_batch = 52 := by
  sorry

end guppy_count_theorem_l1960_196042


namespace complex_expression_equals_two_l1960_196094

theorem complex_expression_equals_two :
  (2023 - Real.pi) ^ 0 + (1/2)⁻¹ + |1 - Real.sqrt 3| - 2 * Real.sin (π/3) = 2 := by
  sorry

end complex_expression_equals_two_l1960_196094


namespace approximate_probability_of_high_quality_l1960_196083

def sample_sizes : List ℕ := [20, 50, 100, 200, 500, 1000, 1500, 2000]

def high_quality_counts : List ℕ := [19, 47, 91, 184, 462, 921, 1379, 1846]

def frequencies : List ℚ := [
  950/1000, 940/1000, 910/1000, 920/1000, 924/1000, 921/1000, 919/1000, 923/1000
]

theorem approximate_probability_of_high_quality (ε : ℚ) (hε : ε = 1/100) :
  ∃ (p : ℚ), abs (p - (List.sum frequencies / frequencies.length)) ≤ ε ∧ p = 92/100 := by
  sorry

end approximate_probability_of_high_quality_l1960_196083


namespace no_fib_rectangle_decomposition_l1960_196064

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- A square with side length that is a Fibonacci number -/
structure FibSquare where
  side : ℕ
  is_fib : ∃ n, fib n = side

/-- A rectangle composed of Fibonacci squares -/
structure FibRectangle where
  squares : List FibSquare
  different_sizes : ∀ i j, i ≠ j → (squares.get i).side ≠ (squares.get j).side
  at_least_two : squares.length ≥ 2

/-- The theorem stating that a rectangle cannot be composed of different-sized Fibonacci squares -/
theorem no_fib_rectangle_decomposition : ¬ ∃ (r : FibRectangle), True := by
  sorry

end no_fib_rectangle_decomposition_l1960_196064


namespace probability_is_9_128_l1960_196048

/-- Four points chosen uniformly at random on a circle -/
def random_points_on_circle : Type := Fin 4 → ℝ × ℝ

/-- The circle's center -/
def circle_center : ℝ × ℝ := (0, 0)

/-- Checks if three points form an obtuse triangle -/
def is_obtuse_triangle (p1 p2 p3 : ℝ × ℝ) : Prop := sorry

/-- The probability of no two points forming an obtuse triangle with the center -/
def probability_no_obtuse_triangle (points : random_points_on_circle) : ℝ := sorry

/-- Main theorem: The probability is 9/128 -/
theorem probability_is_9_128 :
  ∀ points : random_points_on_circle,
  probability_no_obtuse_triangle points = 9 / 128 := by sorry

end probability_is_9_128_l1960_196048


namespace wire_division_l1960_196020

theorem wire_division (total_feet : ℕ) (total_inches : ℕ) (num_parts : ℕ) 
  (h1 : total_feet = 5) 
  (h2 : total_inches = 4) 
  (h3 : num_parts = 4) 
  (h4 : ∀ (feet : ℕ), feet * 12 = feet * (1 : ℕ) * 12) :
  (total_feet * 12 + total_inches) / num_parts = 16 := by
  sorry

end wire_division_l1960_196020


namespace inscribed_cube_volume_l1960_196041

theorem inscribed_cube_volume (large_cube_edge : ℝ) (small_cube_edge : ℝ) (small_cube_volume : ℝ) : 
  large_cube_edge = 12 →
  small_cube_edge * Real.sqrt 3 = large_cube_edge →
  small_cube_volume = small_cube_edge ^ 3 →
  small_cube_volume = 192 * Real.sqrt 3 := by
  sorry

end inscribed_cube_volume_l1960_196041


namespace equation_solution_l1960_196050

theorem equation_solution :
  let f : ℝ → ℝ := λ x => x * (x - 3)^2 * (5 + x)
  {x : ℝ | f x = 0} = {0, 3, -5} := by
sorry

end equation_solution_l1960_196050


namespace divisible_by_24_l1960_196052

theorem divisible_by_24 (n : ℕ+) : ∃ k : ℤ, (n : ℤ)^4 + 2*(n : ℤ)^3 + 11*(n : ℤ)^2 + 10*(n : ℤ) = 24*k := by
  sorry

end divisible_by_24_l1960_196052


namespace shaded_area_of_tiled_floor_l1960_196005

/-- Calculates the shaded area of a floor with specific tiling pattern -/
theorem shaded_area_of_tiled_floor (floor_length floor_width tile_size : ℝ)
  (quarter_circle_radius : ℝ) :
  floor_length = 8 →
  floor_width = 10 →
  tile_size = 1 →
  quarter_circle_radius = 1/2 →
  (floor_length * floor_width) * (tile_size^2 - π * quarter_circle_radius^2) = 80 - 20 * π :=
by
  sorry

end shaded_area_of_tiled_floor_l1960_196005


namespace kim_payment_amount_l1960_196035

def meal_cost : ℝ := 10
def drink_cost : ℝ := 2.5
def tip_percentage : ℝ := 0.2
def change_received : ℝ := 5

theorem kim_payment_amount :
  let total_before_tip := meal_cost + drink_cost
  let tip := tip_percentage * total_before_tip
  let total_with_tip := total_before_tip + tip
  let payment_amount := total_with_tip + change_received
  payment_amount = 20 := by sorry

end kim_payment_amount_l1960_196035


namespace rhombus_perimeter_l1960_196019

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 52 := by
  sorry

end rhombus_perimeter_l1960_196019


namespace gcf_of_50_and_75_l1960_196096

theorem gcf_of_50_and_75 : Nat.gcd 50 75 = 25 := by
  sorry

end gcf_of_50_and_75_l1960_196096


namespace test_questions_l1960_196074

theorem test_questions (total_questions : ℕ) 
  (h1 : total_questions / 2 = (13 : ℕ) + (total_questions - 20) / 4)
  (h2 : total_questions ≥ 20) : total_questions = 32 := by
  sorry

end test_questions_l1960_196074


namespace fraction_to_fourth_power_l1960_196080

theorem fraction_to_fourth_power (a b : ℝ) (hb : b ≠ 0) :
  (2 * a / b) ^ 4 = 16 * a ^ 4 / b ^ 4 := by sorry

end fraction_to_fourth_power_l1960_196080


namespace complex_modulus_problem_l1960_196090

theorem complex_modulus_problem (z : ℂ) (h : (1 + 2*I)*z = 1 - I) : Complex.abs z = Real.sqrt 10 / 5 := by
  sorry

end complex_modulus_problem_l1960_196090


namespace rectangle_circles_radii_sum_l1960_196046

theorem rectangle_circles_radii_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (4 * b^2 + a^2) / (4 * b) + (4 * a^2 + b^2) / (4 * a) ≥ 5 * (a + b) / 4 :=
by sorry

end rectangle_circles_radii_sum_l1960_196046


namespace chocolate_division_l1960_196024

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (piles_for_shaina : ℕ) :
  total_chocolate = 72 / 7 →
  num_piles = 6 →
  piles_for_shaina = 2 →
  (total_chocolate / num_piles) * piles_for_shaina = 24 / 7 := by
  sorry

end chocolate_division_l1960_196024


namespace equation_solution_l1960_196077

theorem equation_solution : ∃! x : ℝ, (2 / (x - 1) = 3 / (x - 2)) ∧ (x = -1) := by
  sorry

end equation_solution_l1960_196077


namespace complex_equation_solution_l1960_196017

theorem complex_equation_solution (z : ℂ) : (z - 1) / (z + 1) = I → z = I := by
  sorry

end complex_equation_solution_l1960_196017


namespace equilateral_triangle_minimum_rotation_angle_l1960_196028

/-- An equilateral triangle is a polygon with three equal sides and three equal angles. -/
structure EquilateralTriangle where
  -- We don't need to define the structure completely for this problem

/-- A figure is rotationally symmetric if it can be rotated around a fixed point by a certain angle
    and coincide with its initial position. -/
class RotationallySymmetric (α : Type*) where
  is_rotationally_symmetric : α → Prop

/-- The minimum rotation angle is the smallest non-zero angle by which a rotationally symmetric
    figure can be rotated to coincide with itself. -/
def minimum_rotation_angle (α : Type*) [RotationallySymmetric α] (figure : α) : ℝ :=
  sorry

theorem equilateral_triangle_minimum_rotation_angle 
  (triangle : EquilateralTriangle) 
  [RotationallySymmetric EquilateralTriangle] 
  (h : RotationallySymmetric.is_rotationally_symmetric triangle) : 
  minimum_rotation_angle EquilateralTriangle triangle = 120 := by
  sorry

end equilateral_triangle_minimum_rotation_angle_l1960_196028


namespace triangle_inequalities_l1960_196089

/-- Triangle inequality theorems -/
theorem triangle_inequalities (a b c : ℝ) (S : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_area : S > 0) : 
  (a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S) ∧ 
  (a^2 + b^2 + c^2 - (a-b)^2 - (b-c)^2 - (c-a)^2 ≥ 4 * Real.sqrt 3 * S) ∧
  ((a^2 + b^2 + c^2 = 4 * Real.sqrt 3 * S ∨ 
    a^2 + b^2 + c^2 - (a-b)^2 - (b-c)^2 - (c-a)^2 = 4 * Real.sqrt 3 * S) ↔ 
   a = b ∧ b = c) := by
  sorry

end triangle_inequalities_l1960_196089


namespace valid_schedules_l1960_196033

/-- Number of periods in a day -/
def total_periods : ℕ := 8

/-- Number of periods in the morning -/
def morning_periods : ℕ := 5

/-- Number of periods in the afternoon -/
def afternoon_periods : ℕ := 3

/-- Number of classes to teach -/
def classes_to_teach : ℕ := 3

/-- Calculate the number of ways to arrange n items taken k at a time -/
def arrange (n k : ℕ) : ℕ := sorry

/-- The number of valid teaching schedules -/
theorem valid_schedules : 
  arrange total_periods classes_to_teach - 
  (morning_periods * arrange morning_periods classes_to_teach) - 
  arrange afternoon_periods classes_to_teach = 312 := by sorry

end valid_schedules_l1960_196033


namespace max_product_sum_2000_l1960_196071

theorem max_product_sum_2000 :
  ∃ (x : ℤ), ∀ (y : ℤ), x * (2000 - x) ≥ y * (2000 - y) ∧ x * (2000 - x) = 1000000 :=
by sorry

end max_product_sum_2000_l1960_196071


namespace roots_of_polynomial_l1960_196045

/-- The polynomial we're considering -/
def p (x : ℝ) : ℝ := x^2 - 9

/-- The proposed factorization of the polynomial -/
def f (x : ℝ) : ℝ := (x - 3) * (x + 3)

theorem roots_of_polynomial :
  (∀ x : ℝ, p x = 0 ↔ x = 3 ∨ x = -3) ∧
  (∀ x : ℝ, p x = f x) :=
sorry

end roots_of_polynomial_l1960_196045


namespace units_digit_sum_cubes_l1960_196079

theorem units_digit_sum_cubes : (24^3 + 42^3) % 10 = 2 := by
  sorry

end units_digit_sum_cubes_l1960_196079


namespace error_percentage_l1960_196030

theorem error_percentage (y : ℝ) (h : y > 0) : 
  (|5 * y - y / 4| / (5 * y)) * 100 = 95 := by
  sorry

end error_percentage_l1960_196030


namespace average_habitable_land_per_person_approx_l1960_196004

-- Define the given constants
def total_population : ℕ := 281000000
def total_land_area : ℝ := 3797000
def habitable_land_percentage : ℝ := 0.8
def feet_per_mile : ℕ := 5280

-- Theorem statement
theorem average_habitable_land_per_person_approx :
  let habitable_land_area : ℝ := total_land_area * habitable_land_percentage
  let total_habitable_sq_feet : ℝ := habitable_land_area * (feet_per_mile ^ 2 : ℝ)
  let avg_sq_feet_per_person : ℝ := total_habitable_sq_feet / total_population
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1000 ∧ |avg_sq_feet_per_person - 300000| < ε :=
by sorry

end average_habitable_land_per_person_approx_l1960_196004


namespace distance_ratio_in_pyramid_l1960_196021

/-- A regular square pyramid with vertex P and base ABCD -/
structure RegularSquarePyramid where
  base_side_length : ℝ
  height : ℝ

/-- A point inside the base of the pyramid -/
structure PointInBase where
  x : ℝ
  y : ℝ

/-- Sum of distances from a point to all faces of the pyramid -/
def sum_distances_to_faces (p : RegularSquarePyramid) (e : PointInBase) : ℝ := sorry

/-- Sum of distances from a point to all edges of the base -/
def sum_distances_to_base_edges (p : RegularSquarePyramid) (e : PointInBase) : ℝ := sorry

/-- The main theorem stating the ratio of distances -/
theorem distance_ratio_in_pyramid (p : RegularSquarePyramid) (e : PointInBase) 
  (h_centroid : e ≠ PointInBase.mk (p.base_side_length / 2) (p.base_side_length / 2)) :
  sum_distances_to_faces p e / sum_distances_to_base_edges p e = 
    8 * Real.sqrt (p.height^2 + p.base_side_length^2 / 2) / p.base_side_length := by
  sorry

end distance_ratio_in_pyramid_l1960_196021


namespace dumpling_selection_probability_l1960_196025

/-- The number of dumplings of each kind in the pot -/
def dumplings_per_kind : ℕ := 5

/-- The number of different kinds of dumplings -/
def kinds_of_dumplings : ℕ := 3

/-- The total number of dumplings in the pot -/
def total_dumplings : ℕ := dumplings_per_kind * kinds_of_dumplings

/-- The number of dumplings to be selected -/
def selected_dumplings : ℕ := 4

/-- The probability of selecting at least one dumpling of each kind -/
def probability_at_least_one_of_each : ℚ := 50 / 91

theorem dumpling_selection_probability :
  (Nat.choose total_dumplings selected_dumplings *
   probability_at_least_one_of_each : ℚ) =
  (Nat.choose kinds_of_dumplings 1 *
   Nat.choose dumplings_per_kind 2 *
   Nat.choose dumplings_per_kind 1 *
   Nat.choose dumplings_per_kind 1 : ℚ) := by
  sorry

end dumpling_selection_probability_l1960_196025


namespace product_equals_three_l1960_196032

theorem product_equals_three : 
  (∀ a b c : ℝ, a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)) → 
  6 * 15 * 3 = 3 := by sorry

end product_equals_three_l1960_196032


namespace angle_bisector_of_lines_l1960_196068

-- Define the two lines
def L₁ (x y : ℝ) : Prop := 4 * x - 3 * y + 1 = 0
def L₂ (x y : ℝ) : Prop := 12 * x + 5 * y + 13 = 0

-- Define the angle bisector
def angle_bisector (x y : ℝ) : Prop := 56 * x - 7 * y + 39 = 0

-- Theorem statement
theorem angle_bisector_of_lines :
  ∀ x y : ℝ, angle_bisector x y ↔ (L₁ x y ∧ L₂ x y → ∃ t : ℝ, t > 0 ∧ 
    abs ((4 * x - 3 * y + 1) / (12 * x + 5 * y + 13)) = t) :=
sorry

end angle_bisector_of_lines_l1960_196068


namespace property_price_reduction_l1960_196018

/-- Represents the price reduction scenario of a property over two years -/
theorem property_price_reduction (x : ℝ) : 
  (20000 : ℝ) * (1 - x)^2 = 16200 ↔ 
  (∃ (initial_price final_price : ℝ), 
    initial_price = 20000 ∧ 
    final_price = 16200 ∧ 
    final_price = initial_price * (1 - x)^2 ∧ 
    0 ≤ x ∧ x < 1) :=
by sorry

end property_price_reduction_l1960_196018


namespace class_size_with_error_l1960_196037

/-- Represents a class with a marking error -/
structure ClassWithError where
  n : ℕ  -- number of pupils
  S : ℕ  -- correct sum of marks
  wrong_mark : ℕ  -- wrongly entered mark
  correct_mark : ℕ  -- correct mark

/-- The conditions of the problem -/
def problem_conditions (c : ClassWithError) : Prop :=
  c.wrong_mark = 79 ∧
  c.correct_mark = 45 ∧
  (c.S + (c.wrong_mark - c.correct_mark)) / c.n = 3/2 * (c.S / c.n)

/-- The theorem stating the solution -/
theorem class_size_with_error (c : ClassWithError) :
  problem_conditions c → c.n = 68 :=
by sorry

end class_size_with_error_l1960_196037


namespace squirrel_is_red_l1960_196098

-- Define the color of the squirrel
inductive SquirrelColor
  | Red
  | Gray

-- Define the state of a hollow
inductive HollowState
  | Empty
  | HasNuts

-- Define the structure for the two hollows
structure Hollows :=
  (first : HollowState)
  (second : HollowState)

-- Define the statements made by the squirrel
def statement1 (h : Hollows) : Prop :=
  h.first = HollowState.Empty

def statement2 (h : Hollows) : Prop :=
  h.first = HollowState.HasNuts ∨ h.second = HollowState.HasNuts

-- Define the truthfulness of the squirrel based on its color
def isTruthful (c : SquirrelColor) : Prop :=
  match c with
  | SquirrelColor.Red => True
  | SquirrelColor.Gray => False

-- Theorem: The squirrel must be red
theorem squirrel_is_red (h : Hollows) :
  (isTruthful SquirrelColor.Red → statement1 h ∧ statement2 h) ∧
  (isTruthful SquirrelColor.Gray → ¬(statement1 h) ∧ ¬(statement2 h)) →
  ∃ (h : Hollows), statement1 h ∧ statement2 h →
  SquirrelColor.Red = SquirrelColor.Red :=
by sorry

end squirrel_is_red_l1960_196098


namespace arithmetic_sequence_sum_l1960_196078

/-- Given an arithmetic sequence where the eighth term is 20 and the common difference is 3,
    prove that the sum of the first three terms is 6. -/
theorem arithmetic_sequence_sum (a : ℕ → ℤ) :
  (∀ n, a (n + 1) - a n = 3) →  -- Common difference is 3
  a 8 = 20 →                   -- Eighth term is 20
  a 1 + a 2 + a 3 = 6 :=        -- Sum of first three terms is 6
by
  sorry

end arithmetic_sequence_sum_l1960_196078


namespace min_value_quadratic_form_l1960_196063

theorem min_value_quadratic_form (x y : ℝ) : 2 * x^2 + 3 * x * y + 4 * y^2 + 5 ≥ 5 := by
  sorry

end min_value_quadratic_form_l1960_196063


namespace ratio_equality_solution_l1960_196049

theorem ratio_equality_solution (x : ℝ) : (0.75 / x = 5 / 9) → x = 1.35 := by
  sorry

end ratio_equality_solution_l1960_196049


namespace no_snow_no_rain_probability_l1960_196040

theorem no_snow_no_rain_probability 
  (prob_snow : ℚ) 
  (prob_rain : ℚ) 
  (days : ℕ) 
  (h1 : prob_snow = 2/3) 
  (h2 : prob_rain = 1/2) 
  (h3 : days = 5) : 
  (1 - prob_snow) * (1 - prob_rain) ^ days = 1/7776 :=
sorry

end no_snow_no_rain_probability_l1960_196040


namespace divisibility_from_point_distribution_l1960_196013

theorem divisibility_from_point_distribution (k n : ℕ) (h_pos_k : k > 0) (h_pos_n : n > 0) (h_k_le_n : k ≤ n)
  (points : Finset ℝ) (h_card : points.card = n)
  (h_divisible : ∀ x ∈ points, (points.filter (λ y => |y - x| ≤ 1)).card % k = 0) :
  k ∣ n := by
sorry

end divisibility_from_point_distribution_l1960_196013


namespace total_games_is_32_l1960_196023

/-- The number of games won by Jerry -/
def jerry_wins : ℕ := 7

/-- The number of games won by Dave -/
def dave_wins : ℕ := jerry_wins + 3

/-- The number of games won by Ken -/
def ken_wins : ℕ := dave_wins + 5

/-- The total number of games played -/
def total_games : ℕ := jerry_wins + dave_wins + ken_wins

theorem total_games_is_32 : total_games = 32 := by
  sorry

end total_games_is_32_l1960_196023


namespace gum_distribution_l1960_196014

theorem gum_distribution (num_cousins : ℕ) (gum_per_cousin : ℕ) : 
  num_cousins = 4 → gum_per_cousin = 5 → num_cousins * gum_per_cousin = 20 := by
  sorry

end gum_distribution_l1960_196014


namespace smallest_n_congruence_l1960_196081

theorem smallest_n_congruence (n : ℕ) : 
  (0 ≤ n ∧ n < 53 ∧ 50 * n % 53 = 47 % 53) → n = 2 := by
  sorry

end smallest_n_congruence_l1960_196081


namespace area_traced_by_rolling_triangle_l1960_196099

/-- The area traced out by rolling an equilateral triangle -/
theorem area_traced_by_rolling_triangle (side_length : ℝ) (h : side_length = 6) :
  let triangle_height : ℝ := side_length * Real.sqrt 3 / 2
  let arc_length : ℝ := π * side_length / 3
  let rectangle_area : ℝ := side_length * arc_length
  let quarter_circle_area : ℝ := π * side_length^2 / 4
  rectangle_area + quarter_circle_area = 21 * π := by
  sorry

#check area_traced_by_rolling_triangle

end area_traced_by_rolling_triangle_l1960_196099


namespace inner_circles_radii_l1960_196051

/-- An isosceles triangle with a 120° angle and an inscribed circle of radius R -/
structure IsoscelesTriangle120 where
  R : ℝ
  R_pos : R > 0

/-- Two equal circles inside the triangle that touch each other,
    where each circle touches one leg of the triangle and the inscribed circle -/
structure InnerCircles (t : IsoscelesTriangle120) where
  radius : ℝ
  radius_pos : radius > 0

/-- The theorem stating the possible radii of the inner circles -/
theorem inner_circles_radii (t : IsoscelesTriangle120) (c : InnerCircles t) :
  c.radius = t.R / 3 ∨ c.radius = (3 - 2 * Real.sqrt 2) / 3 * t.R :=
by sorry

end inner_circles_radii_l1960_196051


namespace max_container_volume_height_for_max_volume_l1960_196085

/-- Represents the volume of a rectangular container as a function of one side length --/
def containerVolume (x : ℝ) : ℝ := x * (x + 0.5) * (3.45 - x)

/-- The total length of the steel strip used for the container frame --/
def totalLength : ℝ := 14.8

/-- Theorem stating the maximum volume of the container --/
theorem max_container_volume :
  ∃ (x : ℝ), x > 0 ∧ x < 3.45 ∧
  containerVolume x = 3.675 ∧
  ∀ (y : ℝ), y > 0 → y < 3.45 → containerVolume y ≤ containerVolume x :=
sorry

/-- Theorem stating the height that achieves the maximum volume --/
theorem height_for_max_volume :
  ∃ (x : ℝ), x > 0 ∧ x < 3.45 ∧
  containerVolume x = 3.675 ∧
  (3.45 - x) = 2.45 :=
sorry

end max_container_volume_height_for_max_volume_l1960_196085


namespace final_tomato_count_l1960_196006

def cherry_tomatoes (initial : ℕ) : ℕ :=
  let after_first_birds := initial - (initial / 3)
  let after_second_birds := after_first_birds - (after_first_birds * 2 / 5)
  let after_growth := after_second_birds + (after_second_birds / 2)
  let after_more_growth := after_growth + 4
  after_more_growth - (after_more_growth / 4)

theorem final_tomato_count : cherry_tomatoes 21 = 13 := by
  sorry

end final_tomato_count_l1960_196006


namespace investment_ratio_l1960_196027

/-- Prove that given two equal investments of $12000, one at 11% and one at 9%,
    the ratio of these investments is 1:1 if the total interest after 1 year is $2400. -/
theorem investment_ratio (investment_11 investment_9 : ℝ) 
  (h1 : investment_11 = 12000)
  (h2 : investment_9 = 12000)
  (h3 : 0.11 * investment_11 + 0.09 * investment_9 = 2400) :
  investment_11 / investment_9 = 1 := by
sorry

end investment_ratio_l1960_196027


namespace probability_point_in_circle_l1960_196082

/-- The probability of a randomly selected point from a square with side length 6
    being inside or on a circle with radius 2 centered at the center of the square -/
theorem probability_point_in_circle (square_side : ℝ) (circle_radius : ℝ) : 
  square_side = 6 →
  circle_radius = 2 →
  (circle_radius^2 * Real.pi) / square_side^2 = Real.pi / 9 := by
  sorry

end probability_point_in_circle_l1960_196082


namespace seashells_found_joan_seashells_l1960_196007

theorem seashells_found (given_to_mike : ℕ) (has_now : ℕ) : ℕ :=
  given_to_mike + has_now

theorem joan_seashells : seashells_found 63 16 = 79 := by
  sorry

end seashells_found_joan_seashells_l1960_196007


namespace factorial_ratio_eleven_nine_l1960_196069

theorem factorial_ratio_eleven_nine : Nat.factorial 11 / Nat.factorial 9 = 110 := by
  sorry

end factorial_ratio_eleven_nine_l1960_196069


namespace marias_initial_savings_l1960_196012

def sweater_price : ℕ := 30
def scarf_price : ℕ := 20
def num_sweaters : ℕ := 6
def num_scarves : ℕ := 6
def remaining_money : ℕ := 200

theorem marias_initial_savings :
  (sweater_price * num_sweaters + scarf_price * num_scarves + remaining_money) = 500 := by
  sorry

end marias_initial_savings_l1960_196012


namespace remaining_cakes_l1960_196055

def cakes_per_day : ℕ := 4
def baking_days : ℕ := 6
def eating_frequency : ℕ := 2

def total_baked (cakes_per_day baking_days : ℕ) : ℕ :=
  cakes_per_day * baking_days

def cakes_eaten (baking_days eating_frequency : ℕ) : ℕ :=
  baking_days / eating_frequency

theorem remaining_cakes :
  total_baked cakes_per_day baking_days - cakes_eaten baking_days eating_frequency = 21 :=
by sorry

end remaining_cakes_l1960_196055


namespace monkey_swing_theorem_l1960_196038

/-- The distance a monkey swings in a given time -/
def monkey_swing_distance (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time * 60

/-- Theorem: A monkey swinging at 1.2 m/s for 30 minutes travels 2160 meters -/
theorem monkey_swing_theorem :
  monkey_swing_distance 1.2 30 = 2160 :=
by sorry

end monkey_swing_theorem_l1960_196038


namespace min_sum_box_dimensions_l1960_196026

theorem min_sum_box_dimensions :
  ∀ (l w h : ℕ+),
    l * w * h = 2002 →
    ∀ (a b c : ℕ+),
      a * b * c = 2002 →
      l + w + h ≤ a + b + c →
      l + w + h = 38 :=
by sorry

end min_sum_box_dimensions_l1960_196026


namespace container_dimensions_l1960_196088

theorem container_dimensions (a b c : ℝ) :
  a * b * 16 = 2400 →
  a * c * 10 = 2400 →
  b * c * 9.6 = 2400 →
  a = 12 ∧ b = 12.5 ∧ c = 20 := by
sorry

end container_dimensions_l1960_196088


namespace initial_interest_rate_is_45_percent_l1960_196003

/-- Given an initial deposit amount and two interest scenarios, 
    prove that the initial interest rate is 45% --/
theorem initial_interest_rate_is_45_percent 
  (P : ℝ) -- Principal amount (initial deposit)
  (r : ℝ) -- Initial interest rate (as a percentage)
  (h1 : P * r / 100 = 405) -- Interest at initial rate is 405
  (h2 : P * (r + 5) / 100 = 450) -- Interest at (r + 5)% is 450
  : r = 45 := by
sorry

end initial_interest_rate_is_45_percent_l1960_196003


namespace tan_alpha_eq_neg_one_l1960_196031

theorem tan_alpha_eq_neg_one (α : ℝ) (h : Real.sin (π/6 - α) = Real.cos (π/6 + α)) : 
  Real.tan α = -1 := by
  sorry

end tan_alpha_eq_neg_one_l1960_196031
