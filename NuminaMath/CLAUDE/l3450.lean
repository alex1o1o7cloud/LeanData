import Mathlib

namespace prob_at_least_one_female_is_four_fifths_l3450_345034

/-- Represents the number of students in each category -/
structure StudentCounts where
  total : ℕ
  maleHigh : ℕ
  femaleHigh : ℕ
  selected : ℕ
  final : ℕ

/-- Calculates the probability of selecting at least one female student -/
def probAtLeastOneFemale (counts : StudentCounts) : ℚ :=
  1 - (counts.maleHigh.choose counts.final) / (counts.maleHigh + counts.femaleHigh).choose counts.final

/-- The main theorem stating the probability of selecting at least one female student -/
theorem prob_at_least_one_female_is_four_fifths (counts : StudentCounts) 
  (h1 : counts.total = 200)
  (h2 : counts.maleHigh = 100)
  (h3 : counts.femaleHigh = 50)
  (h4 : counts.selected = 6)
  (h5 : counts.final = 3) :
  probAtLeastOneFemale counts = 4/5 := by
  sorry


end prob_at_least_one_female_is_four_fifths_l3450_345034


namespace inequality_system_solution_set_l3450_345098

theorem inequality_system_solution_set :
  {x : ℝ | x - 1 < 0 ∧ x + 1 > 0} = {x : ℝ | -1 < x ∧ x < 1} :=
by sorry

end inequality_system_solution_set_l3450_345098


namespace permutations_of_three_objects_l3450_345078

theorem permutations_of_three_objects (n : ℕ) (h : n = 3) : Nat.factorial n = 6 := by
  sorry

end permutations_of_three_objects_l3450_345078


namespace circle_equation_l3450_345056

/-- The standard equation of a circle with center (2, -2) passing through the origin -/
theorem circle_equation : ∀ (x y : ℝ), 
  (x - 2)^2 + (y + 2)^2 = 8 ↔ 
  (x - 2)^2 + (y + 2)^2 = (2 - 0)^2 + (-2 - 0)^2 := by
sorry

end circle_equation_l3450_345056


namespace scarves_per_box_chloes_scarves_l3450_345016

theorem scarves_per_box (num_boxes : ℕ) (mittens_per_box : ℕ) (total_clothing : ℕ) : ℕ :=
  let total_mittens := num_boxes * mittens_per_box
  let total_scarves := total_clothing - total_mittens
  let scarves_per_box := total_scarves / num_boxes
  scarves_per_box

theorem chloes_scarves :
  scarves_per_box 4 6 32 = 2 := by
  sorry

end scarves_per_box_chloes_scarves_l3450_345016


namespace cody_additional_tickets_l3450_345049

/-- Calculates the number of additional tickets won given initial tickets, tickets spent, and final tickets. -/
def additional_tickets_won (initial_tickets spent_tickets final_tickets : ℕ) : ℕ :=
  final_tickets - (initial_tickets - spent_tickets)

/-- Proves that Cody won 6 additional tickets given the problem conditions. -/
theorem cody_additional_tickets :
  let initial_tickets := 49
  let spent_tickets := 25
  let final_tickets := 30
  additional_tickets_won initial_tickets spent_tickets final_tickets = 6 := by
  sorry

end cody_additional_tickets_l3450_345049


namespace problem_statement_l3450_345068

theorem problem_statement (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_prod : x * y * z = 1)
  (h_eq1 : x + 1 / z = 5)
  (h_eq2 : y + 1 / x = 29) :
  z + 1 / y = 1 / 4 := by
  sorry

end problem_statement_l3450_345068


namespace union_of_positive_and_less_than_one_is_reals_l3450_345022

theorem union_of_positive_and_less_than_one_is_reals :
  let A : Set ℝ := {x | x > 0}
  let B : Set ℝ := {x | x < 1}
  A ∪ B = Set.univ := by sorry

end union_of_positive_and_less_than_one_is_reals_l3450_345022


namespace largest_common_divisor_408_330_l3450_345013

theorem largest_common_divisor_408_330 : Nat.gcd 408 330 = 6 := by
  sorry

end largest_common_divisor_408_330_l3450_345013


namespace luigi_pizza_count_l3450_345089

/-- The number of pizzas Luigi bought -/
def num_pizzas : ℕ := 4

/-- The total cost of pizzas in dollars -/
def total_cost : ℕ := 80

/-- The number of pieces each pizza is cut into -/
def pieces_per_pizza : ℕ := 5

/-- The cost of each piece of pizza in dollars -/
def cost_per_piece : ℕ := 4

/-- Theorem stating that the number of pizzas Luigi bought is 4 -/
theorem luigi_pizza_count :
  num_pizzas = 4 ∧
  total_cost = 80 ∧
  pieces_per_pizza = 5 ∧
  cost_per_piece = 4 ∧
  total_cost = num_pizzas * pieces_per_pizza * cost_per_piece :=
by sorry

end luigi_pizza_count_l3450_345089


namespace turquoise_color_perception_l3450_345074

theorem turquoise_color_perception (total : ℕ) (more_blue : ℕ) (both : ℕ) (neither : ℕ) :
  total = 150 →
  more_blue = 90 →
  both = 40 →
  neither = 20 →
  ∃ (more_green : ℕ), more_green = 80 ∧ 
    more_green + more_blue - both + neither = total :=
by sorry

end turquoise_color_perception_l3450_345074


namespace complex_product_theorem_l3450_345057

theorem complex_product_theorem : 
  let z₁ : ℂ := 2 + Complex.I
  let z₂ : ℂ := 1 - Complex.I
  z₁ * z₂ = 3 - Complex.I := by
  sorry

end complex_product_theorem_l3450_345057


namespace tangent_segment_difference_l3450_345054

/-- Represents a quadrilateral inscribed in a circle with an inscribed circle --/
structure InscribedQuadrilateral where
  /-- Side lengths of the quadrilateral --/
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  /-- Proof that the quadrilateral is inscribed in a circle --/
  inscribed_in_circle : True
  /-- Proof that there's a circle inscribed in the quadrilateral --/
  has_inscribed_circle : True

/-- Theorem about the difference of segments created by the inscribed circle's tangency point --/
theorem tangent_segment_difference (q : InscribedQuadrilateral)
    (h1 : q.side1 = 50)
    (h2 : q.side2 = 80)
    (h3 : q.side3 = 140)
    (h4 : q.side4 = 120) :
    ∃ (x y : ℝ), x + y = 140 ∧ |x - y| = 19 := by
  sorry


end tangent_segment_difference_l3450_345054


namespace train_average_speed_l3450_345020

theorem train_average_speed (distance1 distance2 time1 time2 : ℝ) 
  (h1 : distance1 = 250)
  (h2 : distance2 = 350)
  (h3 : time1 = 2)
  (h4 : time2 = 4) :
  (distance1 + distance2) / (time1 + time2) = 100 := by
  sorry

end train_average_speed_l3450_345020


namespace triangle_circles_theorem_l3450_345064

/-- Represents a triangular arrangement of circles -/
structure TriangularArrangement where
  total_circles : ℕ
  longest_side_length : ℕ
  shorter_side_rows : List ℕ

/-- Calculates the number of ways to choose three consecutive circles along the longest side -/
def longest_side_choices (arr : TriangularArrangement) : ℕ :=
  (arr.longest_side_length * (arr.longest_side_length + 1)) / 2

/-- Calculates the number of ways to choose three consecutive circles along a shorter side -/
def shorter_side_choices (arr : TriangularArrangement) : ℕ :=
  arr.shorter_side_rows.sum

/-- Calculates the total number of ways to choose three consecutive circles in any direction -/
def total_choices (arr : TriangularArrangement) : ℕ :=
  longest_side_choices arr + 2 * shorter_side_choices arr

/-- The main theorem stating that for the given arrangement, there are 57 ways to choose three consecutive circles -/
theorem triangle_circles_theorem (arr : TriangularArrangement) 
  (h1 : arr.total_circles = 33)
  (h2 : arr.longest_side_length = 6)
  (h3 : arr.shorter_side_rows = [4, 4, 4, 3, 2, 1]) :
  total_choices arr = 57 := by
  sorry


end triangle_circles_theorem_l3450_345064


namespace expand_expression_l3450_345046

theorem expand_expression (x : ℝ) : -2 * (x + 3) * (x - 2) * (x + 1) = -2*x^3 - 4*x^2 + 10*x + 12 := by
  sorry

end expand_expression_l3450_345046


namespace average_b_c_l3450_345040

theorem average_b_c (a b c : ℝ) 
  (h1 : (a + b) / 2 = 80) 
  (h2 : a - c = 200) : 
  (b + c) / 2 = -20 := by
sorry

end average_b_c_l3450_345040


namespace remaining_speed_calculation_l3450_345094

/-- Calculates the average speed for the remaining part of a trip given:
    - The fraction of the trip completed in the first part
    - The speed of the first part of the trip
    - The average speed for the entire trip
-/
theorem remaining_speed_calculation 
  (first_part_fraction : Real) 
  (first_part_speed : Real) 
  (total_average_speed : Real) :
  first_part_fraction = 0.4 →
  first_part_speed = 40 →
  total_average_speed = 50 →
  (1 - first_part_fraction) * total_average_speed / 
    (1 - first_part_fraction * total_average_speed / first_part_speed) = 60 := by
  sorry

#check remaining_speed_calculation

end remaining_speed_calculation_l3450_345094


namespace mom_tshirt_packages_l3450_345053

/-- The number of packages mom will have when buying t-shirts -/
def packages_bought (shirts_per_package : ℕ) (total_shirts : ℕ) : ℕ :=
  total_shirts / shirts_per_package

/-- Theorem: Mom will have 3 packages when buying 39 t-shirts sold in packages of 13 -/
theorem mom_tshirt_packages :
  packages_bought 13 39 = 3 := by
  sorry

end mom_tshirt_packages_l3450_345053


namespace tv_price_increase_l3450_345033

theorem tv_price_increase (P : ℝ) (x : ℝ) (h : P > 0) :
  (0.80 * P + x / 100 * (0.80 * P) = 1.16 * P) → x = 45 :=
by
  sorry

end tv_price_increase_l3450_345033


namespace dennis_purchase_cost_l3450_345021

/-- The cost of Dennis's purchase after discount --/
def total_cost (pants_price sock_price : ℚ) (pants_quantity sock_quantity : ℕ) (discount : ℚ) : ℚ :=
  let discounted_pants_price := pants_price * (1 - discount)
  let discounted_sock_price := sock_price * (1 - discount)
  (discounted_pants_price * pants_quantity) + (discounted_sock_price * sock_quantity)

/-- Theorem stating the total cost of Dennis's purchase --/
theorem dennis_purchase_cost :
  total_cost 110 60 4 2 (30/100) = 392 := by
  sorry

end dennis_purchase_cost_l3450_345021


namespace sum_of_digits_of_N_l3450_345079

def N : ℕ := 10^3 + 10^4 + 10^5 + 10^6 + 10^7 + 10^8 + 10^9

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_of_N : sum_of_digits N = 7 := by
  sorry

end sum_of_digits_of_N_l3450_345079


namespace toothpicks_per_card_l3450_345081

theorem toothpicks_per_card 
  (total_cards : ℕ) 
  (unused_cards : ℕ) 
  (toothpick_boxes : ℕ) 
  (toothpicks_per_box : ℕ) 
  (h1 : total_cards = 52) 
  (h2 : unused_cards = 16) 
  (h3 : toothpick_boxes = 6) 
  (h4 : toothpicks_per_box = 450) :
  (toothpick_boxes * toothpicks_per_box) / (total_cards - unused_cards) = 75 :=
by
  sorry

end toothpicks_per_card_l3450_345081


namespace trig_inequality_l3450_345039

theorem trig_inequality : 
  let a := Real.sin (31 * π / 180)
  let b := Real.cos (58 * π / 180)
  let c := Real.tan (32 * π / 180)
  c > b ∧ b > a := by sorry

end trig_inequality_l3450_345039


namespace point_coordinates_l3450_345085

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance from a point to the x-axis -/
def distanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def distanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem stating the coordinates of point P given the conditions -/
theorem point_coordinates (p : Point) 
  (h1 : isInSecondQuadrant p)
  (h2 : distanceToXAxis p = 4)
  (h3 : distanceToYAxis p = 5) :
  p = Point.mk (-5) 4 := by
  sorry

end point_coordinates_l3450_345085


namespace y_increases_with_x_l3450_345011

theorem y_increases_with_x (m : ℝ) (x y : ℝ → ℝ) :
  (∀ t, y t = (m^2 + 2) * x t) →
  StrictMono y :=
sorry

end y_increases_with_x_l3450_345011


namespace art_cost_theorem_l3450_345000

def art_cost_problem (cost_A : ℝ) (cost_B : ℝ) (cost_C : ℝ) (cost_D : ℝ) : Prop :=
  let pieces_A := 3
  let pieces_B := 2
  let pieces_C := 3
  let pieces_D := 1
  let total_cost_A := cost_A * pieces_A
  let total_cost_B := cost_B * pieces_B
  let total_cost_C := cost_C * pieces_C
  let total_cost_D := cost_D * pieces_D
  let total_cost := total_cost_A + total_cost_B + total_cost_C + total_cost_D

  (total_cost_A = 45000) ∧
  (cost_B = cost_A * 1.25) ∧
  (cost_C = cost_A * 1.5) ∧
  (cost_D = total_cost_C * 2) ∧
  (total_cost = 285000)

theorem art_cost_theorem : ∃ cost_A cost_B cost_C cost_D, art_cost_problem cost_A cost_B cost_C cost_D :=
  sorry

end art_cost_theorem_l3450_345000


namespace arithmetic_progression_divisibility_l3450_345050

/-- Given three integers a, b, c that form an arithmetic progression with a common difference of 7,
    and one of them is divisible by 7, their product abc is divisible by 294. -/
theorem arithmetic_progression_divisibility (a b c : ℤ) 
  (h1 : b - a = 7)
  (h2 : c - b = 7)
  (h3 : (∃ k : ℤ, a = 7 * k) ∨ (∃ k : ℤ, b = 7 * k) ∨ (∃ k : ℤ, c = 7 * k)) :
  ∃ m : ℤ, a * b * c = 294 * m := by
  sorry

end arithmetic_progression_divisibility_l3450_345050


namespace find_n_l3450_345066

def vector_AB : Fin 2 → ℝ := ![2, 4]
def vector_BC (n : ℝ) : Fin 2 → ℝ := ![-2, 2*n]
def vector_AC : Fin 2 → ℝ := ![0, 2]

theorem find_n : ∃ n : ℝ, 
  (∀ i : Fin 2, vector_AB i + vector_BC n i = vector_AC i) ∧ n = -1 := by
  sorry

end find_n_l3450_345066


namespace area_outside_inscribed_angle_l3450_345009

theorem area_outside_inscribed_angle (R : ℝ) (h : R = 12) :
  let θ : ℝ := 120 * π / 180
  let sector_area := θ / (2 * π) * π * R^2
  let triangle_area := 1/2 * R^2 * Real.sin θ
  sector_area - triangle_area = 48 * π - 72 * Real.sqrt 3 := by
  sorry

end area_outside_inscribed_angle_l3450_345009


namespace sales_volume_decrease_and_may_prediction_l3450_345090

/-- Represents the monthly sales volume decrease rate -/
def monthly_decrease_rate : ℝ := 0.05

/-- Calculates the sales volume after n months given an initial volume and monthly decrease rate -/
def sales_volume (initial_volume : ℝ) (n : ℕ) : ℝ :=
  initial_volume * (1 - monthly_decrease_rate) ^ n

theorem sales_volume_decrease_and_may_prediction
  (january_volume : ℝ)
  (march_volume : ℝ)
  (h1 : january_volume = 6000)
  (h2 : march_volume = 5400)
  (h3 : sales_volume january_volume 2 = march_volume)
  : monthly_decrease_rate = 0.05 ∧ sales_volume january_volume 4 > 4500 := by
  sorry

#eval sales_volume 6000 4

end sales_volume_decrease_and_may_prediction_l3450_345090


namespace equation_solution_l3450_345073

theorem equation_solution (x : ℝ) : 
  (x / 5) / 3 = 15 / (x / 3) → x = 15 * Real.sqrt 3 ∨ x = -15 * Real.sqrt 3 := by
  sorry

end equation_solution_l3450_345073


namespace min_dials_for_lighting_l3450_345099

/-- A regular 12-sided polygon dial with numbers from 1 to 12 -/
structure Dial :=
  (numbers : Fin 12 → Fin 12)

/-- A stack of dials -/
def DialStack := List Dial

/-- The sum of numbers in a column of the dial stack -/
def columnSum (stack : DialStack) (column : Fin 12) : ℕ :=
  stack.foldr (λ dial acc => acc + dial.numbers column) 0

/-- Predicate for when the Christmas tree lights up -/
def lightsUp (stack : DialStack) : Prop :=
  ∀ i j : Fin 12, columnSum stack i % 12 = columnSum stack j % 12

/-- The theorem stating the minimum number of dials required -/
theorem min_dials_for_lighting : 
  ∀ n : ℕ, (∃ stack : DialStack, stack.length = n ∧ lightsUp stack) → n ≥ 12 :=
sorry

end min_dials_for_lighting_l3450_345099


namespace logarithmic_algebraic_equivalence_l3450_345060

theorem logarithmic_algebraic_equivalence : 
  ¬(∀ x : ℝ, (Real.log (x^2 - 4) = Real.log (4*x - 7)) ↔ (x^2 - 4 = 4*x - 7)) :=
by sorry

end logarithmic_algebraic_equivalence_l3450_345060


namespace average_speed_proof_l3450_345023

/-- Proves that the average speed of a trip is 32 km/h given the specified conditions -/
theorem average_speed_proof (total_distance : ℝ) (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) :
  total_distance = 60 →
  distance1 = 30 →
  speed1 = 48 →
  distance2 = 30 →
  speed2 = 24 →
  (total_distance / ((distance1 / speed1) + (distance2 / speed2))) = 32 := by
  sorry

end average_speed_proof_l3450_345023


namespace color_selection_problem_l3450_345038

/-- The number of ways to select k distinct items from a set of n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of colors available -/
def total_colors : ℕ := 9

/-- The number of colors to be selected -/
def colors_to_select : ℕ := 3

theorem color_selection_problem :
  choose total_colors colors_to_select = 84 := by
  sorry

end color_selection_problem_l3450_345038


namespace day_after_53_from_friday_l3450_345065

/-- Days of the week -/
inductive DayOfWeek
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday
  | sunday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.monday => DayOfWeek.tuesday
  | DayOfWeek.tuesday => DayOfWeek.wednesday
  | DayOfWeek.wednesday => DayOfWeek.thursday
  | DayOfWeek.thursday => DayOfWeek.friday
  | DayOfWeek.friday => DayOfWeek.saturday
  | DayOfWeek.saturday => DayOfWeek.sunday
  | DayOfWeek.sunday => DayOfWeek.monday

/-- Function to get the day of the week after a given number of days -/
def dayAfter (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => start
  | Nat.succ m => nextDay (dayAfter start m)

/-- Theorem stating that 53 days after Friday is Tuesday -/
theorem day_after_53_from_friday :
  dayAfter DayOfWeek.friday 53 = DayOfWeek.tuesday := by
  sorry


end day_after_53_from_friday_l3450_345065


namespace books_from_second_shop_l3450_345088

theorem books_from_second_shop
  (books_first_shop : ℕ)
  (cost_first_shop : ℕ)
  (cost_second_shop : ℕ)
  (average_price : ℕ)
  (h1 : books_first_shop = 42)
  (h2 : cost_first_shop = 520)
  (h3 : cost_second_shop = 248)
  (h4 : average_price = 12)
  : ∃ (books_second_shop : ℕ),
    (cost_first_shop + cost_second_shop) / (books_first_shop + books_second_shop) = average_price ∧
    books_second_shop = 22 := by
  sorry

#check books_from_second_shop

end books_from_second_shop_l3450_345088


namespace football_practice_missed_days_l3450_345008

/-- The number of days a football team missed practice due to rain -/
def days_missed (daily_practice_hours : ℕ) (total_practice_hours : ℕ) (days_in_week : ℕ) : ℕ :=
  days_in_week - (total_practice_hours / daily_practice_hours)

/-- Theorem: The football team missed 1 day of practice due to rain -/
theorem football_practice_missed_days :
  days_missed 6 36 7 = 1 := by
  sorry

end football_practice_missed_days_l3450_345008


namespace not_perfect_square_l3450_345012

theorem not_perfect_square (n : ℕ) : ¬∃ (m : ℕ), n^6 + 3*n^5 - 5*n^4 - 15*n^3 + 4*n^2 + 12*n + 3 = m^2 := by
  sorry

end not_perfect_square_l3450_345012


namespace intersection_of_A_and_B_l3450_345036

def A : Set ℝ := {x | x + 1 > 0}
def B : Set ℝ := {x | x^2 - x < 0}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end intersection_of_A_and_B_l3450_345036


namespace unique_valid_sequence_l3450_345044

/-- Represents a sequence of 5 missile numbers. -/
def MissileSequence := Fin 5 → Nat

/-- The total number of missiles. -/
def totalMissiles : Nat := 50

/-- Checks if a sequence is valid according to the problem conditions. -/
def isValidSequence (seq : MissileSequence) : Prop :=
  ∀ i j : Fin 5, i < j →
    (seq i < seq j) ∧
    (seq j ≤ totalMissiles) ∧
    (∃ k : Nat, seq j - seq i = k * (j - i))

/-- The specific sequence given in the correct answer. -/
def correctSequence : MissileSequence :=
  fun i => [3, 13, 23, 33, 43].get i

/-- Theorem stating that the correct sequence is the only valid sequence. -/
theorem unique_valid_sequence :
  (isValidSequence correctSequence) ∧
  (∀ seq : MissileSequence, isValidSequence seq → seq = correctSequence) := by
  sorry


end unique_valid_sequence_l3450_345044


namespace intersection_nonempty_l3450_345091

theorem intersection_nonempty (a : ℕ) (ha : a ≥ 2) :
  ∃ b : ℕ, 1 ≤ b ∧ b ≤ a ∧
  (∃ y : ℕ, (∃ x : ℕ, y = a^x) ∧ (∃ x : ℕ, y = (a+1)^x + b)) := by
  sorry

end intersection_nonempty_l3450_345091


namespace gaussian_guardians_score_l3450_345058

/-- The total points scored by the Gaussian Guardians basketball team -/
def total_points (daniel curtis sid emily kalyn hyojeong ty winston : ℕ) : ℕ :=
  daniel + curtis + sid + emily + kalyn + hyojeong + ty + winston

/-- Theorem stating that the total points scored by the Gaussian Guardians is 54 -/
theorem gaussian_guardians_score :
  total_points 7 8 2 11 6 12 1 7 = 54 := by
  sorry

end gaussian_guardians_score_l3450_345058


namespace correct_number_misread_l3450_345075

theorem correct_number_misread (n : ℕ) (initial_avg correct_avg wrong_num : ℚ) : 
  n = 10 → 
  initial_avg = 15 → 
  correct_avg = 16 → 
  wrong_num = 26 → 
  ∃ (correct_num : ℚ), 
    (n : ℚ) * initial_avg - wrong_num + correct_num = (n : ℚ) * correct_avg ∧ 
    correct_num = 36 :=
by sorry

end correct_number_misread_l3450_345075


namespace triangle_semiperimeter_inequality_l3450_345070

/-- 
For any triangle with semiperimeter p, incircle radius r, and circumcircle radius R,
the inequality p ≥ (3/2) * sqrt(6 * R * r) holds.
-/
theorem triangle_semiperimeter_inequality (p r R : ℝ) 
  (hp : p > 0) (hr : r > 0) (hR : R > 0) : p ≥ (3/2) * Real.sqrt (6 * R * r) := by
  sorry

end triangle_semiperimeter_inequality_l3450_345070


namespace max_sequence_length_l3450_345092

theorem max_sequence_length (x : ℕ → ℕ) (n : ℕ) : 
  (∀ k, k < n - 1 → x k < x (k + 1)) →
  (∀ k, k ≤ n - 2 → x k ∣ x (k + 2)) →
  x n = 1000 →
  n ≤ 13 :=
sorry

end max_sequence_length_l3450_345092


namespace max_value_theorem_l3450_345048

theorem max_value_theorem (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
  ∃ (M : ℝ), M = 3 ∧ ∀ (x y z : ℝ), x^2 + y^2 + z^2 = 1 → 3*x*y - 3*y*z + 2*z^2 ≤ M :=
by sorry

end max_value_theorem_l3450_345048


namespace afternoon_sales_l3450_345043

/-- Represents the sales of pears by a salesman in a day -/
structure PearSales where
  morning : ℕ
  afternoon : ℕ
  total : ℕ

/-- Theorem stating the afternoon sales given the conditions -/
theorem afternoon_sales (sales : PearSales) 
  (h1 : sales.afternoon = 2 * sales.morning)
  (h2 : sales.total = sales.morning + sales.afternoon)
  (h3 : sales.total = 420) :
  sales.afternoon = 280 := by
  sorry

#check afternoon_sales

end afternoon_sales_l3450_345043


namespace complex_fraction_equality_l3450_345035

theorem complex_fraction_equality (a b : ℝ) : 
  (Complex.I + 1) / (Complex.I - 1) = Complex.mk a b → b = 1 := by
  sorry

end complex_fraction_equality_l3450_345035


namespace selfie_difference_l3450_345028

theorem selfie_difference (a b c : ℕ) (h1 : a + b + c = 2430) (h2 : 10 * b = 17 * a) (h3 : 10 * c = 23 * a) : c - a = 637 := by
  sorry

end selfie_difference_l3450_345028


namespace theorem_1_theorem_2_l3450_345031

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Theorem 1: If a is parallel to α and b is perpendicular to α, then a is perpendicular to b
theorem theorem_1 (a b : Line) (α : Plane) :
  parallel a α → perpendicular b α → perpendicular_lines a b :=
by sorry

-- Theorem 2: If a is perpendicular to α and a is parallel to β, then α is perpendicular to β
theorem theorem_2 (a : Line) (α β : Plane) :
  perpendicular a α → parallel a β → perpendicular_planes α β :=
by sorry

end theorem_1_theorem_2_l3450_345031


namespace smallest_perfect_square_divisible_by_2_and_3_l3450_345083

theorem smallest_perfect_square_divisible_by_2_and_3 :
  ∃ n : ℕ, n > 0 ∧ (∃ m : ℕ, n = m^2) ∧ 2 ∣ n ∧ 3 ∣ n ∧
  ∀ k : ℕ, k > 0 → (∃ l : ℕ, k = l^2) → 2 ∣ k → 3 ∣ k → n ≤ k :=
by
  -- The proof goes here
  sorry

end smallest_perfect_square_divisible_by_2_and_3_l3450_345083


namespace correct_calculation_l3450_345045

theorem correct_calculation (a b : ℝ) : 3 * a * b + 2 * a * b = 5 * a * b := by
  sorry

end correct_calculation_l3450_345045


namespace max_k_value_l3450_345029

open Real

theorem max_k_value (f : ℝ → ℝ) (k : ℤ) : 
  (∀ x > 2, f x = x + x * log x) →
  (∀ x > 2, ↑k * (x - 2) < f x) →
  k ≤ 4 ∧ ∃ x > 2, 4 * (x - 2) < f x :=
by sorry

end max_k_value_l3450_345029


namespace alpha_value_l3450_345059

theorem alpha_value (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan (α - β) = 1/2)
  (h4 : Real.tan β = 1/3) : 
  α = π/4 := by
  sorry

end alpha_value_l3450_345059


namespace not_p_sufficient_not_necessary_for_not_q_l3450_345097

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x : ℝ) : Prop := 5*x - 6 > x^2

-- Define the relationship between ¬p and ¬q
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x, ¬(p x) → ¬(q x)) ∧ 
  (∃ x, ¬(p x) ∧ q x) := by
sorry

end not_p_sufficient_not_necessary_for_not_q_l3450_345097


namespace cube_root_125_fourth_root_256_square_root_16_l3450_345003

theorem cube_root_125_fourth_root_256_square_root_16 : 
  (125 : ℝ) ^ (1/3) * (256 : ℝ) ^ (1/4) * (16 : ℝ) ^ (1/2) = 80 := by
  sorry

end cube_root_125_fourth_root_256_square_root_16_l3450_345003


namespace arithmetic_expression_equality_l3450_345014

theorem arithmetic_expression_equality : 2 - 3*(-4) - 7 + 2*(-5) - 9 + 6*(-2) = -24 := by
  sorry

end arithmetic_expression_equality_l3450_345014


namespace odd_function_symmetry_l3450_345004

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of being an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the property of being monotonically decreasing on an interval
def is_monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y ≤ f x

-- State the theorem
theorem odd_function_symmetry (hf_odd : is_odd f) 
  (hf_decreasing : is_monotone_decreasing_on f 1 2) :
  is_monotone_decreasing_on f (-2) (-1) ∧ 
  (∀ x ∈ Set.Icc (-2) (-1), f x ≤ -f 2) ∧
  f (-2) = -f 2 :=
sorry

end odd_function_symmetry_l3450_345004


namespace max_x_minus_y_is_half_l3450_345006

theorem max_x_minus_y_is_half :
  ∀ x y : ℝ, 2 * (x^2 + y^2 - x*y) = x + y →
  ∀ z : ℝ, z = x - y → z ≤ (1/2 : ℝ) ∧ ∃ x₀ y₀ : ℝ, 2 * (x₀^2 + y₀^2 - x₀*y₀) = x₀ + y₀ ∧ x₀ - y₀ = (1/2 : ℝ) :=
by sorry

end max_x_minus_y_is_half_l3450_345006


namespace linear_equation_condition_l3450_345007

theorem linear_equation_condition (a : ℝ) : 
  (∀ x, ∃ k m, (a - 1) * x^(|a|) + 4 = k * x + m) → a = -1 := by
  sorry

end linear_equation_condition_l3450_345007


namespace paul_crayons_l3450_345076

/-- The number of crayons Paul had initially -/
def initial_crayons : ℕ := 253

/-- The number of crayons Paul lost or gave away -/
def lost_crayons : ℕ := 70

/-- The number of crayons Paul had left -/
def remaining_crayons : ℕ := initial_crayons - lost_crayons

theorem paul_crayons : remaining_crayons = 183 := by sorry

end paul_crayons_l3450_345076


namespace range_of_a_l3450_345027

/-- The equation |x^2 - a| - x + 2 = 0 has two distinct real roots -/
def has_two_distinct_roots (a : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ |x^2 - a| - x + 2 = 0 ∧ |y^2 - a| - y + 2 = 0

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : has_two_distinct_roots a) : a > 4 := by
  sorry

end range_of_a_l3450_345027


namespace supermarket_spend_correct_l3450_345024

def supermarket_spend (initial_amount left_amount showroom_spend : ℕ) : ℕ :=
  initial_amount - left_amount - showroom_spend

theorem supermarket_spend_correct (initial_amount left_amount showroom_spend : ℕ) 
  (h1 : initial_amount ≥ left_amount + showroom_spend) :
  supermarket_spend initial_amount left_amount showroom_spend = 
    initial_amount - left_amount - showroom_spend :=
by
  sorry

#eval supermarket_spend 106 26 49

end supermarket_spend_correct_l3450_345024


namespace specific_polyhedron_space_diagonals_l3450_345080

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- The number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  (Q.vertices.choose 2) - Q.edges - 2 * Q.quadrilateral_faces

/-- Theorem: A convex polyhedron Q with 30 vertices, 72 edges, 44 faces
    (of which 30 are triangular and 14 are quadrilateral) has 335 space diagonals -/
theorem specific_polyhedron_space_diagonals :
  let Q : ConvexPolyhedron := {
    vertices := 30,
    edges := 72,
    faces := 44,
    triangular_faces := 30,
    quadrilateral_faces := 14
  }
  space_diagonals Q = 335 := by sorry

end specific_polyhedron_space_diagonals_l3450_345080


namespace total_sodas_sold_restaurant_soda_sales_l3450_345010

/-- Theorem: Total sodas sold given diet soda count and ratio of regular to diet --/
theorem total_sodas_sold (diet_count : ℕ) (regular_ratio diet_ratio : ℕ) : ℕ :=
  let regular_count := (regular_ratio * diet_count) / diet_ratio
  diet_count + regular_count

/-- Proof of the specific problem --/
theorem restaurant_soda_sales : total_sodas_sold 28 9 7 = 64 := by
  sorry

end total_sodas_sold_restaurant_soda_sales_l3450_345010


namespace problem_statement_l3450_345077

/-- Given m > 0, p, and q as defined, prove the conditions for m and x. -/
theorem problem_statement (m : ℝ) (h_m : m > 0) : 
  -- Define p
  let p := fun x : ℝ => (x + 1) * (x - 5) ≤ 0
  -- Define q
  let q := fun x : ℝ => 1 - m ≤ x ∧ x ≤ 1 + m
  -- Part 1: When p is a sufficient condition for q, m ≥ 4
  ((∀ x : ℝ, p x → q x) → m ≥ 4) ∧
  -- Part 2: When m = 5 and (p or q) is true but (p and q) is false, 
  --         x is in the specified range
  (m = 5 → 
    ∀ x : ℝ, ((p x ∨ q x) ∧ ¬(p x ∧ q x)) → 
      ((-4 ≤ x ∧ x < -1) ∨ (5 < x ∧ x < 6))) := by
  sorry

end problem_statement_l3450_345077


namespace trip_time_calculation_l3450_345026

/-- Proves that if a trip takes 4.5 hours at 70 mph, it will take 5.25 hours at 60 mph -/
theorem trip_time_calculation (distance : ℝ) : 
  distance = 70 * 4.5 → distance = 60 * 5.25 := by
  sorry

end trip_time_calculation_l3450_345026


namespace square_root_squared_l3450_345055

theorem square_root_squared : (Real.sqrt 930249)^2 = 930249 := by
  sorry

end square_root_squared_l3450_345055


namespace union_equals_real_l3450_345063

open Set Real

def A : Set ℝ := {x : ℝ | x^2 + x - 6 > 0}
def B : Set ℝ := {x : ℝ | -π < x ∧ x < Real.exp 1}

theorem union_equals_real : A ∪ B = univ := by
  sorry

end union_equals_real_l3450_345063


namespace art_supplies_problem_l3450_345047

/-- Cost of one box of brushes in yuan -/
def brush_cost : ℕ := 17

/-- Cost of one canvas in yuan -/
def canvas_cost : ℕ := 15

/-- Total number of items to purchase -/
def total_items : ℕ := 10

/-- Maximum total cost in yuan -/
def max_total_cost : ℕ := 157

/-- Cost of 2 boxes of brushes and 4 canvases in yuan -/
def cost_2b_4c : ℕ := 94

/-- Cost of 4 boxes of brushes and 2 canvases in yuan -/
def cost_4b_2c : ℕ := 98

theorem art_supplies_problem :
  (2 * brush_cost + 4 * canvas_cost = cost_2b_4c) ∧
  (4 * brush_cost + 2 * canvas_cost = cost_4b_2c) ∧
  (∀ m : ℕ, m ≥ 7 → brush_cost * (total_items - m) + canvas_cost * m ≤ max_total_cost) ∧
  (brush_cost * 2 + canvas_cost * 8 < brush_cost * 3 + canvas_cost * 7) := by
  sorry

end art_supplies_problem_l3450_345047


namespace cloth_sale_calculation_l3450_345071

theorem cloth_sale_calculation (total_selling_price : ℝ) (profit_per_meter : ℝ) (cost_price_per_meter : ℝ)
  (h1 : total_selling_price = 9890)
  (h2 : profit_per_meter = 24)
  (h3 : cost_price_per_meter = 83.5) :
  (total_selling_price / (cost_price_per_meter + profit_per_meter)) = 92 := by
  sorry

end cloth_sale_calculation_l3450_345071


namespace rainfall_problem_l3450_345005

theorem rainfall_problem (total_rainfall : ℝ) (ratio : ℝ) :
  total_rainfall = 30 →
  ratio = 1.5 →
  ∃ (first_week : ℝ) (second_week : ℝ),
    first_week + second_week = total_rainfall ∧
    second_week = ratio * first_week ∧
    second_week = 18 := by
  sorry

end rainfall_problem_l3450_345005


namespace pages_per_day_l3450_345019

/-- Given a book with 240 pages read over 12 days with equal pages per day, prove that 20 pages are read daily. -/
theorem pages_per_day (total_pages : ℕ) (days : ℕ) (pages_per_day : ℕ) : 
  total_pages = 240 → days = 12 → total_pages = days * pages_per_day → pages_per_day = 20 := by
  sorry

end pages_per_day_l3450_345019


namespace jane_apple_purchase_l3450_345041

/-- The price of one apple in dollars -/
def apple_price : ℝ := 2

/-- The amount Jane has to spend in dollars -/
def jane_budget : ℝ := 2

/-- There is no bulk discount -/
axiom no_bulk_discount : ∀ (n : ℕ), n * apple_price = jane_budget → n = 1

/-- The number of apples Jane can buy with her budget -/
def apples_bought : ℕ := 1

theorem jane_apple_purchase :
  apples_bought * apple_price = jane_budget :=
sorry

end jane_apple_purchase_l3450_345041


namespace jose_wandering_time_l3450_345037

/-- Proves that Jose's wandering time is 10 hours given his distance and speed -/
theorem jose_wandering_time : 
  ∀ (distance : ℝ) (speed : ℝ),
  distance = 15 →
  speed = 1.5 →
  distance / speed = 10 :=
by
  sorry

end jose_wandering_time_l3450_345037


namespace polynomial_coefficient_B_l3450_345030

theorem polynomial_coefficient_B (A C D : ℤ) : 
  ∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℕ+), 
    (r₁ * r₂ * r₃ * r₄ * r₅ * r₆ : ℤ) = 64 ∧ 
    (r₁ + r₂ + r₃ + r₄ + r₅ + r₆ : ℤ) = 15 ∧ 
    ∀ (z : ℂ), z^6 - 15*z^5 + A*z^4 + (-244)*z^3 + C*z^2 + D*z + 64 = 
      (z - r₁) * (z - r₂) * (z - r₃) * (z - r₄) * (z - r₅) * (z - r₆) :=
by sorry

end polynomial_coefficient_B_l3450_345030


namespace function_properties_l3450_345069

theorem function_properties :
  (∃ x : ℝ, (10 : ℝ) ^ x = x) ∧
  (∃ x : ℝ, (10 : ℝ) ^ x = x ^ 2) ∧
  (¬ ∀ x : ℝ, (10 : ℝ) ^ x > x) ∧
  (¬ ∀ x : ℝ, x > 0 → (10 : ℝ) ^ x > x ^ 2) ∧
  (¬ ∃ x y : ℝ, x ≠ y ∧ (10 : ℝ) ^ x = -x ∧ (10 : ℝ) ^ y = -y) := by
  sorry

end function_properties_l3450_345069


namespace gary_earnings_l3450_345072

def total_flour : ℚ := 6
def flour_for_cakes : ℚ := 4
def flour_per_cake : ℚ := 1/2
def flour_for_cupcakes : ℚ := 2
def flour_per_cupcake : ℚ := 1/5
def price_per_cake : ℚ := 5/2
def price_per_cupcake : ℚ := 1

def num_cakes : ℚ := flour_for_cakes / flour_per_cake
def num_cupcakes : ℚ := flour_for_cupcakes / flour_per_cupcake

def earnings_from_cakes : ℚ := num_cakes * price_per_cake
def earnings_from_cupcakes : ℚ := num_cupcakes * price_per_cupcake

theorem gary_earnings :
  earnings_from_cakes + earnings_from_cupcakes = 30 :=
by sorry

end gary_earnings_l3450_345072


namespace sum_of_tens_and_units_digits_l3450_345015

def repeating_707 : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707
def repeating_909 : ℕ := 909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909

def product : ℕ := repeating_707 * repeating_909

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def units_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_tens_and_units_digits :
  tens_digit product + units_digit product = 9 := by
  sorry

end sum_of_tens_and_units_digits_l3450_345015


namespace ball_hit_ground_time_l3450_345025

/-- The time when a ball hits the ground, given its height equation -/
theorem ball_hit_ground_time (t : ℝ) : t ≥ 0 → -8*t^2 - 12*t + 72 = 0 → t = 3 := by
  sorry

#check ball_hit_ground_time

end ball_hit_ground_time_l3450_345025


namespace valid_f_forms_l3450_345062

-- Define the function g
def g (x : ℝ) : ℝ := -x^2 - 3

-- Define the properties of function f
def is_valid_f (f : ℝ → ℝ) : Prop :=
  -- f is a quadratic function
  ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c ∧ a ≠ 0 ∧
  -- The minimum value of f(x) on [-1,2] is 1
  (∀ x ∈ Set.Icc (-1) 2, f x ≥ 1) ∧
  (∃ x ∈ Set.Icc (-1) 2, f x = 1) ∧
  -- f(x) + g(x) is an odd function
  ∀ x, f (-x) + g (-x) = -(f x + g x)

-- Theorem statement
theorem valid_f_forms :
  ∀ f : ℝ → ℝ, is_valid_f f →
    (∀ x, f x = x^2 - 2 * Real.sqrt 2 * x + 3) ∨
    (∀ x, f x = x^2 + 3 * x + 3) :=
sorry

end valid_f_forms_l3450_345062


namespace second_fold_perpendicular_l3450_345096

/-- Represents a point on a plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a line on a plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Represents a sheet of paper with one straight edge -/
structure Paper :=
  (straight_edge : Line)

/-- Represents a fold on the paper -/
structure Fold :=
  (line : Line)
  (paper : Paper)

/-- Checks if a point is on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Checks if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Theorem: The second fold creates a line perpendicular to the initial crease -/
theorem second_fold_perpendicular 
  (paper : Paper) 
  (initial_fold : Fold)
  (A : Point)
  (second_fold : Fold)
  (h1 : point_on_line A paper.straight_edge)
  (h2 : point_on_line A initial_fold.line)
  (h3 : point_on_line A second_fold.line)
  (h4 : ∃ (p q : Point), 
    point_on_line p paper.straight_edge ∧ 
    point_on_line q paper.straight_edge ∧
    point_on_line p second_fold.line ∧
    point_on_line q initial_fold.line) :
  perpendicular initial_fold.line second_fold.line :=
sorry

end second_fold_perpendicular_l3450_345096


namespace factor_expression_l3450_345001

theorem factor_expression (x : ℝ) : x * (x - 3) - 5 * (x - 3) = (x - 5) * (x - 3) := by
  sorry

end factor_expression_l3450_345001


namespace student_comprehensive_score_l3450_345052

/-- Represents the scores and weights for a science and technology innovation competition. -/
structure CompetitionScores where
  theoretical_knowledge : ℝ
  innovative_design : ℝ
  on_site_presentation : ℝ
  theoretical_weight : ℝ
  innovative_weight : ℝ
  on_site_weight : ℝ

/-- Calculates the comprehensive score for a given set of competition scores. -/
def comprehensive_score (scores : CompetitionScores) : ℝ :=
  scores.theoretical_knowledge * scores.theoretical_weight +
  scores.innovative_design * scores.innovative_weight +
  scores.on_site_presentation * scores.on_site_weight

/-- Theorem stating that the student's comprehensive score is 90 points. -/
theorem student_comprehensive_score :
  let scores : CompetitionScores := {
    theoretical_knowledge := 95,
    innovative_design := 88,
    on_site_presentation := 90,
    theoretical_weight := 0.2,
    innovative_weight := 0.5,
    on_site_weight := 0.3
  }
  comprehensive_score scores = 90 := by
  sorry


end student_comprehensive_score_l3450_345052


namespace unique_polynomial_satisfying_conditions_l3450_345067

/-- A polynomial function of degree at most 3 -/
def PolynomialDegree3 (g : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, g x = a * x^3 + b * x^2 + c * x + d

/-- The conditions that g must satisfy -/
def SatisfiesConditions (g : ℝ → ℝ) : Prop :=
  (∀ x, g (x^2) = (g x)^2) ∧ 
  (∀ x, g (x^2) = g (g x)) ∧ 
  g 1 = 1

theorem unique_polynomial_satisfying_conditions :
  ∃! g : ℝ → ℝ, PolynomialDegree3 g ∧ SatisfiesConditions g ∧ (∀ x, g x = x^2) := by
  sorry

end unique_polynomial_satisfying_conditions_l3450_345067


namespace product_inspection_probabilities_l3450_345017

def total_units : ℕ := 6
def inspected_units : ℕ := 2
def first_grade_units : ℕ := 3
def second_grade_units : ℕ := 2
def defective_units : ℕ := 1

def probability_both_first_grade : ℚ := 1 / 5
def probability_one_second_grade : ℚ := 8 / 15

def probability_at_most_one_defective (x : ℕ) : ℚ :=
  (Nat.choose x 1 * Nat.choose (total_units - x) 1 + Nat.choose (total_units - x) 2) /
  Nat.choose total_units inspected_units

theorem product_inspection_probabilities :
  (probability_both_first_grade = 1 / 5) ∧
  (probability_one_second_grade = 8 / 15) ∧
  (∀ x : ℕ, x ≤ total_units →
    (probability_at_most_one_defective x ≥ 4 / 5 → x ≤ 3)) :=
by sorry

end product_inspection_probabilities_l3450_345017


namespace smallest_ccd_is_227_l3450_345084

/-- Represents a two-digit number -/
def TwoDigitNumber (c d : ℕ) : Prop :=
  c ≠ 0 ∧ c ≤ 9 ∧ d ≤ 9

/-- Represents a three-digit number -/
def ThreeDigitNumber (c d : ℕ) : Prop :=
  TwoDigitNumber c d ∧ c * 100 + c * 10 + d ≥ 100

/-- The main theorem -/
theorem smallest_ccd_is_227 :
  ∃ (c d : ℕ),
    TwoDigitNumber c d ∧
    ThreeDigitNumber c d ∧
    c ≠ d ∧
    (c * 10 + d : ℚ) = (1 / 7) * (c * 100 + c * 10 + d) ∧
    c * 100 + c * 10 + d = 227 ∧
    ∀ (c' d' : ℕ),
      TwoDigitNumber c' d' →
      ThreeDigitNumber c' d' →
      c' ≠ d' →
      (c' * 10 + d' : ℚ) = (1 / 7) * (c' * 100 + c' * 10 + d') →
      c' * 100 + c' * 10 + d' ≥ 227 :=
by sorry

end smallest_ccd_is_227_l3450_345084


namespace zoo_feeding_theorem_l3450_345086

/-- Represents the number of animal pairs in the zoo -/
def num_pairs : ℕ := 6

/-- Represents the number of ways to feed the animals -/
def feeding_ways : ℕ := 14400

/-- Theorem stating the number of ways to feed the animals in the specified pattern -/
theorem zoo_feeding_theorem :
  (num_pairs = 6) →
  (∃ (male_choices female_choices : ℕ → ℕ),
    (∀ i, i ∈ Finset.range (num_pairs - 1) → male_choices i = num_pairs - 1 - i) ∧
    (∀ i, i ∈ Finset.range num_pairs → female_choices i = num_pairs - 1 - i) ∧
    (feeding_ways = (Finset.prod (Finset.range (num_pairs - 1)) male_choices) *
                    (Finset.prod (Finset.range num_pairs) female_choices))) :=
by sorry

#check zoo_feeding_theorem

end zoo_feeding_theorem_l3450_345086


namespace exponent_multiplication_l3450_345042

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end exponent_multiplication_l3450_345042


namespace unique_number_l3450_345087

theorem unique_number : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  (n + 3) % 3 = 0 ∧ 
  (n + 4) % 4 = 0 ∧ 
  (n + 5) % 5 = 0 ∧ 
  n = 60 := by sorry

end unique_number_l3450_345087


namespace total_peaches_is_450_l3450_345082

/-- Represents the number of baskets in the fruit shop -/
def num_baskets : ℕ := 15

/-- Represents the initial number of red peaches in each basket -/
def initial_red : ℕ := 19

/-- Represents the initial number of green peaches in each basket -/
def initial_green : ℕ := 4

/-- Represents the number of moldy peaches in each basket -/
def moldy : ℕ := 6

/-- Represents the number of red peaches removed from each basket -/
def removed_red : ℕ := 3

/-- Represents the number of green peaches removed from each basket -/
def removed_green : ℕ := 1

/-- Represents the number of freshly harvested peaches added to each basket -/
def added_fresh : ℕ := 5

/-- Calculates the total number of peaches in all baskets after adjustments -/
def total_peaches_after_adjustment : ℕ :=
  num_baskets * ((initial_red - removed_red) + (initial_green - removed_green) + moldy + added_fresh)

/-- Theorem stating that the total number of peaches after adjustments is 450 -/
theorem total_peaches_is_450 : total_peaches_after_adjustment = 450 := by
  sorry

end total_peaches_is_450_l3450_345082


namespace systematic_sampling_fourth_student_l3450_345051

/-- Represents a systematic sampling of students. -/
structure SystematicSample where
  totalStudents : ℕ
  sampleSize : ℕ
  sampleInterval : ℕ
  firstStudent : ℕ

/-- Checks if a student number is in the sample. -/
def isInSample (s : SystematicSample) (studentNumber : ℕ) : Prop :=
  ∃ k : ℕ, studentNumber = s.firstStudent + k * s.sampleInterval ∧ 
           studentNumber ≤ s.totalStudents

theorem systematic_sampling_fourth_student 
  (s : SystematicSample)
  (h1 : s.totalStudents = 60)
  (h2 : s.sampleSize = 4)
  (h3 : s.firstStudent = 3)
  (h4 : isInSample s 33)
  (h5 : isInSample s 48) :
  isInSample s 18 := by
  sorry

#check systematic_sampling_fourth_student

end systematic_sampling_fourth_student_l3450_345051


namespace function_inequality_l3450_345032

open Real

-- Define the function F
noncomputable def F (f : ℝ → ℝ) (x : ℝ) : ℝ := f x / (Real.exp x)

-- State the theorem
theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (hf' : Differentiable ℝ (deriv f))
  (h : ∀ x, deriv (deriv f) x < f x) :
  f 2 < Real.exp 2 * f 0 ∧ f 2017 < Real.exp 2017 * f 0 := by
  sorry

end function_inequality_l3450_345032


namespace sin_cos_cube_sum_l3450_345093

theorem sin_cos_cube_sum (θ : ℝ) (h : 4 * Real.sin θ * Real.cos θ - 5 * Real.sin θ - 5 * Real.cos θ - 1 = 0) :
  Real.sin θ ^ 3 + Real.cos θ ^ 3 = -11/16 := by
  sorry

end sin_cos_cube_sum_l3450_345093


namespace olympiad_sheet_distribution_l3450_345002

theorem olympiad_sheet_distribution (n : ℕ) :
  let initial_total := 2 + 3 + 1 + 1
  let final_total := initial_total + 2 * n
  ¬ ∃ (k : ℕ), final_total = 4 * k := by
  sorry

end olympiad_sheet_distribution_l3450_345002


namespace quadratic_equation_solution_l3450_345095

theorem quadratic_equation_solution : 
  let f : ℝ → ℝ := λ x => 2*x^2 - 3*x - (1 - 2*x)
  (f 1 = 0) ∧ (f (-1/2) = 0) ∧ 
  (∀ x : ℝ, f x = 0 → x = 1 ∨ x = -1/2) := by
sorry

end quadratic_equation_solution_l3450_345095


namespace ice_cream_flavors_count_l3450_345018

/-- The number of ways to distribute n indistinguishable items into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of flavors that can be created by combining 4 scoops of 3 basic flavors -/
def ice_cream_flavors : ℕ := distribute 4 3

theorem ice_cream_flavors_count : ice_cream_flavors = 15 := by sorry

end ice_cream_flavors_count_l3450_345018


namespace distance_between_z₁_and_z₂_l3450_345061

noncomputable def z₁ : ℂ := (Complex.I * 2 + 1)⁻¹ * (Complex.I * 3 - 1)

noncomputable def z₂ : ℂ := 1 + (1 + Complex.I)^10

theorem distance_between_z₁_and_z₂ : 
  Complex.abs (z₂ - z₁) = Real.sqrt 231.68 := by sorry

end distance_between_z₁_and_z₂_l3450_345061
