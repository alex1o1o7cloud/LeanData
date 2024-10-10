import Mathlib

namespace salary_before_raise_l2736_273603

theorem salary_before_raise (new_salary : ℝ) (increase_percentage : ℝ) 
  (h1 : new_salary = 70)
  (h2 : increase_percentage = 0.40) :
  let original_salary := new_salary / (1 + increase_percentage)
  original_salary = 50 := by
sorry

end salary_before_raise_l2736_273603


namespace girls_boys_difference_l2736_273631

theorem girls_boys_difference (girls boys : ℝ) (h1 : girls = 542.0) (h2 : boys = 387.0) :
  girls - boys = 155.0 := by
  sorry

end girls_boys_difference_l2736_273631


namespace prob_neither_red_nor_green_is_one_third_l2736_273690

-- Define the number of pens of each color
def green_pens : ℕ := 5
def black_pens : ℕ := 6
def red_pens : ℕ := 7

-- Define the total number of pens
def total_pens : ℕ := green_pens + black_pens + red_pens

-- Define the probability of picking a pen that is neither red nor green
def prob_neither_red_nor_green : ℚ := black_pens / total_pens

-- Theorem statement
theorem prob_neither_red_nor_green_is_one_third :
  prob_neither_red_nor_green = 1 / 3 := by
  sorry

end prob_neither_red_nor_green_is_one_third_l2736_273690


namespace savings_to_earnings_ratio_l2736_273673

/-- Proves that the ratio of monthly savings to total monthly earnings is 1/2 --/
theorem savings_to_earnings_ratio 
  (car_washing_earnings : ℕ) 
  (dog_walking_earnings : ℕ) 
  (months_to_save : ℕ) 
  (total_savings : ℕ) 
  (h1 : car_washing_earnings = 20)
  (h2 : dog_walking_earnings = 40)
  (h3 : months_to_save = 5)
  (h4 : total_savings = 150) :
  (total_savings / months_to_save) / (car_washing_earnings + dog_walking_earnings) = 1 / 2 :=
by
  sorry


end savings_to_earnings_ratio_l2736_273673


namespace ada_original_seat_l2736_273636

-- Define the type for seats
inductive Seat : Type
  | one : Seat
  | two : Seat
  | three : Seat
  | four : Seat
  | five : Seat

-- Define the type for friends
inductive Friend : Type
  | ada : Friend
  | bea : Friend
  | ceci : Friend
  | dee : Friend
  | edie : Friend

-- Define the seating arrangement as a function from Friend to Seat
def SeatingArrangement : Type := Friend → Seat

-- Define what it means for a seat to be an end seat
def isEndSeat (s : Seat) : Prop :=
  s = Seat.one ∨ s = Seat.five

-- Define the movement of friends
def moveRight (s : Seat) (n : Nat) : Seat :=
  match s, n with
  | Seat.one, 1 => Seat.two
  | Seat.one, 2 => Seat.three
  | Seat.two, 1 => Seat.three
  | Seat.two, 2 => Seat.four
  | Seat.three, 1 => Seat.four
  | Seat.three, 2 => Seat.five
  | Seat.four, 1 => Seat.five
  | _, _ => s  -- Default case: no movement or invalid movement

def moveLeft (s : Seat) (n : Nat) : Seat :=
  match s, n with
  | Seat.two, 1 => Seat.one
  | Seat.three, 1 => Seat.two
  | Seat.four, 1 => Seat.three
  | Seat.five, 1 => Seat.four
  | _, _ => s  -- Default case: no movement or invalid movement

-- Theorem statement
theorem ada_original_seat (initial final : SeatingArrangement) :
  (∀ f : Friend, f ≠ Friend.ada → initial f ≠ Seat.five) →  -- No one except possibly Ada starts in seat 5
  (initial Friend.bea = moveLeft (final Friend.bea) 2) →   -- Bea moved 2 seats right
  (initial Friend.ceci = moveRight (final Friend.ceci) 1) → -- Ceci moved 1 seat left
  (initial Friend.dee = final Friend.edie ∧ initial Friend.edie = final Friend.dee) → -- Dee and Edie switched
  (isEndSeat (final Friend.ada)) →  -- Ada ends up in an end seat
  (initial Friend.ada = Seat.two) :=  -- Prove Ada started in seat 2
sorry

end ada_original_seat_l2736_273636


namespace product_zero_from_sum_conditions_l2736_273620

theorem product_zero_from_sum_conditions (x y z w : ℝ) 
  (sum_condition : x + y + z + w = 0)
  (power_sum_condition : x^7 + y^7 + z^7 + w^7 = 0) :
  w * (w + x) * (w + y) * (w + z) = 0 := by
  sorry

end product_zero_from_sum_conditions_l2736_273620


namespace roger_tray_capacity_l2736_273680

/-- The number of trays Roger can carry at a time -/
def trays_per_trip : ℕ := sorry

/-- The number of trips Roger made -/
def num_trips : ℕ := 3

/-- The number of trays Roger picked up from the first table -/
def trays_table1 : ℕ := 10

/-- The number of trays Roger picked up from the second table -/
def trays_table2 : ℕ := 2

/-- The total number of trays Roger picked up -/
def total_trays : ℕ := trays_table1 + trays_table2

theorem roger_tray_capacity :
  trays_per_trip * num_trips = total_trays ∧ trays_per_trip = 4 := by
  sorry

end roger_tray_capacity_l2736_273680


namespace square_side_length_is_twenty_l2736_273666

/-- The side length of a square that can contain specific numbers of square tiles of different sizes -/
def square_side_length : ℕ := 
  let one_by_one := 4
  let two_by_two := 8
  let three_by_three := 12
  let four_by_four := 16
  let total_area := one_by_one * 1^2 + two_by_two * 2^2 + three_by_three * 3^2 + four_by_four * 4^2
  Nat.sqrt total_area

/-- Theorem stating that the side length of the square is 20 -/
theorem square_side_length_is_twenty : square_side_length = 20 := by
  sorry

end square_side_length_is_twenty_l2736_273666


namespace race_head_start_l2736_273698

theorem race_head_start (L : ℝ) (Va Vb : ℝ) (h : Va = (51 / 44) * Vb) :
  let H := L * (7 / 51)
  (L / Va) = ((L - H) / Vb) := by
  sorry

end race_head_start_l2736_273698


namespace prob_four_suits_in_five_draws_l2736_273661

/-- Represents a standard deck of 52 cards -/
def StandardDeck : Type := Unit

/-- Represents the number of suits in a standard deck -/
def numSuits : ℕ := 4

/-- Represents the number of cards drawn -/
def numDraws : ℕ := 5

/-- Represents the probability of drawing a card from a particular suit -/
def probSuitDraw : ℚ := 1 / 4

/-- The probability of drawing 4 cards representing each of the 4 suits 
    when drawing 5 cards with replacement from a standard 52-card deck -/
theorem prob_four_suits_in_five_draws (deck : StandardDeck) : 
  (3 : ℚ) / 32 = probSuitDraw^3 * (1 - probSuitDraw) * (2 - probSuitDraw) * (3 - probSuitDraw) / 6 :=
sorry

end prob_four_suits_in_five_draws_l2736_273661


namespace complement_of_A_l2736_273669

-- Define the universal set U
def U : Set ℝ := {x | x < 4}

-- Define set A
def A : Set ℝ := {x | x < 1}

-- State the theorem
theorem complement_of_A : 
  (U \ A) = {x : ℝ | 1 ≤ x ∧ x < 4} := by sorry

end complement_of_A_l2736_273669


namespace tan_2alpha_values_l2736_273693

theorem tan_2alpha_values (α : ℝ) (h : 2 * Real.sin (2 * α) = 1 + Real.cos (2 * α)) :
  Real.tan (2 * α) = 4/3 ∨ Real.tan (2 * α) = 0 := by
  sorry

end tan_2alpha_values_l2736_273693


namespace stanleys_distance_difference_l2736_273665

theorem stanleys_distance_difference (run_distance walk_distance : ℝ) : 
  run_distance = 0.4 → walk_distance = 0.2 → run_distance - walk_distance = 0.2 := by
  sorry

end stanleys_distance_difference_l2736_273665


namespace average_salary_problem_l2736_273675

/-- The average monthly salary problem -/
theorem average_salary_problem (initial_average : ℚ) (old_supervisor_salary : ℚ) 
  (new_supervisor_salary : ℚ) (num_workers : ℕ) (total_people : ℕ) 
  (h1 : initial_average = 430)
  (h2 : old_supervisor_salary = 870)
  (h3 : new_supervisor_salary = 780)
  (h4 : num_workers = 8)
  (h5 : total_people = num_workers + 1) :
  let total_initial_salary := initial_average * total_people
  let workers_salary := total_initial_salary - old_supervisor_salary
  let new_total_salary := workers_salary + new_supervisor_salary
  let new_average_salary := new_total_salary / total_people
  new_average_salary = 420 := by sorry

end average_salary_problem_l2736_273675


namespace real_y_condition_l2736_273614

theorem real_y_condition (x y : ℝ) : 
  (9 * y^2 - 6 * x * y + 2 * x + 7 = 0) → 
  (∃ (y : ℝ), 9 * y^2 - 6 * x * y + 2 * x + 7 = 0) ↔ (x ≤ -2 ∨ x ≥ 7) :=
by sorry

end real_y_condition_l2736_273614


namespace midpoint_octahedron_volume_ratio_l2736_273681

-- Define a regular tetrahedron
structure RegularTetrahedron where
  -- Add necessary fields here

-- Define an octahedron formed by midpoints of tetrahedron edges
structure MidpointOctahedron (t : RegularTetrahedron) where
  -- Add necessary fields here

-- Define volume calculation functions
def volume_tetrahedron (t : RegularTetrahedron) : ℝ := sorry

def volume_octahedron (o : MidpointOctahedron t) : ℝ := sorry

-- Theorem statement
theorem midpoint_octahedron_volume_ratio 
  (t : RegularTetrahedron) 
  (o : MidpointOctahedron t) : 
  volume_octahedron o / volume_tetrahedron t = 27 / 64 := by
  sorry

end midpoint_octahedron_volume_ratio_l2736_273681


namespace complex_equation_solution_l2736_273626

theorem complex_equation_solution :
  ∀ (z : ℂ), (2 + Complex.I) * z = 5 * Complex.I → z = 1 + 2 * Complex.I :=
by
  sorry

end complex_equation_solution_l2736_273626


namespace frustum_volume_l2736_273664

/-- The volume of a frustum with specific conditions --/
theorem frustum_volume (r₁ r₂ : ℝ) (h : ℝ) : 
  r₁ = Real.sqrt 3 →
  r₂ = 3 * Real.sqrt 3 →
  h = 6 →
  (1/3 : ℝ) * (π * r₁^2 + π * r₂^2 + Real.sqrt (π^2 * r₁^2 * r₂^2)) * h = 78 * π := by
  sorry

#check frustum_volume

end frustum_volume_l2736_273664


namespace model_c_sample_size_l2736_273659

/-- Calculates the number of units to be sampled from a specific model in stratified sampling. -/
def stratified_sample_size (total_units : ℕ) (sample_size : ℕ) (model_units : ℕ) : ℕ :=
  (model_units * sample_size) / total_units

/-- Theorem stating that the stratified sample size for Model C is 10 units. -/
theorem model_c_sample_size :
  let total_units : ℕ := 1400 + 5600 + 2000
  let sample_size : ℕ := 45
  let model_c_units : ℕ := 2000
  stratified_sample_size total_units sample_size model_c_units = 10 := by
  sorry

end model_c_sample_size_l2736_273659


namespace water_volume_ratio_in_cone_l2736_273615

/-- Theorem: Volume ratio of water in a cone filled to 2/3 of its height -/
theorem water_volume_ratio_in_cone (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let water_height : ℝ := 2 / 3 * h
  let water_radius : ℝ := 2 / 3 * r
  let cone_volume : ℝ := (1 / 3) * π * r^2 * h
  let water_volume : ℝ := (1 / 3) * π * water_radius^2 * water_height
  water_volume / cone_volume = 8 / 27 :=
by sorry

end water_volume_ratio_in_cone_l2736_273615


namespace M_lower_bound_l2736_273646

theorem M_lower_bound (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (1/a - 1) * (1/b - 1) * (1/c - 1) ≥ 8 := by
sorry

end M_lower_bound_l2736_273646


namespace parabola_expression_l2736_273604

/-- A parabola that intersects the x-axis at (-1,0) and (2,0) and has the same shape and direction of opening as y = -2x² -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  root1 : a * (-1)^2 + b * (-1) + c = 0
  root2 : a * 2^2 + b * 2 + c = 0
  shape : a = -2

/-- The expression of the parabola is y = -2x² + 2x + 4 -/
theorem parabola_expression (p : Parabola) : p.a = -2 ∧ p.b = 2 ∧ p.c = 4 := by
  sorry

end parabola_expression_l2736_273604


namespace no_divisible_polynomial_values_l2736_273657

theorem no_divisible_polynomial_values : ¬∃ (m n : ℤ), 
  0 < m ∧ m < n ∧ 
  (n ∣ (m^2 + m - 70)) ∧ 
  ((n + 1) ∣ ((m + 1)^2 + (m + 1) - 70)) := by
  sorry

end no_divisible_polynomial_values_l2736_273657


namespace cement_mixture_weight_l2736_273600

theorem cement_mixture_weight (sand_ratio : ℚ) (water_ratio : ℚ) (gravel_weight : ℚ) 
  (h1 : sand_ratio = 1 / 3)
  (h2 : water_ratio = 1 / 2)
  (h3 : gravel_weight = 8) :
  ∃ (total_weight : ℚ), 
    sand_ratio * total_weight + water_ratio * total_weight + gravel_weight = total_weight ∧ 
    total_weight = 48 := by
sorry

end cement_mixture_weight_l2736_273600


namespace base7_to_base10_conversion_l2736_273677

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The given number in base 7 --/
def base7Number : List Nat := [4, 3, 6, 2, 5]

/-- Theorem: The base 10 equivalent of 52634₇ is 13010 --/
theorem base7_to_base10_conversion :
  base7ToBase10 base7Number = 13010 := by
  sorry

end base7_to_base10_conversion_l2736_273677


namespace sum_x_2y_equals_5_l2736_273692

theorem sum_x_2y_equals_5 (x y : ℕ+) 
  (h : x^3 + 3*x^2*y + 8*x*y^2 + 6*y^3 = 87) : 
  x + 2*y = 5 := by
  sorry

end sum_x_2y_equals_5_l2736_273692


namespace regular_polygon_sides_l2736_273607

-- Define the number of diagonals in a regular polygon
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- Theorem statement
theorem regular_polygon_sides :
  ∀ n : ℕ, n ≥ 3 →
  (num_diagonals n + 2 * n = n^2) → n = 3 := by
sorry

end regular_polygon_sides_l2736_273607


namespace family_hard_shell_tacos_l2736_273656

/-- The number of hard shell tacos bought by a family -/
def hard_shell_tacos : ℕ := sorry

/-- The price of a soft taco in dollars -/
def soft_taco_price : ℕ := 2

/-- The price of a hard shell taco in dollars -/
def hard_shell_taco_price : ℕ := 5

/-- The number of soft tacos bought by the family -/
def family_soft_tacos : ℕ := 3

/-- The number of additional customers -/
def additional_customers : ℕ := 10

/-- The number of soft tacos bought by each additional customer -/
def soft_tacos_per_customer : ℕ := 2

/-- The total revenue in dollars -/
def total_revenue : ℕ := 66

theorem family_hard_shell_tacos :
  hard_shell_tacos = 4 :=
by sorry

end family_hard_shell_tacos_l2736_273656


namespace pipe_b_shut_time_l2736_273621

-- Define the rates at which pipes fill the tank
def pipe_a_rate : ℚ := 1
def pipe_b_rate : ℚ := 1 / 15

-- Define the time it takes for the tank to overflow
def overflow_time : ℚ := 1 / 2  -- 30 minutes = 0.5 hours

-- Define the theorem
theorem pipe_b_shut_time :
  let combined_rate := pipe_a_rate + pipe_b_rate
  let volume_filled_together := combined_rate * overflow_time
  let pipe_b_shut_time := 1 - volume_filled_together
  pipe_b_shut_time * 60 = 28 := by
sorry

end pipe_b_shut_time_l2736_273621


namespace f_at_two_l2736_273658

noncomputable section

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- f' is the derivative of f
axiom is_derivative : ∀ x, deriv f x = f' x

-- f(x) = 2xf'(2) + ln(x-1)
axiom f_def : ∀ x, f x = 2 * x * (f' 2) + Real.log (x - 1)

theorem f_at_two : f 2 = -4 := by sorry

end f_at_two_l2736_273658


namespace square_root_of_sixteen_l2736_273644

theorem square_root_of_sixteen : 
  {x : ℝ | x^2 = 16} = {4, -4} := by sorry

end square_root_of_sixteen_l2736_273644


namespace hugo_first_roll_7_given_win_l2736_273663

-- Define the number of players
def num_players : ℕ := 5

-- Define the number of sides on the first die
def first_die_sides : ℕ := 8

-- Define the number of sides on the subsequent die
def subsequent_die_sides : ℕ := 10

-- Define Hugo's winning probability
def hugo_win_prob : ℚ := 1 / num_players

-- Define the probability of rolling a 7 on the first die
def prob_roll_7 : ℚ := 1 / first_die_sides

-- Define the event that Hugo wins given his first roll was 7
def hugo_win_given_7 : ℚ := 961 / 2560

-- Theorem to prove
theorem hugo_first_roll_7_given_win (num_players : ℕ) (first_die_sides : ℕ) 
  (subsequent_die_sides : ℕ) (hugo_win_prob : ℚ) (prob_roll_7 : ℚ) 
  (hugo_win_given_7 : ℚ) :
  num_players = 5 → 
  first_die_sides = 8 → 
  subsequent_die_sides = 10 → 
  hugo_win_prob = 1 / 5 → 
  prob_roll_7 = 1 / 8 → 
  hugo_win_given_7 = 961 / 2560 → 
  (prob_roll_7 * hugo_win_given_7) / hugo_win_prob = 961 / 2048 := by
  sorry


end hugo_first_roll_7_given_win_l2736_273663


namespace find_number_l2736_273624

theorem find_number : ∃! x : ℝ, 7 * x + 37 = 100 ∧ x = 9 := by
  sorry

end find_number_l2736_273624


namespace tea_mixture_price_l2736_273696

theorem tea_mixture_price (price1 price2 price3 mixture_price : ℝ) 
  (h1 : price1 = 126)
  (h2 : price3 = 175.5)
  (h3 : mixture_price = 153)
  (h4 : price1 + price2 + 2 * price3 = 4 * mixture_price) :
  price2 = 135 := by
  sorry

end tea_mixture_price_l2736_273696


namespace set_operations_l2736_273667

def U : Set ℤ := {x | -3 ≤ x ∧ x ≤ 3}
def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {-1, 0, 1}
def C : Set ℤ := {-2, 0, 2}

theorem set_operations :
  (A ∪ (B ∩ C) = {0, 1, 2, 3}) ∧
  (A ∩ (U \ (B ∪ C)) = {3}) := by
  sorry

end set_operations_l2736_273667


namespace expand_and_subtract_fraction_division_l2736_273674

-- Part 1
theorem expand_and_subtract (m n : ℝ) :
  (2*m + 3*n)^2 - (2*m + n)*(2*m - n) = 12*m*n + 10*n^2 := by sorry

-- Part 2
theorem fraction_division (x y : ℝ) (hx : x ≠ 0) (hxy : x ≠ y) :
  (x - y) / x / (x + (y^2 - 2*x*y) / x) = 1 / (x - y) := by sorry

end expand_and_subtract_fraction_division_l2736_273674


namespace lost_ship_depth_l2736_273640

/-- The depth of a lost ship given the descent rate and time taken to reach it. -/
def depth_of_lost_ship (descent_rate : ℝ) (time_taken : ℝ) : ℝ :=
  descent_rate * time_taken

/-- Theorem: The depth of the lost ship is 2400 feet below sea level. -/
theorem lost_ship_depth :
  let descent_rate : ℝ := 30  -- feet per minute
  let time_taken : ℝ := 80    -- minutes
  depth_of_lost_ship descent_rate time_taken = 2400 := by
  sorry

end lost_ship_depth_l2736_273640


namespace percentage_with_both_colors_l2736_273642

/-- Represents the distribution of flags among children -/
structure FlagDistribution where
  totalFlags : ℕ
  bluePercentage : ℚ
  redPercentage : ℚ
  bothPercentage : ℚ

/-- Theorem stating the percentage of children with both color flags -/
theorem percentage_with_both_colors (fd : FlagDistribution) :
  fd.totalFlags % 2 = 0 ∧
  fd.bluePercentage = 60 / 100 ∧
  fd.redPercentage = 45 / 100 ∧
  fd.bluePercentage + fd.redPercentage > 1 →
  fd.bothPercentage = 5 / 100 := by
  sorry

#check percentage_with_both_colors

end percentage_with_both_colors_l2736_273642


namespace quadratic_roots_and_m_value_l2736_273679

-- Define the quadratic equation
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 + (2-m)*x + (1-m)

-- Theorem statement
theorem quadratic_roots_and_m_value (m : ℝ) :
  (∀ x : ℝ, ∃ y z : ℝ, y ≠ z ∧ quadratic m y = 0 ∧ quadratic m z = 0) ∧
  (m < 0 → (∃ y z : ℝ, y ≠ z ∧ quadratic m y = 0 ∧ quadratic m z = 0 ∧ |y - z| = 3) → m = -3) :=
sorry

end quadratic_roots_and_m_value_l2736_273679


namespace factor_expression_l2736_273678

theorem factor_expression (x : ℝ) : 75 * x^2 + 50 * x = 25 * x * (3 * x + 2) := by
  sorry

end factor_expression_l2736_273678


namespace no_three_digit_perfect_square_sum_l2736_273632

theorem no_three_digit_perfect_square_sum :
  ∀ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 →
  ¬∃ m : ℕ, m^2 = 111 * (a + b + c) :=
by sorry

end no_three_digit_perfect_square_sum_l2736_273632


namespace arithmetic_sequence_property_l2736_273691

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence, if a₃ + a₁₁ = 22, then a₇ = 11 -/
theorem arithmetic_sequence_property (a : ℕ → ℚ) 
  (h_arith : arithmetic_sequence a) (h_sum : a 3 + a 11 = 22) : 
  a 7 = 11 := by
  sorry

end arithmetic_sequence_property_l2736_273691


namespace tory_cookie_sales_l2736_273682

/-- Proves the number of cookie packs Tory sold to his neighbor -/
theorem tory_cookie_sales (total : ℕ) (grandmother : ℕ) (uncle : ℕ) (left_to_sell : ℕ) 
  (h1 : total = 50)
  (h2 : grandmother = 12)
  (h3 : uncle = 7)
  (h4 : left_to_sell = 26) :
  total - left_to_sell - (grandmother + uncle) = 5 := by
  sorry

#check tory_cookie_sales

end tory_cookie_sales_l2736_273682


namespace vector_equality_l2736_273622

/-- Given four non-overlapping points P, A, B, C on a plane, 
    if PA + PB + PC = 0 and AB + AC = m * AP, then m = 3 -/
theorem vector_equality (P A B C : ℝ × ℝ) (m : ℝ) 
  (h1 : (A.1 - P.1, A.2 - P.2) + (B.1 - P.1, B.2 - P.2) + (C.1 - P.1, C.2 - P.2) = (0, 0))
  (h2 : (B.1 - A.1, B.2 - A.2) + (C.1 - A.1, C.2 - A.2) = (m * (A.1 - P.1), m * (A.2 - P.2)))
  (h3 : P ≠ A ∧ P ≠ B ∧ P ≠ C ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C) :
  m = 3 := by
  sorry

end vector_equality_l2736_273622


namespace polynomial_divisibility_l2736_273684

theorem polynomial_divisibility (m n : ℕ) :
  ∃ q : Polynomial ℚ, x^(3*m+2) + (-x^2 - 1)^(3*n+1) + 1 = (x^2 + x + 1) * q := by
  sorry

end polynomial_divisibility_l2736_273684


namespace binomial_zero_binomial_312_0_l2736_273638

theorem binomial_zero (n : ℕ) : Nat.choose n 0 = 1 := by sorry

theorem binomial_312_0 : Nat.choose 312 0 = 1 := by sorry

end binomial_zero_binomial_312_0_l2736_273638


namespace root_sum_arctan_l2736_273648

theorem root_sum_arctan (x₁ x₂ : ℝ) (α β : ℝ) : 
  x₁^2 + 3 * Real.sqrt 3 * x₁ + 4 = 0 →
  x₂^2 + 3 * Real.sqrt 3 * x₂ + 4 = 0 →
  α = Real.arctan x₁ →
  β = Real.arctan x₂ →
  α + β = π / 3 := by
sorry

end root_sum_arctan_l2736_273648


namespace twelfth_odd_multiple_of_five_l2736_273602

theorem twelfth_odd_multiple_of_five : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 2 = 1 ∧ 
  n % 5 = 0 ∧
  (∃ k : ℕ, k = 12 ∧ 
    n = (Finset.filter (λ x => x % 2 = 1 ∧ x % 5 = 0) (Finset.range n)).card) ∧
  n = 115 := by
sorry

end twelfth_odd_multiple_of_five_l2736_273602


namespace car_sale_percentage_l2736_273608

theorem car_sale_percentage (P x : ℝ) : 
  P - 2500 = 30000 →
  x / 100 * P = 30000 - 4000 →
  x = 80 := by
sorry

end car_sale_percentage_l2736_273608


namespace amount_added_to_doubled_number_l2736_273613

theorem amount_added_to_doubled_number (original : ℝ) (total : ℝ) (h1 : original = 6.0) (h2 : 2 * original + (total - 2 * original) = 17) : 
  total - 2 * original = 5.0 := by
  sorry

end amount_added_to_doubled_number_l2736_273613


namespace greatest_multiple_of_four_l2736_273683

theorem greatest_multiple_of_four (x : ℕ) : 
  x > 0 ∧ 
  ∃ k : ℕ, x = 4 * k ∧ 
  x^3 < 8000 → 
  x ≤ 16 ∧ 
  ∃ y : ℕ, y > 0 ∧ ∃ m : ℕ, y = 4 * m ∧ y^3 < 8000 ∧ y = 16 :=
by sorry

end greatest_multiple_of_four_l2736_273683


namespace line_slope_and_intercept_l2736_273619

theorem line_slope_and_intercept :
  ∀ (k b : ℝ),
  (∀ x y : ℝ, 3 * x + 2 * y + 6 = 0 ↔ y = k * x + b) →
  k = -3/2 ∧ b = -3 := by
  sorry

end line_slope_and_intercept_l2736_273619


namespace smallest_natural_with_last_four_digits_l2736_273645

theorem smallest_natural_with_last_four_digits : ∃ (N : ℕ), 
  (∀ (k : ℕ), k < N → ¬(47 * k ≡ 1969 [ZMOD 10000])) ∧ 
  (47 * N ≡ 1969 [ZMOD 10000]) := by
  sorry

end smallest_natural_with_last_four_digits_l2736_273645


namespace sum_of_abs_coeff_equals_729_l2736_273650

/-- Given a polynomial p(x) = a₆x⁶ + a₅x⁵ + ... + a₁x + a₀ that equals (2x-1)⁶,
    the sum of the absolute values of its coefficients is 729. -/
theorem sum_of_abs_coeff_equals_729 (a : Fin 7 → ℤ) : 
  (∀ x, (2*x - 1)^6 = a 6 * x^6 + a 5 * x^5 + a 4 * x^4 + a 3 * x^3 + a 2 * x^2 + a 1 * x + a 0) →
  (|a 0| + |a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| = 729) :=
by sorry

end sum_of_abs_coeff_equals_729_l2736_273650


namespace special_right_triangle_median_property_l2736_273639

/-- A right triangle with a special median property -/
structure SpecialRightTriangle where
  -- Points of the triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- B is the right angle
  right_angle : (B.1 - A.1) * (B.1 - C.1) + (B.2 - A.2) * (B.2 - C.2) = 0
  -- BM is the median from B to AC
  M : ℝ × ℝ
  is_median : M = ((A.1 + C.1) / 2, (A.2 + C.2) / 2)
  -- The special property BM² = AB·BC
  special_property : 
    ((M.1 - B.1)^2 + (M.2 - B.2)^2) = 
    (((A.1 - B.1)^2 + (A.2 - B.2)^2) * ((C.1 - B.1)^2 + (C.2 - B.2)^2))^(1/2)

/-- Theorem: In a SpecialRightTriangle, BM = 1/2 AC -/
theorem special_right_triangle_median_property (t : SpecialRightTriangle) :
  ((t.M.1 - t.B.1)^2 + (t.M.2 - t.B.2)^2) = 
  (1/4) * ((t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2) := by
  sorry

end special_right_triangle_median_property_l2736_273639


namespace discount_difference_l2736_273687

theorem discount_difference : 
  let original_bill : ℝ := 10000
  let single_discount_rate : ℝ := 0.4
  let first_successive_discount_rate : ℝ := 0.36
  let second_successive_discount_rate : ℝ := 0.04
  let single_discounted_amount : ℝ := original_bill * (1 - single_discount_rate)
  let successive_discounted_amount : ℝ := original_bill * (1 - first_successive_discount_rate) * (1 - second_successive_discount_rate)
  successive_discounted_amount - single_discounted_amount = 144 := by
  sorry

end discount_difference_l2736_273687


namespace trajectory_of_moving_circle_l2736_273651

-- Define the fixed circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 1

-- Define the moving circle M
def M (x y r : ℝ) : Prop := ∃ (x₀ y₀ : ℝ), (x - x₀)^2 + (y - y₀)^2 = r^2

-- Define tangency condition
def isTangent (c₁ c₂ : ℝ → ℝ → Prop) (m : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∀ x y r, m x y r → (c₁ x y ∨ c₂ x y)

-- Main theorem
theorem trajectory_of_moving_circle :
  ∀ x y : ℝ, isTangent C₁ C₂ M → (x = 0 ∨ x^2 - y^2 / 3 = 1) :=
sorry

end trajectory_of_moving_circle_l2736_273651


namespace ellipse_properties_l2736_273647

/-- An ellipse with specific properties -/
structure SpecificEllipse where
  foci_on_y_axis : Bool
  center_at_origin : Bool
  minor_axis_length : ℝ
  eccentricity : ℝ

/-- A line passing through a point -/
structure Line where
  point : ℝ × ℝ

/-- A point satisfying a specific condition -/
structure SpecialPoint where
  coords : ℝ × ℝ
  condition : Bool

/-- Theorem about the specific ellipse and related geometric properties -/
theorem ellipse_properties (e : SpecificEllipse) (l : Line) (m : SpecialPoint) :
  e.foci_on_y_axis ∧
  e.center_at_origin ∧
  e.minor_axis_length = 2 * Real.sqrt 3 ∧
  e.eccentricity = 1 / 2 ∧
  l.point = (0, 3) ∧
  m.coords = (2, 0) ∧
  m.condition →
  (∃ (x y : ℝ), y^2 / 4 + x^2 / 3 = 1) ∧
  (∃ (d : ℝ), 0 ≤ d ∧ d < (48 + 8 * Real.sqrt 15) / 21) :=
by sorry

end ellipse_properties_l2736_273647


namespace intersection_complement_equality_l2736_273655

theorem intersection_complement_equality (U A B : Set Nat) : 
  U = {1, 2, 3, 4, 5} → 
  A = {1, 3} → 
  B = {2, 5} → 
  A ∩ (U \ B) = {1, 3} := by
sorry

end intersection_complement_equality_l2736_273655


namespace ball_distribution_ratio_l2736_273623

def num_balls : ℕ := 25
def num_bins : ℕ := 5

def count_distribution (d : List ℕ) : ℕ :=
  (List.prod (d.map (λ x => Nat.choose num_balls x))) / (Nat.factorial (List.length d))

theorem ball_distribution_ratio :
  let r := count_distribution [6, 7, 4, 4, 4] * Nat.factorial 5
  let s := count_distribution [5, 5, 5, 5, 5]
  (r : ℚ) / s = 10 := by
  sorry

end ball_distribution_ratio_l2736_273623


namespace sin_cos_identity_l2736_273695

theorem sin_cos_identity : 
  Real.sin (75 * π / 180) * Real.cos (30 * π / 180) - 
  Real.sin (15 * π / 180) * Real.sin (150 * π / 180) = 
  Real.sqrt 2 / 2 := by
  sorry

end sin_cos_identity_l2736_273695


namespace discount_amount_l2736_273653

/-- Given a shirt with an original price and a discounted price, 
    the discount amount is the difference between the two prices. -/
theorem discount_amount (original_price discounted_price : ℕ) :
  original_price = 22 →
  discounted_price = 16 →
  original_price - discounted_price = 6 := by
sorry

end discount_amount_l2736_273653


namespace shaded_area_square_l2736_273630

theorem shaded_area_square (a : ℝ) (h : a = 4) : 
  let square_area := a ^ 2
  let shaded_area := square_area / 2
  shaded_area = 8 := by
sorry

end shaded_area_square_l2736_273630


namespace main_theorem_l2736_273652

/-- The function y in terms of x and m -/
def y (x m : ℝ) : ℝ := (m + 1) * x^2 - m * x + m - 1

/-- The condition for y < 0 having no solution -/
def no_solution_condition (m : ℝ) : Prop :=
  ∀ x, y x m ≥ 0

/-- The solution set for y ≥ m when m > -2 -/
def solution_set (m : ℝ) : Set ℝ :=
  {x | y x m ≥ m}

theorem main_theorem :
  (∀ m : ℝ, no_solution_condition m ↔ m ≥ 2 * Real.sqrt 3 / 3) ∧
  (∀ m : ℝ, m > -2 →
    (m = -1 → solution_set m = {x | x ≥ 1}) ∧
    (m > -1 → solution_set m = {x | x ≤ -1/(m+1) ∨ x ≥ 1}) ∧
    (-2 < m ∧ m < -1 → solution_set m = {x | 1 ≤ x ∧ x ≤ -1/(m+1)})) :=
by sorry

end main_theorem_l2736_273652


namespace isosceles_triangle_inscribed_circle_and_orthocenter_l2736_273654

/-- An isosceles triangle with unit-length legs -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ := 1

/-- The radius of the inscribed circle of an isosceles triangle -/
noncomputable def inscribedRadius (t : IsoscelesTriangle) : ℝ := sorry

/-- The orthocenter of an isosceles triangle -/
noncomputable def orthocenter (t : IsoscelesTriangle) : ℝ × ℝ := sorry

/-- A point on the semicircle drawn on the base of the triangle -/
noncomputable def semicirclePoint (t : IsoscelesTriangle) : ℝ × ℝ := sorry

theorem isosceles_triangle_inscribed_circle_and_orthocenter 
  (t : IsoscelesTriangle) : 
  (∃ (max_t : IsoscelesTriangle), 
    (∀ (other_t : IsoscelesTriangle), inscribedRadius max_t ≥ inscribedRadius other_t) ∧
    max_t.base = Real.sqrt 5 - 1 ∧
    semicirclePoint max_t = orthocenter max_t) := by
  sorry

end isosceles_triangle_inscribed_circle_and_orthocenter_l2736_273654


namespace common_difference_of_arithmetic_sequence_l2736_273688

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem common_difference_of_arithmetic_sequence (a : ℕ → ℝ) :
  a 1 = 1 →
  a 4 = ∫ x in (1 : ℝ)..2, 3 * x^2 →
  ∃ d, arithmetic_sequence a d ∧ d = 2 :=
sorry

end common_difference_of_arithmetic_sequence_l2736_273688


namespace train_speed_l2736_273676

/- Define the train length in meters -/
def train_length : ℝ := 160

/- Define the time taken to pass in seconds -/
def passing_time : ℝ := 8

/- Define the conversion factor from m/s to km/h -/
def ms_to_kmh : ℝ := 3.6

/- Theorem statement -/
theorem train_speed : 
  (train_length / passing_time) * ms_to_kmh = 72 := by
  sorry

end train_speed_l2736_273676


namespace simplify_fraction_l2736_273662

theorem simplify_fraction : (4^4 + 4^2) / (4^3 - 4) = 17 / 3 := by sorry

end simplify_fraction_l2736_273662


namespace inequality_system_solution_l2736_273689

theorem inequality_system_solution (m : ℝ) : 
  (∀ x : ℝ, (x / 3 < 1 - (x - 3) / 6 ∧ x < m) ↔ x < 3) → m ≥ 3 := by
  sorry

end inequality_system_solution_l2736_273689


namespace canoe_kayak_ratio_l2736_273668

/-- Represents the rental business scenario -/
structure RentalBusiness where
  canoe_price : ℕ
  kayak_price : ℕ
  total_revenue : ℕ
  canoe_kayak_difference : ℕ

/-- Theorem stating the ratio of canoes to kayaks rented -/
theorem canoe_kayak_ratio (rb : RentalBusiness) 
  (h1 : rb.canoe_price = 14)
  (h2 : rb.kayak_price = 15)
  (h3 : rb.total_revenue = 288)
  (h4 : rb.canoe_kayak_difference = 4)
  (h5 : ∃ (k : ℕ), rb.canoe_price * (k + rb.canoe_kayak_difference) + rb.kayak_price * k = rb.total_revenue) :
  ∃ (c k : ℕ), c = k + rb.canoe_kayak_difference ∧ c * rb.canoe_price + k * rb.kayak_price = rb.total_revenue ∧ c * 2 = k * 3 :=
by sorry

end canoe_kayak_ratio_l2736_273668


namespace parallel_vectors_solution_perpendicular_vectors_solution_l2736_273635

-- Define the vectors
def a (x : ℝ) : ℝ × ℝ := (1, 2*x)
def b (x : ℝ) : ℝ × ℝ := (x, 3)
def c : ℝ × ℝ := (-2, 0)

-- Question 1: Parallel vectors condition
def parallel_condition (x : ℝ) : Prop :=
  (1 + 2*x) * 4*x = 4*(2*x + 6)

-- Question 2: Perpendicular vectors condition
def perpendicular_condition (x : ℝ) : Prop :=
  8*x^2 + 32*x + 4 = 0

-- Theorem for parallel vectors
theorem parallel_vectors_solution :
  ∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = -3/2 ∧ parallel_condition x₁ ∧ parallel_condition x₂ :=
sorry

-- Theorem for perpendicular vectors
theorem perpendicular_vectors_solution :
  ∃ x₁ x₂ : ℝ, x₁ = (-4 + Real.sqrt 14)/2 ∧ x₂ = (-4 - Real.sqrt 14)/2 ∧
  perpendicular_condition x₁ ∧ perpendicular_condition x₂ :=
sorry

end parallel_vectors_solution_perpendicular_vectors_solution_l2736_273635


namespace g_of_g_is_even_l2736_273629

def is_even_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = g x

theorem g_of_g_is_even (g : ℝ → ℝ) (h : is_even_function g) : is_even_function (g ∘ g) := by
  sorry

end g_of_g_is_even_l2736_273629


namespace sum_of_fractions_equals_one_l2736_273685

theorem sum_of_fractions_equals_one
  (a b c x y z : ℝ)
  (eq1 : 11 * x + b * y + c * z = 0)
  (eq2 : a * x + 19 * y + c * z = 0)
  (eq3 : a * x + b * y + 37 * z = 0)
  (h1 : a ≠ 11)
  (h2 : x ≠ 0) :
  a / (a - 11) + b / (b - 19) + c / (c - 37) = 1 := by
sorry


end sum_of_fractions_equals_one_l2736_273685


namespace orange_cost_18_pounds_l2736_273672

/-- Calculates the cost of oranges given a rate and a desired weight -/
def orangeCost (ratePrice : ℚ) (rateWeight : ℚ) (desiredWeight : ℚ) : ℚ :=
  (ratePrice / rateWeight) * desiredWeight

theorem orange_cost_18_pounds :
  orangeCost 5 6 18 = 15 := by
  sorry

end orange_cost_18_pounds_l2736_273672


namespace expression_value_l2736_273670

theorem expression_value (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (x^3 + 3*y^3) / 9 = 24.33 := by
  sorry

end expression_value_l2736_273670


namespace decimal_point_problem_l2736_273627

theorem decimal_point_problem :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
  x + y = 13.5927 ∧
  y = 10 * x ∧
  x = 1.2357 ∧ y = 12.357 := by
sorry

end decimal_point_problem_l2736_273627


namespace product_of_two_numbers_l2736_273694

theorem product_of_two_numbers (x y : ℝ) (h1 : x - y = 12) (h2 : x^2 + y^2 = 340) : 
  x * y = 97.9450625 := by
sorry

end product_of_two_numbers_l2736_273694


namespace hcl_equals_h2o_l2736_273643

-- Define the chemical reaction
structure ChemicalReaction where
  hcl : ℝ  -- moles of Hydrochloric acid
  nahco3 : ℝ  -- moles of Sodium bicarbonate
  h2o : ℝ  -- moles of Water formed

-- Define the conditions of the problem
def reaction_conditions (r : ChemicalReaction) : Prop :=
  r.nahco3 = 1 ∧ r.h2o = 1

-- Theorem statement
theorem hcl_equals_h2o (r : ChemicalReaction) 
  (h : reaction_conditions r) : r.hcl = r.h2o := by
  sorry

#check hcl_equals_h2o

end hcl_equals_h2o_l2736_273643


namespace quadratic_contradiction_l2736_273616

theorem quadratic_contradiction : ¬ ∃ (a b c : ℝ), 
  ((a > 0 ∧ b > 0 ∧ c < 0) ∨ (a > 0 ∧ b < 0 ∧ c > 0) ∨ (a < 0 ∧ b > 0 ∧ c > 0)) ∧
  ((a < 0 ∧ b < 0 ∧ c > 0) ∨ (a < 0 ∧ b > 0 ∧ c < 0) ∨ (a > 0 ∧ b < 0 ∧ c < 0)) :=
by sorry

end quadratic_contradiction_l2736_273616


namespace chocolate_mixture_percentage_l2736_273660

theorem chocolate_mixture_percentage (initial_amount : ℝ) (initial_percentage : ℝ) 
  (added_amount : ℝ) (desired_percentage : ℝ) : 
  initial_amount = 220 →
  initial_percentage = 0.5 →
  added_amount = 220 →
  desired_percentage = 0.75 →
  (initial_amount * initial_percentage + added_amount) / (initial_amount + added_amount) = desired_percentage :=
by sorry

end chocolate_mixture_percentage_l2736_273660


namespace b_is_killer_l2736_273671

-- Define the characters
inductive Character : Type
| A : Character
| B : Character
| C : Character

-- Define the actions
def poisoned_water (x y : Character) : Prop := x = Character.A ∧ y = Character.C
def made_hole (x y : Character) : Prop := x = Character.B ∧ y = Character.C
def died_of_thirst (x : Character) : Prop := x = Character.C

-- Define the killer
def is_killer (x : Character) : Prop := x = Character.B

-- Theorem statement
theorem b_is_killer 
  (h1 : poisoned_water Character.A Character.C)
  (h2 : made_hole Character.B Character.C)
  (h3 : died_of_thirst Character.C) :
  is_killer Character.B :=
sorry

end b_is_killer_l2736_273671


namespace hockey_players_l2736_273634

theorem hockey_players (n : ℕ) : 
  n < 30 ∧ 
  2 ∣ n ∧ 
  4 ∣ n ∧ 
  7 ∣ n → 
  n / 4 = 7 := by
  sorry

end hockey_players_l2736_273634


namespace three_conclusions_correct_l2736_273606

-- Define the "heap" for natural numbers
def heap (r : Nat) : Set Nat := {n : Nat | ∃ k : Nat, n = 3 * k + r}

-- Define the four conclusions
def conclusion1 : Prop := 2011 ∈ heap 1
def conclusion2 : Prop := ∀ a b : Nat, a ∈ heap 1 → b ∈ heap 2 → (a + b) ∈ heap 0
def conclusion3 : Prop := (heap 0) ∪ (heap 1) ∪ (heap 2) = Set.univ
def conclusion4 : Prop := ∀ r : Fin 3, ∀ a b : Nat, a ∈ heap r → b ∈ heap r → (a - b) ∉ heap r

-- Theorem stating that exactly 3 out of 4 conclusions are correct
theorem three_conclusions_correct :
  (conclusion1 ∧ conclusion2 ∧ conclusion3 ∧ ¬conclusion4) ∨
  (conclusion1 ∧ conclusion2 ∧ ¬conclusion3 ∧ conclusion4) ∨
  (conclusion1 ∧ ¬conclusion2 ∧ conclusion3 ∧ conclusion4) ∨
  (¬conclusion1 ∧ conclusion2 ∧ conclusion3 ∧ conclusion4) :=
sorry

end three_conclusions_correct_l2736_273606


namespace complex_square_l2736_273605

/-- Given that i^2 = -1, prove that (3 - 4i)^2 = 5 - 24i -/
theorem complex_square (i : ℂ) (h : i^2 = -1) : (3 - 4*i)^2 = 5 - 24*i := by
  sorry

end complex_square_l2736_273605


namespace chocolate_problem_l2736_273618

theorem chocolate_problem (cost_price selling_price : ℝ) 
  (h1 : cost_price * 81 = selling_price * 45)
  (h2 : (selling_price - cost_price) / cost_price = 0.8) :
  81 = 81 := by sorry

end chocolate_problem_l2736_273618


namespace twelve_point_polygons_l2736_273699

/-- The number of distinct convex polygons with 3 or more sides that can be formed from n points on a circle -/
def convex_polygons (n : ℕ) : ℕ :=
  2^n - 1 - n - (n.choose 2)

theorem twelve_point_polygons :
  convex_polygons 12 = 4017 := by
  sorry

end twelve_point_polygons_l2736_273699


namespace team_size_l2736_273628

theorem team_size (average_age : ℝ) (leader_age : ℝ) (average_age_without_leader : ℝ) 
  (h1 : average_age = 25)
  (h2 : leader_age = 45)
  (h3 : average_age_without_leader = 23) :
  ∃ n : ℕ, n * average_age = (n - 1) * average_age_without_leader + leader_age ∧ n = 11 :=
by
  sorry

end team_size_l2736_273628


namespace fraction_square_value_l2736_273601

theorem fraction_square_value (x y : ℚ) (hx : x = 3) (hy : y = 5) :
  ((1 / y) / (1 / x))^2 = 9 / 25 := by
  sorry

end fraction_square_value_l2736_273601


namespace opposite_of_negative_eight_l2736_273637

-- Define the concept of opposite
def opposite (x : Int) : Int := -x

-- State the theorem
theorem opposite_of_negative_eight :
  opposite (-8) = 8 := by sorry

end opposite_of_negative_eight_l2736_273637


namespace complex_counterexample_l2736_273609

theorem complex_counterexample : ∃ z₁ z₂ : ℂ, (Complex.abs z₁ = Complex.abs z₂) ∧ (z₁^2 ≠ z₂^2) := by
  sorry

end complex_counterexample_l2736_273609


namespace max_marbles_for_score_l2736_273611

/-- Represents the size of a marble -/
inductive MarbleSize
| Small
| Medium
| Large

/-- Represents a hole with its score -/
structure Hole :=
  (number : Nat)
  (score : Nat)

/-- Represents the game setup -/
structure GameSetup :=
  (holes : List Hole)
  (maxMarbles : Nat)
  (totalScore : Nat)

/-- Checks if a marble can go through a hole -/
def canGoThrough (size : MarbleSize) (hole : Hole) : Bool :=
  match size with
  | MarbleSize.Small => true
  | MarbleSize.Medium => hole.number ≥ 3
  | MarbleSize.Large => hole.number = 5

/-- Represents a valid game configuration -/
structure GameConfig :=
  (smallMarbles : List Hole)
  (mediumMarbles : List Hole)
  (largeMarbles : List Hole)

/-- Calculates the total score for a game configuration -/
def totalScore (config : GameConfig) : Nat :=
  (config.smallMarbles.map (·.score)).sum +
  (config.mediumMarbles.map (·.score)).sum +
  (config.largeMarbles.map (·.score)).sum

/-- Calculates the total number of marbles used in a game configuration -/
def totalMarbles (config : GameConfig) : Nat :=
  config.smallMarbles.length +
  config.mediumMarbles.length +
  config.largeMarbles.length

/-- The main theorem to prove -/
theorem max_marbles_for_score (setup : GameSetup) :
  (∃ (config : GameConfig),
    totalScore config = setup.totalScore ∧
    totalMarbles config ≤ setup.maxMarbles ∧
    (∀ (other : GameConfig),
      totalScore other = setup.totalScore →
      totalMarbles other ≤ totalMarbles config)) →
  (∃ (maxConfig : GameConfig),
    totalScore maxConfig = setup.totalScore ∧
    totalMarbles maxConfig = 14 ∧
    (∀ (other : GameConfig),
      totalScore other = setup.totalScore →
      totalMarbles other ≤ 14)) :=
by sorry

end max_marbles_for_score_l2736_273611


namespace total_retail_price_proof_l2736_273641

def calculate_retail_price (wholesale_price : ℝ) (profit_margin : ℝ) : ℝ :=
  wholesale_price * (1 + profit_margin)

theorem total_retail_price_proof 
  (P Q R : ℝ)
  (discount1 discount2 discount3 : ℝ)
  (profit_margin1 profit_margin2 profit_margin3 : ℝ)
  (h1 : P = 90)
  (h2 : Q = 120)
  (h3 : R = 150)
  (h4 : discount1 = 0.10)
  (h5 : discount2 = 0.15)
  (h6 : discount3 = 0.20)
  (h7 : profit_margin1 = 0.20)
  (h8 : profit_margin2 = 0.25)
  (h9 : profit_margin3 = 0.30) :
  calculate_retail_price P profit_margin1 +
  calculate_retail_price Q profit_margin2 +
  calculate_retail_price R profit_margin3 = 453 := by
sorry

end total_retail_price_proof_l2736_273641


namespace algebraic_expression_value_l2736_273686

theorem algebraic_expression_value (x : ℝ) : 
  x^2 + 2*x + 7 = 6 → 4*x^2 + 8*x - 5 = -9 := by
  sorry

end algebraic_expression_value_l2736_273686


namespace triangle_abc_properties_l2736_273633

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →  -- Angles are in (0, π)
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Sides are positive
  A + B + C = π →  -- Angle sum in a triangle
  (2*c - a) * Real.cos B = b * Real.cos A →  -- Given equation
  b = 6 →  -- Given condition
  c = 2*a →  -- Given condition
  B = π/3 ∧ (1/2) * a * c * Real.sin B = 6 * Real.sqrt 3 := by
  sorry

end triangle_abc_properties_l2736_273633


namespace polynomial_equality_l2736_273610

theorem polynomial_equality (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ - a₁ + a₂ - a₃ + a₄ - a₅ = -243 := by
sorry

end polynomial_equality_l2736_273610


namespace add_minutes_theorem_l2736_273697

/-- Represents a date and time -/
structure DateTime where
  year : ℕ
  month : ℕ
  day : ℕ
  hour : ℕ
  minute : ℕ

/-- Adds minutes to a DateTime -/
def addMinutes (dt : DateTime) (minutes : ℕ) : DateTime :=
  sorry

/-- The starting DateTime -/
def startTime : DateTime :=
  { year := 2020, month := 2, day := 1, hour := 18, minute := 0 }

/-- The number of minutes to add -/
def minutesToAdd : ℕ := 3457

/-- The expected end DateTime -/
def endTime : DateTime :=
  { year := 2020, month := 2, day := 4, hour := 3, minute := 37 }

/-- Theorem stating that adding minutesToAdd to startTime results in endTime -/
theorem add_minutes_theorem : addMinutes startTime minutesToAdd = endTime :=
  sorry

end add_minutes_theorem_l2736_273697


namespace plane_parallel_criterion_l2736_273612

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation between planes
variable (plane_parallel : Plane → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (line_parallel_plane : Line → Plane → Prop)

-- Define the relation of a line being in a plane
variable (line_in_plane : Line → Plane → Prop)

-- Theorem statement
theorem plane_parallel_criterion
  (α β : Plane)
  (h : ∀ l : Line, line_in_plane l α → line_parallel_plane l β) :
  plane_parallel α β :=
sorry

end plane_parallel_criterion_l2736_273612


namespace solution_set_is_circle_minus_point_l2736_273617

theorem solution_set_is_circle_minus_point :
  ∀ (x y a : ℝ),
  (a * x + y = 2 * a + 3 ∧ x - a * y = a + 4) ↔
  ((x - 3)^2 + (y - 1)^2 = 5 ∧ (x, y) ≠ (2, -1)) :=
by sorry

end solution_set_is_circle_minus_point_l2736_273617


namespace n_has_nine_digits_l2736_273625

/-- The smallest positive integer satisfying the given conditions -/
def n : ℕ := sorry

/-- n is divisible by 30 -/
axiom n_div_30 : 30 ∣ n

/-- n^2 is a perfect cube -/
axiom n_sq_cube : ∃ k : ℕ, n^2 = k^3

/-- n^3 is a perfect square -/
axiom n_cube_square : ∃ k : ℕ, n^3 = k^2

/-- n is the smallest positive integer satisfying the conditions -/
axiom n_smallest : ∀ m : ℕ, m > 0 → (30 ∣ m) → (∃ k : ℕ, m^2 = k^3) → (∃ k : ℕ, m^3 = k^2) → m ≥ n

/-- The number of digits in n -/
def digits (x : ℕ) : ℕ := sorry

theorem n_has_nine_digits : digits n = 9 := by sorry

end n_has_nine_digits_l2736_273625


namespace sum_of_cubes_l2736_273649

theorem sum_of_cubes (a b c : ℕ+) :
  (a + b + c)^3 - a^3 - b^3 - c^3 = 294 →
  a + b + c = 8 := by
sorry

end sum_of_cubes_l2736_273649
