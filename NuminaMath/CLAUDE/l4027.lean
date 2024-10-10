import Mathlib

namespace son_shoveling_time_l4027_402705

/-- Given a driveway shoveling scenario with three people, this theorem proves
    the time it takes for the son to shovel the entire driveway alone. -/
theorem son_shoveling_time (wayne_rate son_rate neighbor_rate : ℝ) 
  (h1 : wayne_rate = 6 * son_rate) 
  (h2 : neighbor_rate = 2 * wayne_rate) 
  (h3 : son_rate + wayne_rate + neighbor_rate = 1 / 2) : 
  1 / son_rate = 38 := by
  sorry

end son_shoveling_time_l4027_402705


namespace unique_solution_condition_l4027_402772

theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (x + 3) / (k * x - 2) = x) ↔ k = -3/4 := by sorry

end unique_solution_condition_l4027_402772


namespace box_weight_example_l4027_402726

/-- Calculates the weight of an open box given its dimensions, thickness, and metal density. -/
def box_weight (length width height thickness : ℝ) (metal_density : ℝ) : ℝ :=
  let outer_volume := length * width * height
  let inner_length := length - 2 * thickness
  let inner_width := width - 2 * thickness
  let inner_height := height - thickness
  let inner_volume := inner_length * inner_width * inner_height
  let metal_volume := outer_volume - inner_volume
  metal_volume * metal_density

/-- Theorem stating that the weight of the specified box is 5504 grams. -/
theorem box_weight_example : 
  box_weight 50 40 23 2 0.5 = 5504 := by
  sorry

end box_weight_example_l4027_402726


namespace discount_calculation_l4027_402758

/-- Calculates the total discount percentage given initial, member, and special promotion discounts -/
def total_discount (initial_discount : ℝ) (member_discount : ℝ) (special_discount : ℝ) : ℝ :=
  let remaining_after_initial := 1 - initial_discount
  let remaining_after_member := remaining_after_initial * (1 - member_discount)
  let final_remaining := remaining_after_member * (1 - special_discount)
  (1 - final_remaining) * 100

/-- Theorem stating that the total discount is 65.8% given the specific discounts -/
theorem discount_calculation :
  total_discount 0.6 0.1 0.05 = 65.8 := by
  sorry

end discount_calculation_l4027_402758


namespace first_friend_cookies_l4027_402780

theorem first_friend_cookies (initial : ℕ) (eaten : ℕ) (brother : ℕ) (second : ℕ) (third : ℕ) (remaining : ℕ) : 
  initial = 22 → 
  eaten = 2 → 
  brother = 1 → 
  second = 5 → 
  third = 5 → 
  remaining = 6 → 
  initial - eaten - brother - second - third - remaining = 3 := by
  sorry

end first_friend_cookies_l4027_402780


namespace peach_basket_problem_l4027_402713

theorem peach_basket_problem (x : ℕ) : 
  (x > 0) →
  (x - (x / 2 + 1) > 0) →
  (x - (x / 2 + 1) - ((x - (x / 2 + 1)) / 2 - 1) = 4) →
  (x = 14) :=
by
  sorry

#check peach_basket_problem

end peach_basket_problem_l4027_402713


namespace problem_solution_l4027_402746

theorem problem_solution (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = -1) :
  a^2 * b - 2 * a * b + a * b^2 = -1 := by
  sorry

end problem_solution_l4027_402746


namespace train_overtake_l4027_402701

/-- The speed of Train A in miles per hour -/
def speed_a : ℝ := 30

/-- The speed of Train B in miles per hour -/
def speed_b : ℝ := 38

/-- The time difference between Train A and Train B's departure in hours -/
def time_diff : ℝ := 2

/-- The distance at which Train B overtakes Train A -/
def overtake_distance : ℝ := 285

theorem train_overtake :
  ∃ t : ℝ, t > 0 ∧ speed_b * t = speed_a * (t + time_diff) ∧ 
  overtake_distance = speed_b * t :=
sorry

end train_overtake_l4027_402701


namespace ordering_of_powers_l4027_402798

theorem ordering_of_powers : 3^15 < 4^12 ∧ 4^12 < 8^9 := by
  sorry

end ordering_of_powers_l4027_402798


namespace some_number_value_l4027_402745

theorem some_number_value (x : ℝ) (h : (55 + 113 / x) * x = 4403) : x = 78 := by
  sorry

end some_number_value_l4027_402745


namespace at_op_sum_equals_six_l4027_402737

-- Define the @ operation for positive integers
def at_op (a b : ℕ+) : ℚ := (a * b : ℚ) / (a + b : ℚ)

-- State the theorem
theorem at_op_sum_equals_six :
  at_op 7 14 + at_op 2 4 = 6 := by sorry

end at_op_sum_equals_six_l4027_402737


namespace walter_fish_fry_guests_l4027_402769

-- Define the constants from the problem
def hushpuppies_per_guest : ℕ := 5
def hushpuppies_per_batch : ℕ := 10
def minutes_per_batch : ℕ := 8
def total_cooking_time : ℕ := 80

-- Define the function to calculate the number of guests
def number_of_guests : ℕ :=
  (total_cooking_time / minutes_per_batch * hushpuppies_per_batch) / hushpuppies_per_guest

-- State the theorem
theorem walter_fish_fry_guests :
  number_of_guests = 20 :=
sorry

end walter_fish_fry_guests_l4027_402769


namespace cylinder_height_relationship_l4027_402718

-- Define the cylinders
structure Cylinder where
  radius : ℝ
  height : ℝ

-- Define the theorem
theorem cylinder_height_relationship (c1 c2 : Cylinder) : 
  -- Conditions
  (c1.radius * c1.radius * c1.height = c2.radius * c2.radius * c2.height) →  -- Equal volumes
  (c2.radius = 1.2 * c1.radius) →                                            -- Second radius is 20% more
  -- Conclusion
  (c1.height = 1.44 * c2.height) :=                                          -- First height is 44% more
by
  sorry

end cylinder_height_relationship_l4027_402718


namespace parentheses_removal_equality_l4027_402734

theorem parentheses_removal_equality (a c : ℝ) : 3*a - (2*a - c) = 3*a - 2*a + c := by
  sorry

end parentheses_removal_equality_l4027_402734


namespace vegetable_planting_methods_l4027_402711

theorem vegetable_planting_methods :
  let total_vegetables : ℕ := 4
  let vegetables_to_choose : ℕ := 3
  let remaining_choices : ℕ := total_vegetables - 1  -- Cucumber is always chosen
  let remaining_to_choose : ℕ := vegetables_to_choose - 1
  let soil_types : ℕ := 3
  
  (remaining_choices.choose remaining_to_choose) * (vegetables_to_choose.factorial) = 18 :=
by sorry

end vegetable_planting_methods_l4027_402711


namespace non_monotonic_range_l4027_402732

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 1)*x + 4*a else Real.log x / Real.log a

def is_monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

theorem non_monotonic_range (a : ℝ) : 
  (¬ is_monotonic (f a)) ↔ 
  (0 < a ∧ a < 1/7) ∨ (1/3 ≤ a ∧ a < 1) ∨ (a > 1) :=
sorry

end non_monotonic_range_l4027_402732


namespace flow_rate_difference_l4027_402792

/-- Proves that the difference between 0.6 times the original flow rate and the reduced flow rate is 1 gallon per minute -/
theorem flow_rate_difference (original_rate reduced_rate : ℝ) 
  (h1 : original_rate = 5.0)
  (h2 : reduced_rate = 2) : 
  0.6 * original_rate - reduced_rate = 1 := by
  sorry

end flow_rate_difference_l4027_402792


namespace courtyard_tile_cost_l4027_402717

/-- Calculate the total cost of tiles for a courtyard -/
theorem courtyard_tile_cost : 
  let courtyard_length : ℝ := 10
  let courtyard_width : ℝ := 25
  let tiles_per_sqft : ℝ := 4
  let green_tile_percentage : ℝ := 0.4
  let green_tile_cost : ℝ := 3
  let red_tile_cost : ℝ := 1.5

  let total_area : ℝ := courtyard_length * courtyard_width
  let total_tiles : ℝ := total_area * tiles_per_sqft
  let green_tiles : ℝ := green_tile_percentage * total_tiles
  let red_tiles : ℝ := total_tiles - green_tiles

  let total_cost : ℝ := green_tiles * green_tile_cost + red_tiles * red_tile_cost

  total_cost = 2100 := by
  sorry

end courtyard_tile_cost_l4027_402717


namespace sugar_per_cup_l4027_402768

def total_sugar : ℝ := 84.6
def num_cups : ℕ := 12

theorem sugar_per_cup : 
  (total_sugar / num_cups : ℝ) = 7.05 := by sorry

end sugar_per_cup_l4027_402768


namespace shortest_side_right_triangle_l4027_402756

theorem shortest_side_right_triangle (a b c : ℝ) : 
  a = 7 → b = 10 → c^2 = a^2 + b^2 → min a (min b c) = 7 := by
  sorry

end shortest_side_right_triangle_l4027_402756


namespace total_hot_dogs_today_l4027_402730

def hot_dogs_lunch : ℕ := 9
def hot_dogs_dinner : ℕ := 2

theorem total_hot_dogs_today : hot_dogs_lunch + hot_dogs_dinner = 11 := by
  sorry

end total_hot_dogs_today_l4027_402730


namespace rational_equation_solution_l4027_402787

theorem rational_equation_solution (C D : ℚ) :
  (∀ x : ℚ, x ≠ 4 ∧ x ≠ 5 →
    (D * x - 17) / (x^2 - 9*x + 20) = C / (x - 4) + 5 / (x - 5)) →
  C + D = 19/5 := by
sorry

end rational_equation_solution_l4027_402787


namespace prime_with_integer_roots_l4027_402771

theorem prime_with_integer_roots (p : ℕ) : 
  Prime p → 
  (∃ x y : ℤ, x^2 + p*x - 530*p = 0 ∧ y^2 + p*y - 530*p = 0) → 
  43 < p ∧ p ≤ 53 := by
sorry

end prime_with_integer_roots_l4027_402771


namespace certain_number_proof_l4027_402770

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem certain_number_proof : 
  ∃! x : ℕ, x > 0 ∧ 
    is_divisible_by (3153 + x) 9 ∧
    is_divisible_by (3153 + x) 70 ∧
    is_divisible_by (3153 + x) 25 ∧
    is_divisible_by (3153 + x) 21 ∧
    ∀ y : ℕ, y > 0 → 
      (is_divisible_by (3153 + y) 9 ∧
       is_divisible_by (3153 + y) 70 ∧
       is_divisible_by (3153 + y) 25 ∧
       is_divisible_by (3153 + y) 21) → 
      x ≤ y :=
by
  sorry

end certain_number_proof_l4027_402770


namespace m_range_when_exists_positive_root_l4027_402707

/-- The quadratic function f(x) = x^2 + mx + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + 1

/-- The proposition that there exists a positive x₀ such that f(x₀) < 0 -/
def exists_positive_root (m : ℝ) : Prop :=
  ∃ x₀ : ℝ, x₀ > 0 ∧ f m x₀ < 0

/-- Theorem stating that if there exists a positive x₀ such that f(x₀) < 0,
    then m is in the open interval (-∞, -2) -/
theorem m_range_when_exists_positive_root :
  ∀ m : ℝ, exists_positive_root m → m < -2 :=
by sorry

end m_range_when_exists_positive_root_l4027_402707


namespace quadratic_transformation_l4027_402712

/-- Given the quadratic equation x^2 + 4x + 4 = 0, which can be transformed
    into the form (x + h)^2 = k, prove that h + k = 2. -/
theorem quadratic_transformation (h k : ℝ) : 
  (∀ x, x^2 + 4*x + 4 = 0 ↔ (x + h)^2 = k) → h + k = 2 :=
by sorry

end quadratic_transformation_l4027_402712


namespace cos_sin_three_pi_eighths_l4027_402750

theorem cos_sin_three_pi_eighths : 
  Real.cos (3 * π / 8) ^ 2 - Real.sin (3 * π / 8) ^ 2 = -Real.sqrt 2 / 2 := by
  sorry

end cos_sin_three_pi_eighths_l4027_402750


namespace library_configuration_count_l4027_402791

/-- The number of different configurations for 8 identical books in a library,
    where at least one book must remain in the library and at least one must be checked out. -/
def library_configurations : ℕ := 7

/-- The total number of books in the library -/
def total_books : ℕ := 8

/-- Proposition that there are exactly 7 different configurations for the books in the library -/
theorem library_configuration_count :
  (∀ config : ℕ, 1 ≤ config ∧ config ≤ total_books - 1) →
  (∀ config : ℕ, config ≤ total_books - config) →
  library_configurations = (total_books - 1) := by
  sorry

end library_configuration_count_l4027_402791


namespace rectangular_to_polar_conversion_l4027_402793

theorem rectangular_to_polar_conversion :
  let x : ℝ := 1
  let y : ℝ := -Real.sqrt 3
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := 5 * Real.pi / 3
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ r = 2 ∧ x = r * Real.cos θ ∧ y = r * Real.sin θ :=
by sorry

end rectangular_to_polar_conversion_l4027_402793


namespace complex_absolute_value_problem_l4027_402724

theorem complex_absolute_value_problem : 
  let z₁ : ℂ := 3 - 5*I
  let z₂ : ℂ := 3 + 5*I
  Complex.abs z₁ * Complex.abs z₂ + 2 * Complex.abs z₁ = 34 + 2 * Real.sqrt 34 :=
by sorry

end complex_absolute_value_problem_l4027_402724


namespace increase_by_percentage_l4027_402714

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (final : ℝ) :
  initial = 70 →
  percentage = 50 →
  final = initial * (1 + percentage / 100) →
  final = 105 := by
  sorry

end increase_by_percentage_l4027_402714


namespace cube_sum_eq_343_l4027_402749

theorem cube_sum_eq_343 (x y : ℝ) :
  x^3 + 21*x*y + y^3 = 343 → x + y = 7 ∨ x + y = -14 := by
  sorry

end cube_sum_eq_343_l4027_402749


namespace initial_coloring_books_count_l4027_402797

/-- Proves that the initial number of coloring books in stock is 40 --/
theorem initial_coloring_books_count (books_sold : ℕ) (books_per_shelf : ℕ) (shelves_used : ℕ) : 
  books_sold = 20 → books_per_shelf = 4 → shelves_used = 5 → 
  books_sold + books_per_shelf * shelves_used = 40 := by
  sorry

end initial_coloring_books_count_l4027_402797


namespace astronomy_club_committee_probability_l4027_402764

/-- The probability of selecting a committee with more boys than girls -/
theorem astronomy_club_committee_probability :
  let total_members : ℕ := 24
  let boys : ℕ := 14
  let girls : ℕ := 10
  let committee_size : ℕ := 5
  let total_committees : ℕ := Nat.choose total_members committee_size
  let committees_more_boys : ℕ := 
    Nat.choose boys 3 * Nat.choose girls 2 +
    Nat.choose boys 4 * Nat.choose girls 1 +
    Nat.choose boys 5
  (committees_more_boys : ℚ) / total_committees = 7098 / 10626 := by
sorry

end astronomy_club_committee_probability_l4027_402764


namespace sequence_inequality_l4027_402702

theorem sequence_inequality (k : ℝ) : 
  (∀ n : ℕ+, (n + 1)^2 + k*(n + 1) + 2 > n^2 + k*n + 2) ↔ k > -3 :=
by sorry

end sequence_inequality_l4027_402702


namespace smallest_x_satisfying_equation_l4027_402775

theorem smallest_x_satisfying_equation : 
  ∃ (x : ℝ), x > 0 ∧ 
  (⌊x^2⌋ : ℝ) - x * (⌊x⌋ : ℝ) = 8 ∧ 
  (∀ y : ℝ, y > 0 ∧ (⌊y^2⌋ : ℝ) - y * (⌊y⌋ : ℝ) = 8 → x ≤ y) ∧
  x = 89 / 9 := by
sorry

end smallest_x_satisfying_equation_l4027_402775


namespace parallel_vectors_x_value_l4027_402703

/-- Two vectors in ℝ² are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (x, -6)
  parallel a b → x = -4 :=
by
  sorry

#check parallel_vectors_x_value

end parallel_vectors_x_value_l4027_402703


namespace parabola_horizontal_shift_l4027_402751

/-- A parabola is defined by its coefficient and vertex -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The equation of a parabola in vertex form is y = a(x-h)^2 + k -/
def parabola_equation (p : Parabola) (x : ℝ) : ℝ :=
  p.a * (x - p.h)^2 + p.k

theorem parabola_horizontal_shift 
  (p1 p2 : Parabola) 
  (h1 : p1.a = p2.a) 
  (h2 : p1.k = p2.k) 
  (h3 : p1.h = p2.h + 3) : 
  ∀ x, parabola_equation p1 x = parabola_equation p2 (x - 3) :=
sorry

end parabola_horizontal_shift_l4027_402751


namespace sufficient_but_not_necessary_condition_l4027_402740

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a ≥ 10) →
  (∀ x ∈ Set.Icc 1 3, x^2 - a ≤ 0) ∧
  ¬(∀ a : ℝ, (∀ x ∈ Set.Icc 1 3, x^2 - a ≤ 0) → (a ≥ 10)) :=
by sorry

end sufficient_but_not_necessary_condition_l4027_402740


namespace min_omega_value_l4027_402782

/-- Given a function f(x) = 2 * sin(ω * x) where ω > 0, and f(x) has a minimum value of -2
    in the interval [-π/3, π/6], prove that the minimum value of ω is 3/2. -/
theorem min_omega_value (ω : ℝ) : 
  (ω > 0) →
  (∀ x ∈ Set.Icc (-π/3) (π/6), 2 * Real.sin (ω * x) ≥ -2) →
  (∃ x ∈ Set.Icc (-π/3) (π/6), 2 * Real.sin (ω * x) = -2) →
  ω ≥ 3/2 :=
sorry

end min_omega_value_l4027_402782


namespace visible_part_of_third_mountain_l4027_402773

/-- Represents a mountain with a height and position on a great circle. -/
structure Mountain where
  height : ℝ
  position : ℝ

/-- Represents the Earth as a sphere. -/
structure Earth where
  radius : ℝ

/-- Calculates the visible height of a distant mountain. -/
def visibleHeight (earth : Earth) (m1 m2 m3 : Mountain) : ℝ :=
  sorry

theorem visible_part_of_third_mountain
  (earth : Earth)
  (m1 m2 m3 : Mountain)
  (h_earth_radius : earth.radius = 6366000) -- in meters
  (h_m1_height : m1.height = 2500)
  (h_m2_height : m2.height = 3000)
  (h_m3_height : m3.height = 8800)
  (h_m1_m2_distance : m2.position - m1.position = 1 * π / 180) -- 1 degree in radians
  (h_m2_m3_distance : m3.position - m2.position = 1.5 * π / 180) -- 1.5 degrees in radians
  : visibleHeight earth m1 m2 m3 = 1500 := by
  sorry

end visible_part_of_third_mountain_l4027_402773


namespace pythagorean_theorem_l4027_402763

-- Define a right-angled triangle
def RightTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

-- Theorem statement
theorem pythagorean_theorem (a b c : ℝ) :
  RightTriangle a b c → a^2 + b^2 = c^2 :=
by
  sorry

end pythagorean_theorem_l4027_402763


namespace max_min_xy_constraint_l4027_402755

theorem max_min_xy_constraint (x y : ℝ) : 
  x^2 + x*y + y^2 ≤ 1 → 
  (∃ (max min : ℝ), 
    (∀ z, x - y + 2*x*y ≤ z → z ≤ max) ∧ 
    (∀ w, min ≤ w → w ≤ x - y + 2*x*y) ∧
    max = 25/24 ∧ min = -4) := by
  sorry

end max_min_xy_constraint_l4027_402755


namespace ratio_difference_theorem_l4027_402738

theorem ratio_difference_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / b = 2 / 3 → (a + 4) / (b + 4) = 5 / 7 → b - a = 8 := by
  sorry

end ratio_difference_theorem_l4027_402738


namespace complex_difference_modulus_l4027_402722

theorem complex_difference_modulus (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 1)
  (h2 : Complex.abs z₂ = 1)
  (h3 : Complex.abs (z₁ + z₂) = Real.sqrt 3) :
  Complex.abs (z₁ - z₂) = 1 := by
  sorry

end complex_difference_modulus_l4027_402722


namespace cube_sum_gt_mixed_product_l4027_402719

theorem cube_sum_gt_mixed_product (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : 
  a^3 + b^3 > a^2 * b + a * b^2 := by
  sorry

end cube_sum_gt_mixed_product_l4027_402719


namespace sector_arc_length_l4027_402754

/-- Given a sector with central angle π/3 and radius 3, its arc length is π. -/
theorem sector_arc_length (α : Real) (r : Real) (l : Real) : 
  α = π / 3 → r = 3 → l = r * α → l = π := by
  sorry

end sector_arc_length_l4027_402754


namespace impossible_odd_black_cells_impossible_one_black_cell_l4027_402795

/-- Represents a chessboard --/
structure Chessboard where
  black_cells : ℕ

/-- Represents the operation of repainting a row or column --/
def repaint (board : Chessboard) : Chessboard :=
  { black_cells := board.black_cells + (8 - 2 * (board.black_cells % 8)) }

/-- Theorem stating that it's impossible to achieve an odd number of black cells --/
theorem impossible_odd_black_cells (initial_board : Chessboard) 
  (h : Even initial_board.black_cells) :
  ∀ (final_board : Chessboard), 
  (∃ (n : ℕ), final_board = (n.iterate repaint initial_board)) → 
  Even final_board.black_cells :=
sorry

/-- Corollary: It's impossible to achieve exactly one black cell --/
theorem impossible_one_black_cell (initial_board : Chessboard) 
  (h : Even initial_board.black_cells) :
  ¬∃ (final_board : Chessboard), 
  (∃ (n : ℕ), final_board = (n.iterate repaint initial_board)) ∧ 
  final_board.black_cells = 1 :=
sorry

end impossible_odd_black_cells_impossible_one_black_cell_l4027_402795


namespace compound_interest_principal_l4027_402752

/-- Given a sum of 8820 after 2 years with an interest rate of 5% per annum compounded yearly,
    prove that the initial principal amount was 8000. -/
theorem compound_interest_principal (sum : ℝ) (years : ℕ) (rate : ℝ) (principal : ℝ) 
    (h1 : sum = 8820)
    (h2 : years = 2)
    (h3 : rate = 0.05)
    (h4 : sum = principal * (1 + rate) ^ years) :
  principal = 8000 := by
  sorry

end compound_interest_principal_l4027_402752


namespace periodic_function_value_l4027_402715

theorem periodic_function_value (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f (x + 4) = f x) 
  (h2 : f 0.5 = 9) : 
  f 8.5 = 9 := by sorry

end periodic_function_value_l4027_402715


namespace rectangle_perimeters_l4027_402761

def is_valid_perimeter (p : ℕ) : Prop :=
  ∃ (x y : ℕ), 
    (x > 0 ∧ y > 0) ∧
    (3 * (2 * (x + y)) = 10) ∧
    (p = 2 * (x + y) ∨ p = 2 * (3 * x) ∨ p = 2 * (3 * y))

theorem rectangle_perimeters : 
  {p : ℕ | is_valid_perimeter p} = {14, 16, 18, 22, 26} :=
by sorry

end rectangle_perimeters_l4027_402761


namespace point_b_value_l4027_402778

/-- Given a point A representing 3 on the number line, moving 3 units from A to reach point B 
    results in B representing either 0 or 6. -/
theorem point_b_value (A B : ℝ) : 
  A = 3 → (B - A = 3 ∨ A - B = 3) → (B = 0 ∨ B = 6) := by
  sorry

end point_b_value_l4027_402778


namespace long_sleeve_shirts_count_l4027_402767

theorem long_sleeve_shirts_count (total_shirts short_sleeve_shirts : ℕ) 
  (h1 : total_shirts = 9)
  (h2 : short_sleeve_shirts = 4) :
  total_shirts - short_sleeve_shirts = 5 := by
sorry

end long_sleeve_shirts_count_l4027_402767


namespace prob_red_given_red_half_l4027_402735

/-- A bag with red and yellow balls -/
structure Bag where
  total : ℕ
  red : ℕ
  yellow : ℕ
  h_total : total = red + yellow

/-- The probability of drawing a red ball in the second draw given a red ball in the first draw -/
def prob_red_given_red (b : Bag) : ℚ :=
  (b.red - 1) / (b.total - 1)

/-- The theorem stating the probability is 1/2 for the given bag -/
theorem prob_red_given_red_half (b : Bag) 
  (h_total : b.total = 5)
  (h_red : b.red = 3)
  (h_yellow : b.yellow = 2) : 
  prob_red_given_red b = 1/2 := by
sorry

end prob_red_given_red_half_l4027_402735


namespace down_jacket_price_reduction_l4027_402731

/-- Represents the price reduction problem for down jackets --/
theorem down_jacket_price_reduction
  (initial_sales : ℕ)
  (initial_profit_per_piece : ℕ)
  (sales_increase_per_yuan : ℕ)
  (target_daily_profit : ℕ)
  (h1 : initial_sales = 20)
  (h2 : initial_profit_per_piece = 40)
  (h3 : sales_increase_per_yuan = 2)
  (h4 : target_daily_profit = 1200) :
  ∃ (price_reduction : ℕ),
    (initial_profit_per_piece - price_reduction) *
    (initial_sales + sales_increase_per_yuan * price_reduction) = target_daily_profit ∧
    price_reduction = 20 :=
by sorry

end down_jacket_price_reduction_l4027_402731


namespace smallest_divisible_number_is_correct_l4027_402776

/-- The smallest six-digit number exactly divisible by 25, 35, 45, and 15 -/
def smallest_divisible_number : ℕ := 100800

/-- Predicate to check if a number is six digits -/
def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n < 1000000

theorem smallest_divisible_number_is_correct :
  is_six_digit smallest_divisible_number ∧
  smallest_divisible_number % 25 = 0 ∧
  smallest_divisible_number % 35 = 0 ∧
  smallest_divisible_number % 45 = 0 ∧
  smallest_divisible_number % 15 = 0 ∧
  ∀ n : ℕ, is_six_digit n →
    n % 25 = 0 → n % 35 = 0 → n % 45 = 0 → n % 15 = 0 →
    n ≥ smallest_divisible_number :=
by sorry

#eval smallest_divisible_number

end smallest_divisible_number_is_correct_l4027_402776


namespace stating_alice_probability_after_two_turns_l4027_402741

/-- The probability that Alice passes the ball to Bob -/
def alice_pass_prob : ℚ := 2/3

/-- The probability that Bob passes the ball to Alice -/
def bob_pass_prob : ℚ := 1/2

/-- The probability that Alice has the ball after two turns -/
def alice_has_ball_after_two_turns : ℚ := 4/9

/-- 
Theorem stating that given the game rules, the probability 
that Alice has the ball after two turns is 4/9 
-/
theorem alice_probability_after_two_turns : 
  alice_has_ball_after_two_turns = 
    (alice_pass_prob * bob_pass_prob) + ((1 - alice_pass_prob) * (1 - alice_pass_prob)) := by
  sorry

end stating_alice_probability_after_two_turns_l4027_402741


namespace jerry_action_figures_l4027_402794

/-- Calculates the total number of action figures on Jerry's shelf -/
def total_action_figures (initial_figures : ℕ) (figures_per_set : ℕ) (added_sets : ℕ) : ℕ :=
  initial_figures + figures_per_set * added_sets

/-- Theorem stating that Jerry's shelf has 18 action figures in total -/
theorem jerry_action_figures :
  total_action_figures 8 5 2 = 18 := by
sorry

end jerry_action_figures_l4027_402794


namespace distance_between_points_l4027_402789

theorem distance_between_points (x₁ x₂ y₁ y₂ : ℝ) :
  x₁^2 + y₁^2 = 29 →
  x₂^2 + y₂^2 = 29 →
  x₁ + y₁ = 11 →
  x₂ + y₂ = 11 →
  x₁ ≠ x₂ →
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = Real.sqrt 2 := by
  sorry

end distance_between_points_l4027_402789


namespace same_suit_in_rows_l4027_402736

/-- Represents a playing card suit -/
inductive Suit
| clubs
| diamonds
| hearts
| spades

/-- Represents a card in the grid -/
structure Card where
  suit : Suit
  rank : Nat

/-- Represents the 13 × 4 grid of cards -/
def CardGrid := Fin 13 → Fin 4 → Card

/-- Checks if two cards are adjacent -/
def adjacent (c1 c2 : Card) : Prop :=
  c1.suit = c2.suit ∨ c1.rank = c2.rank

/-- The condition that adjacent cards in the grid are of the same suit or rank -/
def adjacency_condition (grid : CardGrid) : Prop :=
  ∀ i j, (i.val < 12 → adjacent (grid i j) (grid (i + 1) j)) ∧
         (j.val < 3 → adjacent (grid i j) (grid i (j + 1)))

/-- The statement to be proved -/
theorem same_suit_in_rows (grid : CardGrid) 
  (h : adjacency_condition grid) : 
  ∀ j, ∀ i1 i2, (grid i1 j).suit = (grid i2 j).suit :=
sorry

end same_suit_in_rows_l4027_402736


namespace finite_solutions_factorial_difference_l4027_402709

theorem finite_solutions_factorial_difference (u : ℕ+) :
  ∃ (S : Finset (ℕ × ℕ × ℕ)), ∀ (n a b : ℕ), 
    n! = u^a - u^b → (n, a, b) ∈ S :=
sorry

end finite_solutions_factorial_difference_l4027_402709


namespace adam_has_more_apples_l4027_402788

/-- The number of apples Adam has -/
def adam_apples : ℕ := 10

/-- The number of apples Jackie has -/
def jackie_apples : ℕ := 2

/-- The difference in apples between Adam and Jackie -/
def apple_difference : ℕ := adam_apples - jackie_apples

theorem adam_has_more_apples : apple_difference = 8 := by
  sorry

end adam_has_more_apples_l4027_402788


namespace packet_B_height_l4027_402727

/-- Growth rate of Packet A sunflowers -/
def R_A (x y : ℝ) : ℝ := 2 * x + y

/-- Growth rate of Packet B sunflowers -/
def R_B (x y : ℝ) : ℝ := 3 * x - y

/-- Theorem stating the height of Packet B sunflowers on day 10 -/
theorem packet_B_height (h_A : ℝ) (h_B : ℝ) :
  R_A 10 6 = 26 →
  R_B 10 6 = 24 →
  h_A = 192 →
  h_A = h_B + 0.2 * h_B →
  h_B = 160 := by
  sorry

#check packet_B_height

end packet_B_height_l4027_402727


namespace range_of_a_for_increasing_f_l4027_402748

-- Define the function f
noncomputable def f (a : ℝ) : ℝ → ℝ := 
  λ x => if x ≤ 1 then (4 - a) * x else a^x

-- State the theorem
theorem range_of_a_for_increasing_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) → 
  (a ∈ Set.Icc 2 4 ∧ a ≠ 4) := by sorry

end range_of_a_for_increasing_f_l4027_402748


namespace rectangle_perimeter_l4027_402766

/-- A rectangle with given diagonal and area has a specific perimeter -/
theorem rectangle_perimeter (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  a^2 + b^2 = 25^2 → a * b = 168 → 2 * (a + b) = 62 := by
  sorry

#check rectangle_perimeter

end rectangle_perimeter_l4027_402766


namespace jenny_distance_difference_l4027_402710

theorem jenny_distance_difference (run_distance walk_distance : ℝ) 
  (h1 : run_distance = 0.6)
  (h2 : walk_distance = 0.4) : 
  run_distance - walk_distance = 0.2 := by sorry

end jenny_distance_difference_l4027_402710


namespace stripe_area_on_cylindrical_silo_l4027_402765

/-- The area of a stripe wrapping around a cylindrical silo -/
theorem stripe_area_on_cylindrical_silo 
  (diameter : ℝ) 
  (stripe_width : ℝ) 
  (revolutions : ℕ) 
  (h_diameter : diameter = 40) 
  (h_stripe_width : stripe_width = 4) 
  (h_revolutions : revolutions = 3) : 
  stripe_width * revolutions * π * diameter = 480 * π := by
  sorry

end stripe_area_on_cylindrical_silo_l4027_402765


namespace complementary_angles_problem_l4027_402781

theorem complementary_angles_problem (C D : Real) : 
  C + D = 90 →  -- Angles C and D are complementary
  C = 3 * D →   -- The measure of angle C is 3 times angle D
  C = 67.5 :=   -- The measure of angle C is 67.5°
by
  sorry

end complementary_angles_problem_l4027_402781


namespace toys_storage_time_l4027_402785

/-- The time required to put all toys in the box -/
def time_to_store_toys (total_toys : ℕ) (net_gain_per_interval : ℕ) (interval_seconds : ℕ) : ℚ :=
  (total_toys : ℚ) / (net_gain_per_interval : ℚ) * (interval_seconds : ℚ) / 60

/-- Theorem stating that it takes 15 minutes to store all toys -/
theorem toys_storage_time :
  time_to_store_toys 30 1 30 = 15 := by
  sorry

#eval time_to_store_toys 30 1 30

end toys_storage_time_l4027_402785


namespace folded_rectangle_long_side_l4027_402784

/-- A rectangular sheet of paper with a special folding property -/
structure FoldedRectangle where
  short_side : ℝ
  long_side : ℝ
  is_folded_to_midpoint : Bool
  triangles_congruent : Bool

/-- The folded rectangle satisfies the problem conditions -/
def satisfies_conditions (r : FoldedRectangle) : Prop :=
  r.short_side = 8 ∧ r.is_folded_to_midpoint ∧ r.triangles_congruent

/-- The theorem stating that under the given conditions, the long side must be 12 units -/
theorem folded_rectangle_long_side
  (r : FoldedRectangle)
  (h : satisfies_conditions r) :
  r.long_side = 12 :=
by
  sorry

end folded_rectangle_long_side_l4027_402784


namespace polynomial_property_l4027_402747

-- Define the polynomial Q(x)
def Q (p q d : ℝ) (x : ℝ) : ℝ := x^3 + p*x^2 + q*x + d

-- State the theorem
theorem polynomial_property (p q d : ℝ) :
  -- The mean of zeros equals the product of zeros taken two at a time
  (-p/3 = q) →
  -- The mean of zeros equals the sum of coefficients
  (-p/3 = 1 + p + q + d) →
  -- The y-intercept is 5
  (Q p q d 0 = 5) →
  -- Then q = 2
  q = 2 := by sorry

end polynomial_property_l4027_402747


namespace gloria_cabin_theorem_l4027_402733

/-- Represents the problem of calculating Gloria's remaining money after buying a cabin --/
def gloria_cabin_problem (cabin_price cash_on_hand cypress_count pine_count maple_count cypress_price pine_price maple_price : ℕ) : Prop :=
  let total_from_trees := cypress_count * cypress_price + pine_count * pine_price + maple_count * maple_price
  let total_amount := total_from_trees + cash_on_hand
  let money_left := total_amount - cabin_price
  money_left = 350

/-- Theorem stating that Gloria will have $350 left after buying the cabin --/
theorem gloria_cabin_theorem : gloria_cabin_problem 129000 150 20 600 24 100 200 300 := by
  sorry

end gloria_cabin_theorem_l4027_402733


namespace andrey_solved_half_l4027_402774

theorem andrey_solved_half (N : ℕ) (x : ℕ) : 
  (N - x - (N - x) / 3 = N / 3) → 
  (x : ℚ) / N = 1 / 2 := by
  sorry

end andrey_solved_half_l4027_402774


namespace gcd_6724_13104_l4027_402779

theorem gcd_6724_13104 : Nat.gcd 6724 13104 = 8 := by
  sorry

end gcd_6724_13104_l4027_402779


namespace faulty_clock_correct_time_fraction_l4027_402753

/-- Represents a faulty digital clock that displays '5' instead of '2' over a 24-hour period -/
structure FaultyClock where
  /-- The number of hours in a day -/
  hours_per_day : ℕ
  /-- The number of minutes in an hour -/
  minutes_per_hour : ℕ
  /-- The number of hours affected by the fault -/
  faulty_hours : ℕ
  /-- The number of minutes per hour affected by the fault -/
  faulty_minutes : ℕ

/-- The fraction of the day a faulty clock displays the correct time -/
def correct_time_fraction (c : FaultyClock) : ℚ :=
  ((c.hours_per_day - c.faulty_hours) / c.hours_per_day) *
  ((c.minutes_per_hour - c.faulty_minutes) / c.minutes_per_hour)

/-- Theorem stating that the fraction of the day the faulty clock displays the correct time is 9/16 -/
theorem faulty_clock_correct_time_fraction :
  ∃ (c : FaultyClock), c.hours_per_day = 24 ∧ c.minutes_per_hour = 60 ∧
  c.faulty_hours = 6 ∧ c.faulty_minutes = 15 ∧
  correct_time_fraction c = 9 / 16 :=
by
  sorry

end faulty_clock_correct_time_fraction_l4027_402753


namespace exponential_growth_dominance_l4027_402786

theorem exponential_growth_dominance (n : ℕ) (h : n ≥ 10) : 2^n ≥ n^3 := by
  sorry

end exponential_growth_dominance_l4027_402786


namespace data_ratio_l4027_402760

theorem data_ratio (a b c : ℝ) 
  (h1 : a = b - c) 
  (h2 : a = 12) 
  (h3 : a + b + c = 96) : 
  b / a = 4 := by
sorry

end data_ratio_l4027_402760


namespace tetrahedron_surface_area_l4027_402762

/-- Tetrahedron with specific properties -/
structure Tetrahedron where
  -- Base is a square with side length 3
  baseSideLength : ℝ := 3
  -- PD length is 4
  pdLength : ℝ := 4
  -- Lateral faces PAD and PCD are perpendicular to the base
  lateralFacesPerpendicular : Prop

/-- Calculate the surface area of the tetrahedron -/
def surfaceArea (t : Tetrahedron) : ℝ :=
  -- We don't implement the actual calculation here
  sorry

/-- Theorem stating the surface area of the tetrahedron -/
theorem tetrahedron_surface_area (t : Tetrahedron) : 
  surfaceArea t = 9 + 6 * Real.sqrt 7 := by
  sorry

end tetrahedron_surface_area_l4027_402762


namespace circle_area_tripled_radius_l4027_402706

theorem circle_area_tripled_radius (r : ℝ) (hr : r > 0) :
  let A := π * r^2
  let A' := π * (3*r)^2
  A' = 9 * A ∧ A' ≠ 3 * A :=
by
  sorry

end circle_area_tripled_radius_l4027_402706


namespace cone_volume_from_cylinder_l4027_402716

/-- Given a cylinder with volume 72π cm³, prove that a cone with double the height 
    and the same radius as the cylinder has a volume of 48π cm³. -/
theorem cone_volume_from_cylinder (r h : ℝ) : 
  (π * r^2 * h = 72 * π) → 
  (1/3 : ℝ) * π * r^2 * (2 * h) = 48 * π := by
sorry


end cone_volume_from_cylinder_l4027_402716


namespace quadratic_point_value_l4027_402744

/-- If the point (1,a) lies on the graph of y = 2x^2, then a = 2 -/
theorem quadratic_point_value (a : ℝ) : (2 : ℝ) * (1 : ℝ)^2 = a → a = 2 := by
  sorry

end quadratic_point_value_l4027_402744


namespace students_taking_neither_math_nor_chemistry_l4027_402799

theorem students_taking_neither_math_nor_chemistry :
  let total_students : ℕ := 150
  let math_students : ℕ := 80
  let chemistry_students : ℕ := 60
  let both_subjects : ℕ := 15
  let neither_subject : ℕ := total_students - (math_students + chemistry_students - both_subjects)
  neither_subject = 25 := by
  sorry

end students_taking_neither_math_nor_chemistry_l4027_402799


namespace remaining_volume_of_cube_with_hole_l4027_402725

/-- The remaining volume of a cube with a square hole cut through its center -/
theorem remaining_volume_of_cube_with_hole (cube_side : ℝ) (hole_side : ℝ) : 
  cube_side = 8 → hole_side = 4 → 
  cube_side ^ 3 - (hole_side ^ 2 * cube_side) = 384 := by
  sorry

end remaining_volume_of_cube_with_hole_l4027_402725


namespace inequality_transformation_l4027_402743

theorem inequality_transformation (a : ℝ) : 
  (∀ x : ℝ, a * x > 2 ↔ x < 2 / a) → a < 0 := by
  sorry

end inequality_transformation_l4027_402743


namespace sphere_dimensions_l4027_402759

-- Define the hole dimensions
def hole_diameter : ℝ := 12
def hole_depth : ℝ := 2

-- Define the sphere
def sphere_radius : ℝ := 10

-- Theorem statement
theorem sphere_dimensions (r : ℝ) (h : r = sphere_radius) :
  -- The radius satisfies the Pythagorean theorem for the right triangle formed
  (r - hole_depth) ^ 2 + (hole_diameter / 2) ^ 2 = r ^ 2 ∧
  -- The surface area of the sphere is 400π
  4 * Real.pi * r ^ 2 = 400 * Real.pi := by
  sorry

end sphere_dimensions_l4027_402759


namespace remaining_fuel_after_three_hours_l4027_402783

/-- Represents the remaining fuel in a car's tank after driving for a certain time -/
def remaining_fuel (initial_fuel : ℝ) (consumption_rate : ℝ) (hours : ℝ) : ℝ :=
  initial_fuel - consumption_rate * hours

/-- Theorem stating that the remaining fuel after 3 hours matches the expression a-3b -/
theorem remaining_fuel_after_three_hours (a b : ℝ) :
  remaining_fuel a b 3 = a - 3 * b := by
  sorry

end remaining_fuel_after_three_hours_l4027_402783


namespace nine_times_two_sevenths_squared_l4027_402796

theorem nine_times_two_sevenths_squared :
  9 * (2 / 7)^2 = 36 / 49 := by sorry

end nine_times_two_sevenths_squared_l4027_402796


namespace counseling_rooms_count_l4027_402777

theorem counseling_rooms_count :
  ∃ (x : ℕ) (total_students : ℕ),
    (total_students = 20 * x + 32) ∧
    (total_students = 24 * (x - 1)) ∧
    (x = 14) := by
  sorry

end counseling_rooms_count_l4027_402777


namespace locus_of_midpoints_l4027_402708

/-- The locus of midpoints theorem -/
theorem locus_of_midpoints 
  (P : ℝ × ℝ) 
  (h_P : P = (4, -2)) 
  (Q : ℝ × ℝ) 
  (h_Q : Q.1^2 + Q.2^2 = 4) 
  (M : ℝ × ℝ) 
  (h_M : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) :
  (M.1 - 2)^2 + (M.2 + 1)^2 = 1 := by
  sorry

end locus_of_midpoints_l4027_402708


namespace solution_to_system_l4027_402757

theorem solution_to_system : ∃ (x y : ℝ), 
  x^2 * y - x * y^2 - 3*x + 3*y + 1 = 0 ∧
  x^3 * y - x * y^3 - 3*x^2 + 3*y^2 + 3 = 0 ∧
  x = 2 ∧ y = 1 := by
  sorry

end solution_to_system_l4027_402757


namespace equation_solution_l4027_402700

theorem equation_solution : ∃ x : ℝ, 
  x * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2800.0000000000005 ∧ 
  abs (x - 1.4) < 0.00000000000001 := by
  sorry

end equation_solution_l4027_402700


namespace company_median_salary_l4027_402742

/-- Represents a job position with its title, number of employees, and salary --/
structure Position where
  title : String
  count : Nat
  salary : Nat

/-- The list of positions in the company --/
def company_positions : List Position := [
  { title := "President", count := 1, salary := 140000 },
  { title := "Vice-President", count := 10, salary := 100000 },
  { title := "Director", count := 15, salary := 80000 },
  { title := "Manager", count := 5, salary := 55000 },
  { title := "Associate Director", count := 9, salary := 52000 },
  { title := "Administrative Specialist", count := 35, salary := 25000 }
]

/-- The total number of employees in the company --/
def total_employees : Nat := 75

/-- Calculates the median salary of the company --/
def median_salary (positions : List Position) (total : Nat) : Nat :=
  sorry

/-- Theorem stating that the median salary of the company is $52,000 --/
theorem company_median_salary :
  median_salary company_positions total_employees = 52000 := by
  sorry

end company_median_salary_l4027_402742


namespace divisible_by_thirty_l4027_402723

theorem divisible_by_thirty (n : ℕ) (h : n > 0) : ∃ k : ℤ, n^19 - n^7 = 30 * k := by
  sorry

end divisible_by_thirty_l4027_402723


namespace abc_product_l4027_402739

theorem abc_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 30 * Real.rpow 4 (1/3))
  (hac : a * c = 40 * Real.rpow 4 (1/3))
  (hbc : b * c = 24 * Real.rpow 4 (1/3)) :
  a * b * c = 120 := by
sorry

end abc_product_l4027_402739


namespace binary_10011_equals_19_l4027_402704

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_10011_equals_19 :
  binary_to_decimal [true, true, false, false, true] = 19 := by
  sorry

end binary_10011_equals_19_l4027_402704


namespace fraction_comparison_and_inequality_l4027_402790

theorem fraction_comparison_and_inequality : 
  (37 : ℚ) / 29 < 41 / 31 ∧ 
  41 / 31 < 31 / 23 ∧ 
  37 / 29 ≠ 4 / 3 ∧ 
  41 / 31 ≠ 4 / 3 ∧ 
  31 / 23 ≠ 4 / 3 :=
by sorry

end fraction_comparison_and_inequality_l4027_402790


namespace square_perimeter_l4027_402720

theorem square_perimeter (s : ℝ) (h1 : s > 0) : 
  let rectangle_perimeter := 2 * (s + s / 5)
  rectangle_perimeter = 48 → 4 * s = 80 := by
sorry

end square_perimeter_l4027_402720


namespace distance_difference_l4027_402729

/-- The distance biked by Bjorn after six hours -/
def bjorn_distance : ℕ := 75

/-- The distance biked by Alberto after six hours -/
def alberto_distance : ℕ := 105

/-- Alberto bikes faster than Bjorn -/
axiom alberto_faster : alberto_distance > bjorn_distance

/-- The difference in distance biked between Alberto and Bjorn after six hours is 30 miles -/
theorem distance_difference : alberto_distance - bjorn_distance = 30 := by
  sorry

end distance_difference_l4027_402729


namespace expand_and_simplify_l4027_402721

theorem expand_and_simplify (x y : ℝ) : 12 * (3 * x + 4 * y - 2) = 36 * x + 48 * y - 24 := by
  sorry

end expand_and_simplify_l4027_402721


namespace twenty_four_divides_Q_largest_divisor_of_Q_l4027_402728

/-- The product of three consecutive positive even integers -/
def Q (n : ℕ) : ℕ := (2*n) * (2*n + 2) * (2*n + 4)

/-- 24 divides Q for all positive n -/
theorem twenty_four_divides_Q (n : ℕ) (h : n > 0) : 24 ∣ Q n := by sorry

/-- 24 is the largest integer that divides Q for all positive n -/
theorem largest_divisor_of_Q :
  ∀ d : ℕ, (∀ n : ℕ, n > 0 → d ∣ Q n) → d ≤ 24 := by sorry

end twenty_four_divides_Q_largest_divisor_of_Q_l4027_402728
