import Mathlib

namespace election_win_margin_l3707_370716

theorem election_win_margin (total_votes : ℕ) (winner_votes : ℕ) (winner_percentage : ℚ) 
  (h1 : winner_percentage = 62 / 100)
  (h2 : winner_votes = 806)
  (h3 : winner_votes = (winner_percentage * total_votes).floor) :
  winner_votes - (total_votes - winner_votes) = 312 := by
  sorry

end election_win_margin_l3707_370716


namespace solve_equation_l3707_370726

theorem solve_equation : ∃ x : ℚ, (40 / 60 : ℚ) = Real.sqrt (x / 60) → x = 80 / 3 := by
  sorry

end solve_equation_l3707_370726


namespace minimum_books_in_library_l3707_370704

theorem minimum_books_in_library (physics chemistry biology : ℕ) : 
  physics + chemistry + biology > 0 →
  3 * chemistry = 2 * physics →
  4 * biology = 3 * chemistry →
  ∃ (k : ℕ), k * (physics + chemistry + biology) = 3003 →
  3003 ≤ physics + chemistry + biology :=
by sorry

end minimum_books_in_library_l3707_370704


namespace y_derivative_l3707_370711

-- Define the function y
noncomputable def y (x : ℝ) : ℝ := Real.log (1 / Real.sqrt (1 + x^2))

-- State the theorem
theorem y_derivative (x : ℝ) : 
  deriv y x = -x / (1 + x^2) := by sorry

end y_derivative_l3707_370711


namespace bread_price_is_two_l3707_370707

/-- The price of a can of spam in dollars -/
def spam_price : ℚ := 3

/-- The price of a jar of peanut butter in dollars -/
def peanut_butter_price : ℚ := 5

/-- The number of cans of spam bought -/
def spam_quantity : ℕ := 12

/-- The number of jars of peanut butter bought -/
def peanut_butter_quantity : ℕ := 3

/-- The number of loaves of bread bought -/
def bread_quantity : ℕ := 4

/-- The total amount paid in dollars -/
def total_paid : ℚ := 59

/-- The price of a loaf of bread in dollars -/
def bread_price : ℚ := 2

theorem bread_price_is_two :
  spam_price * spam_quantity +
  peanut_butter_price * peanut_butter_quantity +
  bread_price * bread_quantity = total_paid := by
  sorry

end bread_price_is_two_l3707_370707


namespace president_secretary_choice_count_l3707_370749

/-- Represents the number of ways to choose a president and secretary of the same gender -/
def choose_president_and_secretary (total_members : ℕ) (boys : ℕ) (girls : ℕ) : ℕ :=
  (boys * (boys - 1)) + (girls * (girls - 1))

/-- Theorem: Given a society of 25 members (15 boys and 10 girls), 
    the number of ways to choose a president and a secretary of the same gender, 
    where no one can hold both positions, is equal to 300. -/
theorem president_secretary_choice_count :
  choose_president_and_secretary 25 15 10 = 300 := by
  sorry

end president_secretary_choice_count_l3707_370749


namespace complex_numbers_on_circle_l3707_370714

/-- Given non-zero complex numbers a₁, a₂, a₃, a₄, a₅ satisfying certain conditions,
    prove that they lie on the same circle in the complex plane. -/
theorem complex_numbers_on_circle (a₁ a₂ a₃ a₄ a₅ : ℂ) (S : ℝ) 
    (h_nonzero : a₁ ≠ 0 ∧ a₂ ≠ 0 ∧ a₃ ≠ 0 ∧ a₄ ≠ 0 ∧ a₅ ≠ 0)
    (h_ratio : a₂ / a₁ = a₃ / a₂ ∧ a₃ / a₂ = a₄ / a₃ ∧ a₄ / a₃ = a₅ / a₄)
    (h_sum : a₁ + a₂ + a₃ + a₄ + a₅ = 4 * (1 / a₁ + 1 / a₂ + 1 / a₃ + 1 / a₄ + 1 / a₅))
    (h_sum_real : a₁ + a₂ + a₃ + a₄ + a₅ = S)
    (h_S_bound : abs S ≤ 2) :
  ∃ r : ℝ, r > 0 ∧ Complex.abs a₁ = r ∧ Complex.abs a₂ = r ∧ 
    Complex.abs a₃ = r ∧ Complex.abs a₄ = r ∧ Complex.abs a₅ = r :=
by sorry

end complex_numbers_on_circle_l3707_370714


namespace solution_set_when_a_is_4_range_of_a_for_inequality_l3707_370725

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 + 2
def g (a x : ℝ) : ℝ := |x - a| - |x - 1|

-- Theorem for part 1
theorem solution_set_when_a_is_4 :
  {x : ℝ | f x > g 4 x} = {x : ℝ | x > 1 ∨ x ≤ -1} := by sorry

-- Theorem for part 2
theorem range_of_a_for_inequality :
  (∀ x₁ x₂ : ℝ, f x₁ ≥ g a x₂) ↔ -1 ≤ a ∧ a ≤ 3 := by sorry

end solution_set_when_a_is_4_range_of_a_for_inequality_l3707_370725


namespace right_triangle_sets_l3707_370740

theorem right_triangle_sets :
  -- Set A
  (5^2 + 12^2 = 13^2) ∧
  -- Set B
  ((Real.sqrt 2)^2 + (Real.sqrt 3)^2 = (Real.sqrt 5)^2) ∧
  -- Set C
  (3^2 + (Real.sqrt 7)^2 = 4^2) ∧
  -- Set D
  (2^2 + 3^2 ≠ 4^2) := by
  sorry

end right_triangle_sets_l3707_370740


namespace club_members_after_four_years_l3707_370751

/-- Calculates the number of people in the club after a given number of years -/
def club_members (initial_regular_members : ℕ) (years : ℕ) : ℕ :=
  initial_regular_members * (2 ^ years)

/-- Theorem stating the number of people in the club after 4 years -/
theorem club_members_after_four_years :
  let initial_total := 9
  let initial_board_members := 3
  let initial_regular_members := initial_total - initial_board_members
  club_members initial_regular_members 4 = 96 := by
  sorry

end club_members_after_four_years_l3707_370751


namespace grid_sum_bottom_corners_l3707_370742

/-- Represents a 3x3 grid where each cell contains a number -/
def Grid := Fin 3 → Fin 3 → Nat

/-- Checks if a given number appears exactly once in each row -/
def rowValid (g : Grid) (n : Nat) : Prop :=
  ∀ i : Fin 3, ∃! j : Fin 3, g i j = n

/-- Checks if a given number appears exactly once in each column -/
def colValid (g : Grid) (n : Nat) : Prop :=
  ∀ j : Fin 3, ∃! i : Fin 3, g i j = n

/-- Checks if the grid contains only the numbers 4, 5, and 6 -/
def gridContainsOnly456 (g : Grid) : Prop :=
  ∀ i j : Fin 3, g i j = 4 ∨ g i j = 5 ∨ g i j = 6

/-- The main theorem statement -/
theorem grid_sum_bottom_corners (g : Grid) :
  rowValid g 4 ∧ rowValid g 5 ∧ rowValid g 6 ∧
  colValid g 4 ∧ colValid g 5 ∧ colValid g 6 ∧
  gridContainsOnly456 g ∧
  g 0 0 = 5 ∧ g 1 1 = 4 →
  g 2 0 + g 2 2 = 10 := by
  sorry

end grid_sum_bottom_corners_l3707_370742


namespace log_base_32_integer_count_l3707_370757

theorem log_base_32_integer_count : 
  ∃! (n : ℕ), n > 0 ∧ (∃ (S : Finset ℕ), 
    (∀ b ∈ S, b > 0 ∧ ∃ (k : ℕ), k > 0 ∧ (↑b : ℝ) ^ k = 32) ∧
    S.card = n) :=
by sorry

end log_base_32_integer_count_l3707_370757


namespace number_division_result_l3707_370719

theorem number_division_result (x : ℝ) (h : 5 * x = 100) : x / 10 = 2 := by
  sorry

end number_division_result_l3707_370719


namespace quadrilateral_rhombus_l3707_370767

-- Define the points
variable (A B C D P Q R S : ℝ × ℝ)

-- Define the properties of the quadrilaterals
def is_convex_quadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

def are_external_similar_isosceles_triangles (A B C D P Q R S : ℝ × ℝ) : Prop := sorry

def is_rectangle (P Q R S : ℝ × ℝ) : Prop := sorry

def sides_not_equal (P Q R S : ℝ × ℝ) : Prop := sorry

def is_rhombus (A B C D : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem quadrilateral_rhombus 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : are_external_similar_isosceles_triangles A B C D P Q R S)
  (h3 : is_rectangle P Q R S)
  (h4 : sides_not_equal P Q R S) :
  is_rhombus A B C D :=
by sorry

end quadrilateral_rhombus_l3707_370767


namespace empty_boxes_count_l3707_370705

/-- The number of boxes containing neither pens, pencils, nor markers -/
def empty_boxes (total boxes_with_pencils boxes_with_pens boxes_with_both_pens_pencils
                 boxes_with_markers boxes_with_pencils_markers : ℕ) : ℕ :=
  total - (boxes_with_pencils + boxes_with_pens - boxes_with_both_pens_pencils + 
           boxes_with_markers - boxes_with_pencils_markers)

theorem empty_boxes_count :
  ∀ (total boxes_with_pencils boxes_with_pens boxes_with_both_pens_pencils
     boxes_with_markers boxes_with_pencils_markers : ℕ),
  total = 15 →
  boxes_with_pencils = 9 →
  boxes_with_pens = 5 →
  boxes_with_both_pens_pencils = 3 →
  boxes_with_markers = 4 →
  boxes_with_pencils_markers = 2 →
  boxes_with_markers ≤ boxes_with_pencils →
  boxes_with_both_pens_pencils ≤ min boxes_with_pencils boxes_with_pens →
  boxes_with_pencils_markers ≤ min boxes_with_pencils boxes_with_markers →
  empty_boxes total boxes_with_pencils boxes_with_pens boxes_with_both_pens_pencils
              boxes_with_markers boxes_with_pencils_markers = 2 :=
by
  sorry

end empty_boxes_count_l3707_370705


namespace rose_pollen_diameter_scientific_notation_l3707_370779

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The diameter of the rose pollen in meters -/
def rose_pollen_diameter : ℝ := 0.0000028

/-- The scientific notation representation of the rose pollen diameter -/
def rose_pollen_scientific : ScientificNotation :=
  { coefficient := 2.8
  , exponent := -6
  , is_valid := by sorry }

/-- Theorem stating that the rose pollen diameter is correctly expressed in scientific notation -/
theorem rose_pollen_diameter_scientific_notation :
  rose_pollen_diameter = rose_pollen_scientific.coefficient * (10 : ℝ) ^ rose_pollen_scientific.exponent :=
by sorry

end rose_pollen_diameter_scientific_notation_l3707_370779


namespace angle_A_value_min_value_expression_l3707_370752

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  R : Real -- circumradius

-- Define the given condition
def triangle_condition (t : Triangle) : Prop :=
  2 * t.R - t.a = (t.a * (t.b^2 + t.c^2 - t.a^2)) / (t.a^2 + t.c^2 - t.b^2)

-- Theorem 1
theorem angle_A_value (t : Triangle) 
  (h1 : triangle_condition t) 
  (h2 : t.A ≠ π/2) 
  (h3 : t.B = π/6) : 
  t.A = π/6 := by sorry

-- Theorem 2
theorem min_value_expression (t : Triangle) 
  (h1 : triangle_condition t) 
  (h2 : t.A ≠ π/2) : 
  ∃ (min : Real), (∀ (t' : Triangle), triangle_condition t' → t'.A ≠ π/2 → 
    (2 * t'.a^2 - t'.c^2) / t'.b^2 ≥ min) ∧ 
  min = 4 * Real.sqrt 2 - 7 := by sorry

end angle_A_value_min_value_expression_l3707_370752


namespace smallest_multiplier_for_perfect_square_l3707_370713

def y : ℕ := 2^3^4^5^6^7^8^9

theorem smallest_multiplier_for_perfect_square (k : ℕ) : 
  k > 0 ∧ 
  ∃ m : ℕ, k * y = m^2 ∧ 
  ∀ l : ℕ, l > 0 ∧ l < k → ¬∃ n : ℕ, l * y = n^2 
  ↔ 
  k = 10 := by sorry

end smallest_multiplier_for_perfect_square_l3707_370713


namespace cos_210_degrees_l3707_370796

theorem cos_210_degrees : Real.cos (210 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_degrees_l3707_370796


namespace jelly_beans_remaining_l3707_370729

/-- The number of jelly beans remaining in a container after distribution -/
def remaining_jelly_beans (total : ℕ) (num_people : ℕ) (first_group : ℕ) (second_group : ℕ) (second_group_beans : ℕ) : ℕ :=
  total - (first_group * (2 * second_group_beans) + second_group * second_group_beans)

/-- Proof that 1600 jelly beans remain in the container -/
theorem jelly_beans_remaining :
  remaining_jelly_beans 8000 10 6 4 400 = 1600 := by
  sorry

#eval remaining_jelly_beans 8000 10 6 4 400

end jelly_beans_remaining_l3707_370729


namespace smallest_number_divisibility_l3707_370760

theorem smallest_number_divisibility : ∃! n : ℕ, 
  (∀ m : ℕ, m < n → ¬(((m + 7) % 8 = 0) ∧ ((m + 7) % 11 = 0) ∧ ((m + 7) % 24 = 0))) ∧
  ((n + 7) % 8 = 0) ∧ ((n + 7) % 11 = 0) ∧ ((n + 7) % 24 = 0) ∧
  n = 257 :=
by sorry

end smallest_number_divisibility_l3707_370760


namespace ball_probability_l3707_370703

theorem ball_probability (red green yellow total : ℕ) (p_green : ℚ) :
  red = 8 →
  green = 10 →
  total = red + green + yellow →
  p_green = 1 / 4 →
  p_green = green / total →
  yellow / total = 11 / 20 :=
by
  sorry

end ball_probability_l3707_370703


namespace prob_twice_daughters_is_37_256_l3707_370762

-- Define the number of children
def num_children : ℕ := 8

-- Define the probability of having a daughter (equal to the probability of having a son)
def p_daughter : ℚ := 1/2

-- Define the function to calculate the probability of having exactly k daughters out of n children
def prob_k_daughters (n k : ℕ) : ℚ :=
  (n.choose k) * (p_daughter ^ k) * ((1 - p_daughter) ^ (n - k))

-- Define the probability of having at least twice as many daughters as sons
def prob_twice_daughters : ℚ :=
  prob_k_daughters num_children num_children +
  prob_k_daughters num_children (num_children - 1) +
  prob_k_daughters num_children (num_children - 2)

-- Theorem statement
theorem prob_twice_daughters_is_37_256 : prob_twice_daughters = 37/256 := by
  sorry

end prob_twice_daughters_is_37_256_l3707_370762


namespace Q_greater_than_P_l3707_370731

/-- A number consisting of 2010 digits of 1 -/
def a : ℕ := 10^2010 - 1

/-- P defined as the product of 2010 digits of 8 and 2010 digits of 3 -/
def P : ℕ := (8 * a) * (3 * a)

/-- Q defined as the product of 2010 digits of 4 and (2009 digits of 6 followed by 7) -/
def Q : ℕ := (4 * a) * (6 * a + 1)

/-- Theorem stating that Q is greater than P -/
theorem Q_greater_than_P : Q > P := by
  sorry

end Q_greater_than_P_l3707_370731


namespace box_production_equations_l3707_370715

/-- Represents the number of iron sheets available -/
def total_sheets : ℕ := 40

/-- Represents the number of box bodies that can be made from one sheet -/
def bodies_per_sheet : ℕ := 15

/-- Represents the number of box bottoms that can be made from one sheet -/
def bottoms_per_sheet : ℕ := 20

/-- Represents the ratio of box bottoms to box bodies in a complete set -/
def bottoms_to_bodies_ratio : ℕ := 2

/-- Theorem stating that the given system of equations correctly represents the problem -/
theorem box_production_equations (x y : ℕ) : 
  (x + y = total_sheets ∧ 
   2 * bodies_per_sheet * x = bottoms_per_sheet * y) ↔ 
  (x + y = total_sheets ∧ 
   bottoms_to_bodies_ratio * (bodies_per_sheet * x) = bottoms_per_sheet * y) :=
sorry

end box_production_equations_l3707_370715


namespace f_is_quadratic_l3707_370746

/-- A quadratic equation in terms of x is of the form ax² + bx + c = 0, where a ≠ 0 --/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation (x-1)(x-2)=0 --/
def f (x : ℝ) : ℝ := (x - 1) * (x - 2)

/-- Theorem stating that f is a quadratic equation --/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end f_is_quadratic_l3707_370746


namespace school_journey_time_l3707_370774

/-- The time for a journey to school, given specific conditions about forgetting an item -/
theorem school_journey_time : ∃ (t : ℝ), 
  (t > 0) ∧ 
  (t - 6 > 0) ∧
  (∃ (x : ℝ), x > 0 ∧ x = t / 5) ∧
  ((9/5) * t = t + 2) ∧ 
  (t = 20) := by
  sorry

end school_journey_time_l3707_370774


namespace tan_angle_through_P_l3707_370782

/-- An angle in the coordinate plane -/
structure Angle :=
  (initial_side : Set (ℝ × ℝ))
  (terminal_side : Set (ℝ × ℝ))

/-- The tangent of an angle -/
def tan (α : Angle) : ℝ := sorry

/-- The non-negative half-axis of the x-axis -/
def non_negative_x_axis : Set (ℝ × ℝ) := sorry

/-- A point P(-2,1) in the coordinate plane -/
def point_P : ℝ × ℝ := (-2, 1)

/-- The line passing through the origin and point P -/
def line_through_origin_and_P : Set (ℝ × ℝ) := sorry

theorem tan_angle_through_P :
  ∀ α : Angle,
  α.initial_side = non_negative_x_axis →
  α.terminal_side = line_through_origin_and_P →
  tan α = -1/2 := by sorry

end tan_angle_through_P_l3707_370782


namespace perpendicular_vectors_and_angle_l3707_370755

theorem perpendicular_vectors_and_angle (θ φ : ℝ) : 
  (0 < θ) → (θ < π) →
  (π / 2 < φ) → (φ < π) →
  (2 * Real.cos θ + Real.sin θ = 0) →
  (Real.sin (θ - φ) = Real.sqrt 10 / 10) →
  (Real.tan θ = -2 ∧ Real.cos φ = -(Real.sqrt 2 / 10)) :=
by sorry

end perpendicular_vectors_and_angle_l3707_370755


namespace negation_of_all_divisible_by_five_are_even_l3707_370741

theorem negation_of_all_divisible_by_five_are_even :
  (¬ ∀ n : ℤ, 5 ∣ n → Even n) ↔ (∃ n : ℤ, 5 ∣ n ∧ ¬Even n) := by
  sorry

end negation_of_all_divisible_by_five_are_even_l3707_370741


namespace range_of_x_given_conditions_l3707_370722

def is_monotone_increasing_on_nonpositive (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y

def is_symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem range_of_x_given_conditions (f : ℝ → ℝ) 
  (h1 : is_monotone_increasing_on_nonpositive f)
  (h2 : is_symmetric_about_y_axis f)
  (h3 : ∀ x, f (x - 2) > f 2) :
  ∀ x, (0 < x ∧ x < 4) ↔ (f (x - 2) > f 2) :=
by sorry

end range_of_x_given_conditions_l3707_370722


namespace video_game_sales_earnings_l3707_370787

def total_games : ℕ := 15
def non_working_games : ℕ := 9
def price_per_game : ℕ := 5

theorem video_game_sales_earnings : 
  (total_games - non_working_games) * price_per_game = 30 := by
  sorry

end video_game_sales_earnings_l3707_370787


namespace fish_in_tank_l3707_370792

theorem fish_in_tank (total : ℕ) (blue : ℕ) (spotted : ℕ) : 
  (3 * blue = total) →  -- One third of the fish are blue
  (2 * spotted = blue) →  -- Half of the blue fish have spots
  (spotted = 10) →  -- There are 10 blue, spotted fish
  total = 60 := by
  sorry

end fish_in_tank_l3707_370792


namespace sum_a_plus_d_l3707_370770

theorem sum_a_plus_d (a b c d : ℝ) 
  (eq1 : a + b = 16) 
  (eq2 : b + c = 9) 
  (eq3 : c + d = 3) : 
  a + d = 13 := by
sorry

end sum_a_plus_d_l3707_370770


namespace franks_candy_bags_l3707_370720

/-- Given that Frank puts 11 pieces of candy in each bag and has 22 pieces of candy in total,
    prove that the number of bags Frank would have is equal to 2. -/
theorem franks_candy_bags (pieces_per_bag : ℕ) (total_pieces : ℕ) (h1 : pieces_per_bag = 11) (h2 : total_pieces = 22) :
  total_pieces / pieces_per_bag = 2 := by
  sorry

end franks_candy_bags_l3707_370720


namespace expression_value_l3707_370778

theorem expression_value (x : ℝ) (h : 5 * x^2 - x - 2 = 0) :
  (2*x + 1) * (2*x - 1) + x * (x - 1) = 1 := by
  sorry

end expression_value_l3707_370778


namespace nickels_count_l3707_370798

/-- Given 30 coins (nickels and dimes) with a total value of 240 cents, prove that the number of nickels is 12. -/
theorem nickels_count (n d : ℕ) : 
  n + d = 30 →  -- Total number of coins
  5 * n + 10 * d = 240 →  -- Total value in cents
  n = 12 :=  -- Number of nickels
by sorry

end nickels_count_l3707_370798


namespace quadratic_to_cubic_approximation_l3707_370765

/-- Given that x^2 - 6x + 1 can be approximated by a(x-h)^3 + k for some constants a and k,
    prove that h = 2. -/
theorem quadratic_to_cubic_approximation (x : ℝ) :
  ∃ (a k : ℝ), (∀ ε > 0, ∃ δ > 0, ∀ x, |x - 3| < δ → 
    |x^2 - 6*x + 1 - (a * (x - 2)^3 + k)| < ε) →
  2 = 2 :=
sorry

end quadratic_to_cubic_approximation_l3707_370765


namespace distance_between_complex_points_l3707_370728

theorem distance_between_complex_points :
  let z₁ : ℂ := 2 + 3*I
  let z₂ : ℂ := -2 + 2*I
  Complex.abs (z₁ - z₂) = Real.sqrt 17 := by
  sorry

end distance_between_complex_points_l3707_370728


namespace solution_to_congruence_l3707_370706

theorem solution_to_congruence (n : ℤ) : 
  0 ≤ n ∧ n < 103 ∧ (100 * n) % 103 = 34 % 103 → n = 52 := by
  sorry

end solution_to_congruence_l3707_370706


namespace tan_75_degrees_l3707_370743

theorem tan_75_degrees : Real.tan (75 * π / 180) = 2 + Real.sqrt 3 := by
  sorry

end tan_75_degrees_l3707_370743


namespace jennifer_apples_l3707_370708

/-- The number of apples Jennifer started with -/
def initial_apples : ℕ := sorry

/-- The number of apples Jennifer found -/
def found_apples : ℕ := 74

/-- The total number of apples Jennifer ended up with -/
def total_apples : ℕ := 81

/-- Theorem stating that the initial number of apples plus the found apples equals the total apples -/
theorem jennifer_apples : initial_apples + found_apples = total_apples := by sorry

end jennifer_apples_l3707_370708


namespace max_ab_value_l3707_370776

/-- Given a line ax + by - 6 = 0 (a > 0, b > 0) intercepted by the circle x^2 + y^2 - 2x - 4y = 0
    to form a chord of length 2√5, the maximum value of ab is 9/2 -/
theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), a * x + b * y - 6 = 0 ∧ 
                x^2 + y^2 - 2*x - 4*y = 0 ∧ 
                ∃ (x1 y1 x2 y2 : ℝ), 
                  a * x1 + b * y1 - 6 = 0 ∧ 
                  x1^2 + y1^2 - 2*x1 - 4*y1 = 0 ∧
                  a * x2 + b * y2 - 6 = 0 ∧ 
                  x2^2 + y2^2 - 2*x2 - 4*y2 = 0 ∧
                  (x2 - x1)^2 + (y2 - y1)^2 = 20) →
  a * b ≤ 9/2 :=
by sorry

end max_ab_value_l3707_370776


namespace six_grade_sequences_l3707_370773

/-- Represents the number of ways to assign n grades under the given conditions -/
def gradeSequences (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 3
  | n + 2 => 2 * gradeSequences (n + 1) + 2 * gradeSequences n

/-- The number of ways to assign 6 grades under the given conditions is 448 -/
theorem six_grade_sequences : gradeSequences 6 = 448 := by
  sorry

/-- Helper lemma: The recurrence relation holds for all n ≥ 2 -/
lemma recurrence_relation (n : ℕ) (h : n ≥ 2) :
  gradeSequences n = 2 * gradeSequences (n - 1) + 2 * gradeSequences (n - 2) := by
  sorry

end six_grade_sequences_l3707_370773


namespace cats_not_eating_l3707_370766

theorem cats_not_eating (total : ℕ) (likes_apples : ℕ) (likes_fish : ℕ) (likes_both : ℕ) 
  (h1 : total = 75)
  (h2 : likes_apples = 15)
  (h3 : likes_fish = 55)
  (h4 : likes_both = 8) :
  total - (likes_apples + likes_fish - likes_both) = 13 :=
by sorry

end cats_not_eating_l3707_370766


namespace sum_of_solutions_is_zero_l3707_370735

theorem sum_of_solutions_is_zero :
  let f : ℝ → ℝ := λ x => Real.sqrt (9 - x^2 / 4)
  (∀ x, f x = 3 → x = 0) ∧ (∃ x, f x = 3) :=
by
  sorry

end sum_of_solutions_is_zero_l3707_370735


namespace infinite_series_sum_l3707_370791

/-- The sum of the infinite series ∑(k=1 to ∞) k^2 / 2^k equals 6 -/
theorem infinite_series_sum : 
  (∑' k : ℕ, (k^2 : ℝ) / (2 : ℝ)^k) = 6 := by sorry

end infinite_series_sum_l3707_370791


namespace snail_race_l3707_370750

/-- The race problem with three snails -/
theorem snail_race (speed_1 : ℝ) (time_3 : ℝ) : 
  speed_1 = 2 →  -- First snail's speed
  time_3 = 2 →   -- Time taken by the third snail
  (∃ (speed_2 speed_3 distance time_1 : ℝ), 
    speed_2 = 2 * speed_1 ∧             -- Second snail's speed
    speed_3 = 5 * speed_2 ∧             -- Third snail's speed
    distance = speed_3 * time_3 ∧       -- Total distance
    time_1 * speed_1 = distance ∧       -- First snail's time
    time_1 = 20) :=                     -- First snail took 20 minutes
by sorry

end snail_race_l3707_370750


namespace jacqueline_erasers_l3707_370730

/-- The number of boxes of erasers Jacqueline has -/
def num_boxes : ℕ := 4

/-- The number of erasers in each box -/
def erasers_per_box : ℕ := 10

/-- The total number of erasers Jacqueline has -/
def total_erasers : ℕ := num_boxes * erasers_per_box

theorem jacqueline_erasers : total_erasers = 40 :=
by sorry

end jacqueline_erasers_l3707_370730


namespace intersection_of_A_and_B_l3707_370736

def A : Set ℝ := {x | x - 1 ≥ 0}
def B : Set ℝ := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end intersection_of_A_and_B_l3707_370736


namespace quadratic_inequality_l3707_370747

theorem quadratic_inequality (x : ℝ) : 9 - x^2 < 0 → x < -3 ∨ x > 3 := by
  sorry

end quadratic_inequality_l3707_370747


namespace mikes_coins_value_l3707_370789

/-- Represents the number of coins Mike has -/
def total_coins : ℕ := 17

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Calculates the total value of Mike's coins in cents -/
def total_value (dimes quarters : ℕ) : ℕ :=
  dimes * dime_value + quarters * quarter_value

theorem mikes_coins_value :
  ∃ (dimes quarters : ℕ),
    dimes + quarters = total_coins ∧
    quarters + 3 = 2 * dimes ∧
    total_value dimes quarters = 345 := by
  sorry

end mikes_coins_value_l3707_370789


namespace magic_square_sum_l3707_370769

/-- Represents a 3x3 magic square with numbers 1, 2, 3 -/
def MagicSquare : Type := Fin 3 → Fin 3 → Fin 3

/-- Checks if a row contains 1, 2, 3 exactly once -/
def valid_row (square : MagicSquare) (row : Fin 3) : Prop :=
  ∀ n : Fin 3, ∃! col : Fin 3, square row col = n

/-- Checks if a column contains 1, 2, 3 exactly once -/
def valid_column (square : MagicSquare) (col : Fin 3) : Prop :=
  ∀ n : Fin 3, ∃! row : Fin 3, square row col = n

/-- Checks if the main diagonal contains 1, 2, 3 exactly once -/
def valid_diagonal (square : MagicSquare) : Prop :=
  ∀ n : Fin 3, ∃! i : Fin 3, square i i = n

/-- Defines a valid magic square -/
def is_valid_square (square : MagicSquare) : Prop :=
  (∀ row : Fin 3, valid_row square row) ∧
  (∀ col : Fin 3, valid_column square col) ∧
  valid_diagonal square

theorem magic_square_sum :
  ∀ square : MagicSquare,
  is_valid_square square →
  square 0 0 = 2 →
  (square 1 0).val + (square 2 2).val + (square 1 1).val = 6 :=
by sorry

end magic_square_sum_l3707_370769


namespace sin_value_given_conditions_l3707_370732

theorem sin_value_given_conditions (θ : Real) 
  (h1 : Real.sin θ + Real.cos θ = 7/5)
  (h2 : Real.tan θ < 1) : 
  Real.sin θ = 3/5 := by
  sorry

end sin_value_given_conditions_l3707_370732


namespace parabola_circle_tangent_l3707_370748

/-- Represents a parabola in the form y^2 = -8x --/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  directrix : ℝ → Prop

/-- Represents a circle --/
structure Circle where
  center : ℝ × ℝ
  passes_through : ℝ × ℝ

/-- Theorem: The line x = 2 is a common tangent to all circles whose centers 
    lie on the parabola y^2 = -8x and pass through the point (-2, 0) --/
theorem parabola_circle_tangent (p : Parabola) (c : Circle) : 
  p.equation = (fun x y ↦ y^2 = -8*x) →
  p.focus = (-2, 0) →
  p.directrix = (fun x ↦ x = 2) →
  c.passes_through = (-2, 0) →
  (∃ (y : ℝ), p.equation c.center.1 y) →
  (fun x ↦ x = 2) = (fun x ↦ ∃ (y : ℝ), c.center = (x, y) ∧ 
    (c.center.1 - (-2))^2 + (c.center.2 - 0)^2 = (c.center.1 - 2)^2 + c.center.2^2) :=
by sorry

end parabola_circle_tangent_l3707_370748


namespace thirty_five_operations_sufficient_l3707_370784

/-- A type representing a 10x10 grid of integers -/
def Grid := Fin 10 → Fin 10 → ℕ

/-- Predicate to check if a number is composite -/
def IsComposite (n : ℕ) : Prop := n > 1 ∧ ∃ m, 1 < m ∧ m < n ∧ n % m = 0

/-- Predicate to check if two cells are adjacent -/
def IsAdjacent (i j k l : Fin 10) : Prop :=
  (i = k ∧ (j.val + 1 = l.val ∨ j.val = l.val + 1)) ∨
  (j = l ∧ (i.val + 1 = k.val ∨ i.val = k.val + 1))

/-- Predicate to check if a grid satisfies the composite sum condition -/
def SatisfiesCompositeSumCondition (g : Grid) : Prop :=
  ∀ i j k l, IsAdjacent i j k l → IsComposite (g i j + g k l)

/-- Function to represent a swap operation -/
def Swap (g : Grid) (i j k l : Fin 10) : Grid :=
  fun x y => if (x = i ∧ y = j) ∨ (x = k ∧ y = l) then g k l else g x y

/-- Theorem stating that 35 operations are sufficient to achieve the goal -/
theorem thirty_five_operations_sufficient :
  ∃ (initial : Grid) (swaps : List (Fin 10 × Fin 10 × Fin 10 × Fin 10)),
    (∀ i j, initial i j ∈ Set.range (fun n : Fin 100 => n.val + 1)) ∧
    swaps.length ≤ 35 ∧
    SatisfiesCompositeSumCondition (swaps.foldl (fun g (i, j, k, l) => Swap g i j k l) initial) :=
  sorry


end thirty_five_operations_sufficient_l3707_370784


namespace twelfth_term_of_specific_arithmetic_sequence_l3707_370758

/-- An arithmetic sequence with a given first term and second term -/
def arithmeticSequence (a₁ : ℚ) (a₂ : ℚ) : ℕ → ℚ :=
  λ n => a₁ + (n - 1) * (a₂ - a₁)

/-- Theorem: The 12th term of the arithmetic sequence with first term 1/2 and second term 5/6 is 25/6 -/
theorem twelfth_term_of_specific_arithmetic_sequence :
  arithmeticSequence (1/2) (5/6) 12 = 25/6 := by
  sorry

end twelfth_term_of_specific_arithmetic_sequence_l3707_370758


namespace sin_alpha_value_l3707_370785

theorem sin_alpha_value (α β : Real) 
  (h1 : (0 : Real) < α ∧ α < Real.pi / 2)
  (h2 : -Real.pi / 2 < β ∧ β < 0)
  (h3 : Real.sin β = -5 / 13)
  (h4 : Real.sqrt ((Real.cos α - Real.cos β)^2 + (Real.sin α - Real.sin β)^2) = 2 * Real.sqrt 5 / 5) :
  Real.sin α = 33 / 65 := by
sorry

end sin_alpha_value_l3707_370785


namespace derivative_even_implies_b_zero_l3707_370739

/-- A cubic polynomial function -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + 2

/-- The derivative of f -/
def f' (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

/-- A function is even if f(x) = f(-x) for all x -/
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

/-- If the derivative of f is even, then b = 0 -/
theorem derivative_even_implies_b_zero (a b c : ℝ) :
  is_even (f' a b c) → b = 0 := by sorry

end derivative_even_implies_b_zero_l3707_370739


namespace min_toothpicks_theorem_l3707_370761

/-- A geometric figure made of toothpicks -/
structure ToothpickFigure where
  upward_triangles : ℕ
  downward_triangles : ℕ
  horizontal_toothpicks : ℕ

/-- The minimum number of toothpicks to remove to eliminate all triangles -/
def min_toothpicks_to_remove (figure : ToothpickFigure) : ℕ :=
  figure.horizontal_toothpicks

/-- Theorem stating the minimum number of toothpicks to remove -/
theorem min_toothpicks_theorem (figure : ToothpickFigure) 
  (h1 : figure.upward_triangles = 15)
  (h2 : figure.downward_triangles = 10)
  (h3 : figure.horizontal_toothpicks = 15) :
  min_toothpicks_to_remove figure = 15 := by
  sorry

#check min_toothpicks_theorem

end min_toothpicks_theorem_l3707_370761


namespace point_P_in_fourth_quadrant_iff_a_in_range_l3707_370772

/-- A point in the fourth quadrant has a positive x-coordinate and a negative y-coordinate -/
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- The coordinates of point P are defined in terms of parameter a -/
def point_P (a : ℝ) : ℝ × ℝ := (2*a + 4, 3*a - 6)

/-- Theorem stating the range of a for point P to be in the fourth quadrant -/
theorem point_P_in_fourth_quadrant_iff_a_in_range :
  ∀ a : ℝ, fourth_quadrant (point_P a).1 (point_P a).2 ↔ -2 < a ∧ a < 2 := by
  sorry

end point_P_in_fourth_quadrant_iff_a_in_range_l3707_370772


namespace trajectory_of_P_l3707_370753

-- Define the circle F
def circle_F (x y : ℝ) : Prop := x^2 - 2*x + y^2 - 11 = 0

-- Define point A
def point_A : ℝ × ℝ := (-1, 0)

-- Define the moving point B on circle F
def point_B : Set (ℝ × ℝ) := {p : ℝ × ℝ | circle_F p.1 p.2}

-- Define the perpendicular bisector of AB
def perp_bisector (A B : ℝ × ℝ) : Set (ℝ × ℝ) := 
  {P : ℝ × ℝ | (P.1 - A.1)^2 + (P.2 - A.2)^2 = (P.1 - B.1)^2 + (P.2 - B.2)^2}

-- Define point P
def point_P (B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P : ℝ × ℝ | P ∈ perp_bisector point_A B ∧ 
               ∃ t : ℝ, P = (t * B.1 + (1-t) * 1, t * B.2)}

-- Theorem statement
theorem trajectory_of_P :
  ∀ P : ℝ × ℝ, (∃ B ∈ point_B, P ∈ point_P B) → 
  P.1^2 / 3 + P.2^2 / 2 = 1 :=
sorry

end trajectory_of_P_l3707_370753


namespace product_properties_l3707_370709

-- Define a function to represent the product of all combinations
def product_of_combinations (a : List ℕ) : ℝ :=
  sorry

-- Theorem statement
theorem product_properties (a : List ℕ) :
  (∃ m : ℤ, product_of_combinations a = m) ∧
  (∃ n : ℤ, product_of_combinations a = n^2) :=
sorry

end product_properties_l3707_370709


namespace geometric_sequence_product_l3707_370795

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The roots of the equation x^2 - 2x - 3 = 0 -/
def roots_equation (x y : ℝ) : Prop :=
  x^2 - 2*x - 3 = 0 ∧ y^2 - 2*y - 3 = 0

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  roots_equation (a 1) (a 4) →
  a 2 * a 3 = -3 := by
sorry

end geometric_sequence_product_l3707_370795


namespace circle_line_intersection_l3707_370738

-- Define the circle C
def circle_equation (x y : ℝ) : Prop :=
  (x - 2)^2 + (y + 3)^2 = 13

-- Define the line l
def line_equation (x y θ : ℝ) : Prop :=
  ∃ t, x = 4 + t * Real.cos θ ∧ y = t * Real.sin θ

-- Define the intersection condition
def intersects_at_two_points (C : ℝ → ℝ → Prop) (l : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∃ x₁ y₁ x₂ y₂ θ, 
    C x₁ y₁ ∧ C x₂ y₂ ∧ 
    l x₁ y₁ θ ∧ l x₂ y₂ θ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

-- Define the distance condition
def distance_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = 16

-- Main theorem
theorem circle_line_intersection :
  intersects_at_two_points circle_equation line_equation →
  (∃ x₁ y₁ x₂ y₂ θ, 
    circle_equation x₁ y₁ ∧ circle_equation x₂ y₂ ∧
    line_equation x₁ y₁ θ ∧ line_equation x₂ y₂ θ ∧
    distance_condition x₁ y₁ x₂ y₂) →
  ∃ k, k = 0 ∨ k = -12/5 :=
sorry

end circle_line_intersection_l3707_370738


namespace roof_dimension_difference_l3707_370790

/-- Proves that for a rectangular roof where the length is 7 times the width
    and the area is 847 square feet, the difference between the length
    and the width is 66 feet. -/
theorem roof_dimension_difference (width : ℝ) (length : ℝ) : 
  length = 7 * width →
  length * width = 847 →
  length - width = 66 := by
sorry

end roof_dimension_difference_l3707_370790


namespace equation_and_inequalities_solution_l3707_370793

theorem equation_and_inequalities_solution :
  (∃! x : ℝ, (3 / (x - 1) = 1 / (2 * x + 3))) ∧
  (∀ x : ℝ, (3 * x - 1 ≥ x + 1 ∧ x + 3 > 4 * x - 2) ↔ (1 ≤ x ∧ x < 5 / 3)) := by
  sorry

end equation_and_inequalities_solution_l3707_370793


namespace tammy_haircuts_l3707_370775

/-- The number of paid haircuts required to get a free haircut -/
def haircuts_for_free : ℕ := 14

/-- The number of free haircuts Tammy has already received -/
def free_haircuts_received : ℕ := 5

/-- The number of haircuts Tammy needs for her next free one -/
def haircuts_until_next_free : ℕ := 5

/-- The total number of haircuts Tammy has gotten -/
def total_haircuts : ℕ := 79

theorem tammy_haircuts :
  total_haircuts = 
    (free_haircuts_received * haircuts_for_free) + 
    (haircuts_for_free - haircuts_until_next_free) :=
by sorry

end tammy_haircuts_l3707_370775


namespace range_of_t_l3707_370777

theorem range_of_t (t α β : ℝ) 
  (h1 : t = Real.cos β ^ 3 + (α / 2) * Real.cos β)
  (h2 : α ≤ t)
  (h3 : t ≤ α - 5 * Real.cos β) :
  -2/3 ≤ t ∧ t ≤ 1 := by
  sorry

end range_of_t_l3707_370777


namespace jack_plates_left_l3707_370780

/-- Represents the number of plates Jack has with different patterns -/
structure PlateCollection where
  flower : ℕ
  checked : ℕ
  polkadot : ℕ

/-- Calculates the total number of plates after Jack's actions -/
def total_plates_after_actions (initial : PlateCollection) : ℕ :=
  (initial.flower - 1) + initial.checked + (2 * initial.checked)

/-- Theorem stating that Jack has 27 plates left after his actions -/
theorem jack_plates_left (initial : PlateCollection) 
  (h1 : initial.flower = 4)
  (h2 : initial.checked = 8)
  (h3 : initial.polkadot = 0) : 
  total_plates_after_actions initial = 27 := by
  sorry

#check jack_plates_left

end jack_plates_left_l3707_370780


namespace sequence_product_theorem_l3707_370733

def arithmetic_sequence (n : ℕ) : ℕ :=
  2 * n - 1

def geometric_sequence (n : ℕ) : ℕ :=
  2^(n - 1)

theorem sequence_product_theorem :
  let a := arithmetic_sequence
  let b := geometric_sequence
  b (a 1) * b (a 3) * b (a 5) = 4096 := by
  sorry

end sequence_product_theorem_l3707_370733


namespace race_heartbeats_l3707_370744

/-- Calculates the total number of heartbeats during a race. -/
def total_heartbeats (heart_rate : ℕ) (pace : ℕ) (race_distance : ℕ) : ℕ :=
  let race_time := pace * race_distance
  race_time * heart_rate

/-- Proves that the total number of heartbeats during a 26-mile race is 19500,
    given the specified heart rate and pace. -/
theorem race_heartbeats :
  total_heartbeats 150 5 26 = 19500 :=
by sorry

end race_heartbeats_l3707_370744


namespace min_distance_theorem_l3707_370718

/-- Represents a rectangular cave with four points A, B, C, and D -/
structure RectangularCave where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  AD : ℝ

/-- The minimum distance to cover all paths from A to C in a rectangular cave -/
def min_distance_all_paths (cave : RectangularCave) : ℝ :=
  cave.AB + cave.BC + cave.CD + cave.AD

/-- Theorem stating the minimum distance to cover all paths from A to C -/
theorem min_distance_theorem (cave : RectangularCave) 
  (h1 : cave.AB + cave.BC + cave.CD = 22)
  (h2 : cave.AD + cave.CD + cave.BC = 29)
  (h3 : cave.AB + cave.BC + (cave.AB + cave.AD) = 30) :
  min_distance_all_paths cave = 47 := by
  sorry

#eval min_distance_all_paths ⟨10, 5, 7, 12⟩

end min_distance_theorem_l3707_370718


namespace ac_value_l3707_370701

theorem ac_value (x : ℕ+) 
  (h1 : ∃ y : ℕ, (2 * x + 1 : ℕ) = y^2)
  (h2 : ∃ z : ℕ, (3 * x + 1 : ℕ) = z^2) : 
  x = 40 := by
sorry

end ac_value_l3707_370701


namespace trash_time_fraction_l3707_370764

def movie_time : ℕ := 120 -- 2 hours in minutes
def homework_time : ℕ := 30
def cleaning_time : ℕ := homework_time / 2
def dog_walking_time : ℕ := homework_time + 5
def time_left : ℕ := 35

def total_known_tasks : ℕ := homework_time + cleaning_time + dog_walking_time

theorem trash_time_fraction (trash_time : ℕ) : 
  trash_time = movie_time - time_left - total_known_tasks →
  trash_time * 6 = homework_time :=
by sorry

end trash_time_fraction_l3707_370764


namespace complement_A_intersect_B_l3707_370786

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} := by sorry

end complement_A_intersect_B_l3707_370786


namespace sphere_radius_l3707_370745

/-- The radius of a sphere that forms a quarter-sphere with radius 4∛4 cm is 4 cm. -/
theorem sphere_radius (r : ℝ) : r = 4 * Real.rpow 4 (1/3) → 4 = (1/4)^(1/3) * r := by
  sorry

end sphere_radius_l3707_370745


namespace simplified_expression_equals_two_thirds_l3707_370710

theorem simplified_expression_equals_two_thirds :
  let x : ℚ := 5
  (1 - 1 / (x + 1)) / ((x^2 - x) / (x^2 - 2*x + 1)) = 2/3 :=
by sorry

end simplified_expression_equals_two_thirds_l3707_370710


namespace g_sum_equals_negative_two_l3707_370788

/-- Piecewise function g(x, y) -/
noncomputable def g (x y : ℝ) : ℝ :=
  if x - y ≤ 1 then (x^2 * y - x + 3) / (3 * x)
  else (x^2 * y - y - 3) / (-3 * y)

/-- Theorem stating that g(3,2) + g(4,1) = -2 -/
theorem g_sum_equals_negative_two : g 3 2 + g 4 1 = -2 := by
  sorry

end g_sum_equals_negative_two_l3707_370788


namespace sufficient_not_necessary_l3707_370763

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a > 2 → a^2 > 2*a) ∧ 
  (∃ a, a ≤ 2 ∧ a^2 > 2*a) := by
  sorry

end sufficient_not_necessary_l3707_370763


namespace power_sum_value_l3707_370754

theorem power_sum_value (a : ℝ) (m n : ℤ) (h1 : a^m = 2) (h2 : a^n = 1) :
  a^(m + 2*n) = 2 := by
  sorry

end power_sum_value_l3707_370754


namespace first_applicant_earnings_l3707_370723

def first_applicant_salary : ℕ := 42000
def first_applicant_training_months : ℕ := 3
def first_applicant_training_cost_per_month : ℕ := 1200

def second_applicant_salary : ℕ := 45000
def second_applicant_revenue : ℕ := 92000
def second_applicant_bonus_percentage : ℚ := 1 / 100

def difference_in_earnings : ℕ := 850

theorem first_applicant_earnings :
  let first_total_cost := first_applicant_salary + first_applicant_training_months * first_applicant_training_cost_per_month
  let second_total_cost := second_applicant_salary + (second_applicant_salary : ℚ) * second_applicant_bonus_percentage
  let second_net_earnings := second_applicant_revenue - second_total_cost
  first_total_cost + (second_net_earnings - difference_in_earnings) = 45700 :=
by sorry

end first_applicant_earnings_l3707_370723


namespace f_prime_minus_g_prime_at_one_l3707_370737

-- Define f and g as differentiable functions on ℝ
variable (f g : ℝ → ℝ)

-- Define the conditions
variable (h1 : Differentiable ℝ f)
variable (h2 : Differentiable ℝ g)
variable (h3 : ∀ x, f x = x * g x + x^2 - 1)
variable (h4 : f 1 = 1)

-- State the theorem
theorem f_prime_minus_g_prime_at_one :
  deriv f 1 - deriv g 1 = 3 := by sorry

end f_prime_minus_g_prime_at_one_l3707_370737


namespace dye_job_price_is_correct_l3707_370734

/-- The price of a haircut in dollars -/
def haircut_price : ℕ := 30

/-- The price of a perm in dollars -/
def perm_price : ℕ := 40

/-- The cost of hair dye for one dye job in dollars -/
def dye_cost : ℕ := 10

/-- The number of haircuts scheduled -/
def num_haircuts : ℕ := 4

/-- The number of perms scheduled -/
def num_perms : ℕ := 1

/-- The number of dye jobs scheduled -/
def num_dye_jobs : ℕ := 2

/-- The amount of tips in dollars -/
def tips : ℕ := 50

/-- The total earnings at the end of the day in dollars -/
def total_earnings : ℕ := 310

/-- The price of a dye job in dollars -/
def dye_job_price : ℕ := 60

theorem dye_job_price_is_correct : 
  num_haircuts * haircut_price + 
  num_perms * perm_price + 
  num_dye_jobs * (dye_job_price - dye_cost) + 
  tips = total_earnings := by sorry

end dye_job_price_is_correct_l3707_370734


namespace right_triangle_GHI_side_GH_l3707_370797

/-- Represents a right triangle GHI with angle G = 30°, angle H = 90°, and HI = 10 -/
structure RightTriangleGHI where
  G : Real
  H : Real
  I : Real
  angleG : G = 30 * π / 180
  angleH : H = π / 2
  rightAngle : H = π / 2
  sideHI : I = 10

/-- Theorem stating that in the given right triangle GHI, GH = 10√3 -/
theorem right_triangle_GHI_side_GH (t : RightTriangleGHI) : 
  Real.sqrt ((10 * Real.sqrt 3) ^ 2) = 10 * Real.sqrt 3 := by
  sorry

end right_triangle_GHI_side_GH_l3707_370797


namespace cube_volume_surface_area_l3707_370799

/-- For a cube with volume 8y cubic units and surface area 6y square units, y = 64 -/
theorem cube_volume_surface_area (y : ℝ) (h1 : y > 0) : 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 8*y ∧ 6*s^2 = 6*y) → y = 64 := by
sorry

end cube_volume_surface_area_l3707_370799


namespace fraction_order_l3707_370794

theorem fraction_order : 
  let f₁ : ℚ := 16 / 12
  let f₂ : ℚ := 20 / 16
  let f₃ : ℚ := 18 / 14
  let f₄ : ℚ := 22 / 17
  f₂ < f₃ ∧ f₃ < f₄ ∧ f₄ < f₁ :=
by
  sorry

end fraction_order_l3707_370794


namespace circumscribed_sphere_area_l3707_370724

theorem circumscribed_sphere_area (x y z : ℝ) (h1 : x * y = 6) (h2 : y * z = 10) (h3 : z * x = 15) :
  4 * Real.pi * ((x^2 + y^2 + z^2) / 4) = 38 * Real.pi :=
by sorry

end circumscribed_sphere_area_l3707_370724


namespace equation_solution_l3707_370702

theorem equation_solution : ∃ x : ℚ, (2 / 5 : ℚ) - (1 / 7 : ℚ) = 1 / x ∧ x = 35 / 9 := by
  sorry

end equation_solution_l3707_370702


namespace scaled_arithmetic_sequence_l3707_370721

/-- Given an arithmetic sequence and a non-zero constant, prove that scaling the sequence by the constant results in another arithmetic sequence with a scaled common difference. -/
theorem scaled_arithmetic_sequence
  (a : ℕ → ℝ) -- The original arithmetic sequence
  (d : ℝ) -- Common difference of the original sequence
  (c : ℝ) -- Scaling constant
  (h₁ : c ≠ 0) -- Assumption that c is non-zero
  (h₂ : ∀ n : ℕ, a (n + 1) - a n = d) -- Definition of arithmetic sequence
  : ∀ n : ℕ, (c * a (n + 1)) - (c * a n) = c * d := by
  sorry

end scaled_arithmetic_sequence_l3707_370721


namespace woodburning_profit_l3707_370756

/-- Calculates the profit from selling woodburnings -/
theorem woodburning_profit
  (num_sold : ℕ)
  (price_per_item : ℝ)
  (cost : ℝ)
  (h1 : num_sold = 20)
  (h2 : price_per_item = 15)
  (h3 : cost = 100) :
  num_sold * price_per_item - cost = 200 :=
by
  sorry

end woodburning_profit_l3707_370756


namespace shaded_fraction_of_rectangle_l3707_370759

/-- Given a rectangle with dimensions 12 and 18, prove that the fraction of the rectangle
    that is shaded is 1/12, where the shaded region is 1/3 of a quarter of the rectangle. -/
theorem shaded_fraction_of_rectangle (width : ℕ) (height : ℕ) (shaded_area : ℚ) :
  width = 12 →
  height = 18 →
  shaded_area = (1 / 3) * (1 / 4) * (width * height) →
  shaded_area / (width * height) = 1 / 12 :=
by sorry

end shaded_fraction_of_rectangle_l3707_370759


namespace orthogonal_equal_magnitude_vectors_l3707_370771

/-- Given two vectors a and b in R^3, prove that if they are orthogonal and have equal magnitude,
    then their components satisfy specific values. -/
theorem orthogonal_equal_magnitude_vectors 
  (a b : ℝ × ℝ × ℝ) 
  (h_a : a.1 = 4 ∧ a.2.2 = -2) 
  (h_b : b.1 = 1 ∧ b.2.1 = 2) 
  (h_orthogonal : a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2 = 0) 
  (h_equal_magnitude : a.1^2 + a.2.1^2 + a.2.2^2 = b.1^2 + b.2.1^2 + b.2.2^2) :
  a.2.1 = 11/4 ∧ b.2.2 = 19/4 := by
  sorry

end orthogonal_equal_magnitude_vectors_l3707_370771


namespace number_equation_proof_l3707_370727

theorem number_equation_proof : ∃ x : ℝ, 5020 - (x / 100.4) = 5015 ∧ x = 502 := by
  sorry

end number_equation_proof_l3707_370727


namespace special_function_property_l3707_370712

/-- A function satisfying the given property for all real numbers -/
def SatisfiesProperty (g : ℝ → ℝ) : Prop :=
  ∀ c d : ℝ, c^2 * g d = d^2 * g c

theorem special_function_property (g : ℝ → ℝ) 
  (h1 : SatisfiesProperty g) (h2 : g 4 ≠ 0) : 
  (g 7 - g 3) / g 4 = 2.5 := by sorry

end special_function_property_l3707_370712


namespace number_puzzle_l3707_370781

theorem number_puzzle (x : ℤ) : x - 62 + 45 = 55 → 7 * x = 504 := by
  sorry

end number_puzzle_l3707_370781


namespace original_equals_scientific_l3707_370717

/-- Represents 1 million -/
def million : ℝ := 10^6

/-- The number to be converted -/
def original_number : ℝ := 456.87 * million

/-- The scientific notation representation -/
def scientific_notation : ℝ := 4.5687 * 10^8

theorem original_equals_scientific : original_number = scientific_notation := by
  sorry

end original_equals_scientific_l3707_370717


namespace smallest_four_digit_divisible_by_three_l3707_370768

theorem smallest_four_digit_divisible_by_three :
  ∃ n : ℕ, (1000 ≤ n ∧ n ≤ 9999) ∧ 
           n % 3 = 0 ∧
           (∀ m : ℕ, (1000 ≤ m ∧ m < n) → m % 3 ≠ 0) ∧
           n = 1002 := by
  sorry

end smallest_four_digit_divisible_by_three_l3707_370768


namespace line_opposite_sides_m_range_l3707_370700

/-- A line in 2D space defined by the equation 3x - 2y + m = 0 -/
structure Line (m : ℝ) where
  equation : ℝ → ℝ → ℝ
  eq_def : equation = fun x y => 3 * x - 2 * y + m

/-- Determines if two points are on opposite sides of a line -/
def opposite_sides (l : Line m) (p1 p2 : ℝ × ℝ) : Prop :=
  l.equation p1.1 p1.2 * l.equation p2.1 p2.2 < 0

theorem line_opposite_sides_m_range (m : ℝ) (l : Line m) :
  opposite_sides l (3, 1) (-4, 6) → -7 < m ∧ m < 24 := by
  sorry


end line_opposite_sides_m_range_l3707_370700


namespace position_of_2018_l3707_370783

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that returns the sequence of natural numbers whose digits sum to 11, in ascending order -/
def sequence_sum_11 : List ℕ := sorry

/-- The position of a number in a list, with 1-based indexing -/
def position_in_list (n : ℕ) (l : List ℕ) : ℕ := sorry

theorem position_of_2018 : 
  position_in_list 2018 sequence_sum_11 = 134 := by sorry

end position_of_2018_l3707_370783
