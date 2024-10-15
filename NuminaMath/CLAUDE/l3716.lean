import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_fraction_subtraction_l3716_371669

theorem arithmetic_fraction_subtraction :
  (2 + 4 + 6) / (1 + 3 + 5) - (1 + 3 + 5) / (2 + 4 + 6) = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_fraction_subtraction_l3716_371669


namespace NUMINAMATH_CALUDE_henrikhs_distance_l3716_371600

/-- The number of blocks Henrikh lives from his office. -/
def blocks : ℕ :=
  sorry

/-- The time in minutes it takes Henrikh to walk to work. -/
def walkTime : ℚ :=
  blocks

/-- The time in minutes it takes Henrikh to cycle to work. -/
def cycleTime : ℚ :=
  blocks * (20 / 60)

theorem henrikhs_distance :
  blocks = 12 ∧ walkTime = cycleTime + 8 :=
by sorry

end NUMINAMATH_CALUDE_henrikhs_distance_l3716_371600


namespace NUMINAMATH_CALUDE_num_technicians_correct_l3716_371606

/-- Represents the number of technicians in a workshop. -/
def num_technicians : ℕ := 7

/-- Represents the total number of workers in the workshop. -/
def total_workers : ℕ := 49

/-- Represents the average salary of all workers in the workshop. -/
def avg_salary_all : ℕ := 8000

/-- Represents the average salary of technicians in the workshop. -/
def avg_salary_technicians : ℕ := 20000

/-- Represents the average salary of non-technician workers in the workshop. -/
def avg_salary_rest : ℕ := 6000

/-- Theorem stating that the number of technicians satisfies the given conditions. -/
theorem num_technicians_correct :
  num_technicians * avg_salary_technicians +
  (total_workers - num_technicians) * avg_salary_rest =
  total_workers * avg_salary_all :=
by sorry

end NUMINAMATH_CALUDE_num_technicians_correct_l3716_371606


namespace NUMINAMATH_CALUDE_helen_cookies_l3716_371654

/-- The number of chocolate chip cookies Helen baked yesterday -/
def cookies_yesterday : ℕ := 1081 - 554

/-- The total number of chocolate chip cookies Helen baked -/
def total_cookies : ℕ := 1081

/-- The number of chocolate chip cookies Helen baked this morning -/
def cookies_this_morning : ℕ := 554

theorem helen_cookies : cookies_yesterday = 527 := by
  sorry

end NUMINAMATH_CALUDE_helen_cookies_l3716_371654


namespace NUMINAMATH_CALUDE_not_always_sufficient_condition_l3716_371644

theorem not_always_sufficient_condition : 
  ¬(∀ (a b c : ℝ), a > b → a * c^2 > b * c^2) :=
by sorry

end NUMINAMATH_CALUDE_not_always_sufficient_condition_l3716_371644


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l3716_371679

-- Define the types for our variables
variables {a b c : ℝ}

-- Define the solution set of the first inequality
def solution_set_1 : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}

-- Define the solution set of the second inequality
def solution_set_2 : Set ℝ := {x | x ≤ -1 ∨ x ≥ -1/2}

-- State the theorem
theorem inequality_solution_sets :
  (∀ x, ax^2 - b*x + c ≥ 0 ↔ x ∈ solution_set_1) →
  (∀ x, c*x^2 + b*x + a ≤ 0 ↔ x ∈ solution_set_2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l3716_371679


namespace NUMINAMATH_CALUDE_greatest_integer_for_all_real_domain_l3716_371643

theorem greatest_integer_for_all_real_domain (a : ℤ) : 
  (∀ x : ℝ, (x^2 + a*x + 15 ≠ 0)) ↔ a ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_for_all_real_domain_l3716_371643


namespace NUMINAMATH_CALUDE_hostel_expenditure_increase_l3716_371622

theorem hostel_expenditure_increase 
  (initial_students : ℕ) 
  (new_students : ℕ) 
  (budget_decrease : ℕ) 
  (new_total_expenditure : ℕ) 
  (h1 : initial_students = 100)
  (h2 : new_students = 132)
  (h3 : budget_decrease = 10)
  (h4 : new_total_expenditure = 5400) :
  ∃ (original_avg_budget : ℕ),
    new_total_expenditure - initial_students * original_avg_budget = 300 := by
  sorry

end NUMINAMATH_CALUDE_hostel_expenditure_increase_l3716_371622


namespace NUMINAMATH_CALUDE_ppf_combination_l3716_371647

/-- Production Possibility Frontier (PPF) for a single female -/
def individual_ppf (K : ℝ) : ℝ := 40 - 2 * K

/-- Combined Production Possibility Frontier (PPF) for two females -/
def combined_ppf (K : ℝ) : ℝ := 80 - 2 * K

theorem ppf_combination (K : ℝ) (h : K ≤ 40) :
  combined_ppf K = individual_ppf (K / 2) + individual_ppf (K / 2) :=
by sorry

#check ppf_combination

end NUMINAMATH_CALUDE_ppf_combination_l3716_371647


namespace NUMINAMATH_CALUDE_sin_two_alpha_zero_l3716_371619

theorem sin_two_alpha_zero (α : Real) (f : Real → Real)
  (h1 : ∀ x, f x = Real.sin x - Real.cos x)
  (h2 : f α = 1) : Real.sin (2 * α) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_alpha_zero_l3716_371619


namespace NUMINAMATH_CALUDE_third_month_sale_l3716_371636

def average_sale : ℕ := 6500
def num_months : ℕ := 6
def sale_month1 : ℕ := 6535
def sale_month2 : ℕ := 6927
def sale_month4 : ℕ := 7230
def sale_month5 : ℕ := 6562
def sale_month6 : ℕ := 4891

theorem third_month_sale :
  ∃ (sale_month3 : ℕ),
    sale_month3 = average_sale * num_months - (sale_month1 + sale_month2 + sale_month4 + sale_month5 + sale_month6) ∧
    sale_month3 = 6855 := by
  sorry

end NUMINAMATH_CALUDE_third_month_sale_l3716_371636


namespace NUMINAMATH_CALUDE_white_surface_fraction_is_seven_eighths_l3716_371615

/-- Represents a cube with white and black smaller cubes -/
structure ColoredCube where
  edge_length : ℕ
  total_small_cubes : ℕ
  white_cubes : ℕ
  black_cubes : ℕ

/-- Calculates the fraction of white surface area for a colored cube -/
def white_surface_fraction (c : ColoredCube) : ℚ :=
  sorry

/-- Theorem: The fraction of white surface area for the given cube configuration is 7/8 -/
theorem white_surface_fraction_is_seven_eighths :
  let c : ColoredCube := {
    edge_length := 4,
    total_small_cubes := 64,
    white_cubes := 48,
    black_cubes := 16
  }
  white_surface_fraction c = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_white_surface_fraction_is_seven_eighths_l3716_371615


namespace NUMINAMATH_CALUDE_polynomial_with_negative_integer_roots_l3716_371639

/-- A polynomial of degree 4 with integer coefficients -/
structure Polynomial4 where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- The polynomial function corresponding to a Polynomial4 -/
def poly_func (p : Polynomial4) : ℝ → ℝ :=
  fun x ↦ x^4 + p.a * x^3 + p.b * x^2 + p.c * x + p.d

/-- Predicate stating that all roots of a polynomial are negative integers -/
def all_roots_negative_integers (p : Polynomial4) : Prop :=
  ∀ x : ℝ, poly_func p x = 0 → (∃ n : ℤ, x = ↑n ∧ n < 0)

theorem polynomial_with_negative_integer_roots
  (p : Polynomial4)
  (h_roots : all_roots_negative_integers p)
  (h_sum : p.a + p.b + p.c + p.d = 2009) :
  p.d = 528 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_with_negative_integer_roots_l3716_371639


namespace NUMINAMATH_CALUDE_distribute_seven_balls_four_boxes_l3716_371656

/-- Represents the number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 7 distinguishable balls into 4 indistinguishable boxes is 495 -/
theorem distribute_seven_balls_four_boxes : distribute_balls 7 4 = 495 := by sorry

end NUMINAMATH_CALUDE_distribute_seven_balls_four_boxes_l3716_371656


namespace NUMINAMATH_CALUDE_greatest_integer_solution_l3716_371666

theorem greatest_integer_solution (x : ℤ) : 
  (∀ y : ℤ, y > x → 7 - 5 * y + y^2 ≥ 28) ∧ 
  (7 - 5 * x + x^2 < 28) → 
  x = 7 := by
sorry

end NUMINAMATH_CALUDE_greatest_integer_solution_l3716_371666


namespace NUMINAMATH_CALUDE_danai_decorations_l3716_371688

/-- The total number of decorations Danai will put up -/
def total_decorations (skulls broomsticks spiderwebs cauldrons additional_budget left_to_put_up : ℕ) : ℕ :=
  skulls + broomsticks + spiderwebs + (spiderwebs * 2) + cauldrons + additional_budget + left_to_put_up

/-- Theorem stating the total number of decorations Danai will put up -/
theorem danai_decorations : 
  total_decorations 12 4 12 1 20 10 = 83 := by
  sorry

end NUMINAMATH_CALUDE_danai_decorations_l3716_371688


namespace NUMINAMATH_CALUDE_ellipse_condition_l3716_371675

/-- The equation of the graph -/
def graph_equation (x y k : ℝ) : Prop :=
  3 * x^2 + 9 * y^2 - 6 * x + 27 * y = k

/-- The condition for a non-degenerate ellipse -/
def is_non_degenerate_ellipse (k : ℝ) : Prop :=
  k > -93/4

/-- Theorem: The graph is a non-degenerate ellipse iff k > -93/4 -/
theorem ellipse_condition :
  ∀ k, (∃ x y, graph_equation x y k) ↔ is_non_degenerate_ellipse k :=
by sorry

end NUMINAMATH_CALUDE_ellipse_condition_l3716_371675


namespace NUMINAMATH_CALUDE_problem_solution_l3716_371687

theorem problem_solution (x y z : ℝ) : 
  3 * x = 0.75 * y → 
  x + z = 24 → 
  z = 8 → 
  y = 64 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3716_371687


namespace NUMINAMATH_CALUDE_factorial_of_factorial_divided_by_factorial_l3716_371690

theorem factorial_of_factorial_divided_by_factorial :
  (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = 25852016738884976640000 := by
  sorry

end NUMINAMATH_CALUDE_factorial_of_factorial_divided_by_factorial_l3716_371690


namespace NUMINAMATH_CALUDE_problem_statement_l3716_371676

theorem problem_statement (a b x y : ℕ+) (P : ℕ) 
  (h1 : ∃ k : ℕ, a * x + b * y = k * (a^2 + b^2))
  (h2 : P = x^2 + y^2)
  (h3 : Nat.Prime P) :
  (P ∣ (a^2 + b^2)) ∧ (a = x ∧ b = y) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3716_371676


namespace NUMINAMATH_CALUDE_thirty_five_million_scientific_notation_l3716_371646

-- Define the value of one million
def million : ℝ := 10^6

-- Theorem statement
theorem thirty_five_million_scientific_notation :
  35 * million = 3.5 * 10^7 := by
  sorry

end NUMINAMATH_CALUDE_thirty_five_million_scientific_notation_l3716_371646


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3716_371663

-- Define the circles
def circle1 (x y a : ℝ) : Prop := x^2 + y^2 + 2*a*x + a^2 - 9 = 0
def circle2 (x y b : ℝ) : Prop := x^2 + y^2 - 4*b*y - 1 + 4*b^2 = 0

-- Define the theorem
theorem min_value_of_expression (a b : ℝ) 
  (h1 : ∃ x y, circle1 x y a)
  (h2 : ∃ x y, circle2 x y b)
  (h3 : ∃ t1 t2 t3 : ℝ × ℝ, t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧ 
    (∀ x y, circle1 x y a → (t1.1 * x + t1.2 * y = 1 ∨ t2.1 * x + t2.2 * y = 1 ∨ t3.1 * x + t3.2 * y = 1)) ∧
    (∀ x y, circle2 x y b → (t1.1 * x + t1.2 * y = 1 ∨ t2.1 * x + t2.2 * y = 1 ∨ t3.1 * x + t3.2 * y = 1)))
  (h4 : a ≠ 0)
  (h5 : b ≠ 0) :
  ∃ m : ℝ, m = 1 ∧ ∀ a b : ℝ, a ≠ 0 → b ≠ 0 → 4 / a^2 + 1 / b^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3716_371663


namespace NUMINAMATH_CALUDE_cooper_pies_per_day_l3716_371660

/-- The number of days Cooper makes pies -/
def days : ℕ := 12

/-- The number of pies Ashley eats -/
def pies_eaten : ℕ := 50

/-- The number of pies remaining -/
def pies_remaining : ℕ := 34

/-- The number of pies Cooper makes per day -/
def pies_per_day : ℕ := 7

theorem cooper_pies_per_day :
  days * pies_per_day - pies_eaten = pies_remaining :=
by sorry

end NUMINAMATH_CALUDE_cooper_pies_per_day_l3716_371660


namespace NUMINAMATH_CALUDE_decagon_adjacent_vertex_probability_l3716_371648

/-- A decagon is a polygon with 10 vertices -/
def Decagon : ℕ := 10

/-- The number of vertices adjacent to any given vertex in a decagon -/
def AdjacentVertices : ℕ := 2

/-- The probability of selecting two adjacent vertices when choosing two distinct vertices at random from a decagon -/
theorem decagon_adjacent_vertex_probability : 
  (AdjacentVertices : ℚ) / (Decagon - 1 : ℚ) = 2 / 9 := by sorry

end NUMINAMATH_CALUDE_decagon_adjacent_vertex_probability_l3716_371648


namespace NUMINAMATH_CALUDE_bee_multiplier_l3716_371659

/-- Given the number of bees seen on two consecutive days, 
    prove that the ratio of bees on the second day to the first day is 3 -/
theorem bee_multiplier (bees_day1 bees_day2 : ℕ) 
  (h1 : bees_day1 = 144) 
  (h2 : bees_day2 = 432) : 
  (bees_day2 : ℚ) / bees_day1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_bee_multiplier_l3716_371659


namespace NUMINAMATH_CALUDE_product_terminal_zeros_l3716_371674

/-- The number of terminal zeros in a positive integer -/
def terminalZeros (n : ℕ) : ℕ := sorry

/-- The product of 50 and 480 -/
def product : ℕ := 50 * 480

theorem product_terminal_zeros :
  terminalZeros product = 3 := by sorry

end NUMINAMATH_CALUDE_product_terminal_zeros_l3716_371674


namespace NUMINAMATH_CALUDE_smallest_positive_largest_negative_smallest_absolute_l3716_371694

theorem smallest_positive_largest_negative_smallest_absolute (triangle : ℕ) (O : ℤ) (square : ℚ) : 
  (∀ n : ℕ, n > 0 → triangle ≤ n) →
  (∀ z : ℤ, z < 0 → z ≤ O) →
  (∀ q : ℚ, q ≠ 0 → |square| ≤ |q|) →
  triangle > 0 →
  O < 0 →
  (square + triangle) * O = -1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_largest_negative_smallest_absolute_l3716_371694


namespace NUMINAMATH_CALUDE_new_weekly_earnings_l3716_371673

-- Define the original weekly earnings
def original_earnings : ℝ := 60

-- Define the percentage increase
def percentage_increase : ℝ := 0.30

-- Theorem to prove the new weekly earnings
theorem new_weekly_earnings :
  original_earnings * (1 + percentage_increase) = 78 := by
  sorry

end NUMINAMATH_CALUDE_new_weekly_earnings_l3716_371673


namespace NUMINAMATH_CALUDE_least_three_digit_7_heavy_l3716_371634

def is_7_heavy (n : ℕ) : Prop := n % 7 > 4

theorem least_three_digit_7_heavy : 
  (∀ m : ℕ, 100 ≤ m ∧ m < 104 → ¬ is_7_heavy m) ∧ 
  is_7_heavy 104 := by
  sorry

end NUMINAMATH_CALUDE_least_three_digit_7_heavy_l3716_371634


namespace NUMINAMATH_CALUDE_ladder_distance_l3716_371683

theorem ladder_distance (c a b : ℝ) : 
  c = 25 → a = 20 → c^2 = a^2 + b^2 → b = 15 :=
by sorry

end NUMINAMATH_CALUDE_ladder_distance_l3716_371683


namespace NUMINAMATH_CALUDE_direction_vector_b_l3716_371618

def point_1 : ℝ × ℝ := (-3, 4)
def point_2 : ℝ × ℝ := (2, -1)

theorem direction_vector_b (b : ℝ) : 
  (∃ (k : ℝ), k ≠ 0 ∧ (point_2.1 - point_1.1, point_2.2 - point_1.2) = (k * b, k * (-1))) → 
  b = 1 := by
sorry

end NUMINAMATH_CALUDE_direction_vector_b_l3716_371618


namespace NUMINAMATH_CALUDE_original_sequence_reappearance_l3716_371650

/-- The cycle length of the letter sequence -/
def letter_cycle_length : ℕ := 8

/-- The cycle length of the digit sequence -/
def digit_cycle_length : ℕ := 5

/-- The line number where the original sequence reappears -/
def reappearance_line : ℕ := 40

theorem original_sequence_reappearance :
  Nat.lcm letter_cycle_length digit_cycle_length = reappearance_line :=
by sorry

end NUMINAMATH_CALUDE_original_sequence_reappearance_l3716_371650


namespace NUMINAMATH_CALUDE_power_exceeds_any_number_l3716_371653

theorem power_exceeds_any_number (p M : ℝ) (hp : p > 0) (hM : M > 0) :
  ∃ n : ℕ, (1 + p)^n > M := by sorry

end NUMINAMATH_CALUDE_power_exceeds_any_number_l3716_371653


namespace NUMINAMATH_CALUDE_direct_square_variation_problem_l3716_371607

/-- A function representing direct variation with the square of x -/
def direct_square_variation (k : ℝ) (x : ℝ) : ℝ := k * x^2

theorem direct_square_variation_problem (y : ℝ → ℝ) :
  (∃ k : ℝ, ∀ x, y x = direct_square_variation k x) →  -- y varies directly as the square of x
  y 3 = 18 →  -- y = 18 when x = 3
  y 6 = 72 :=  -- y = 72 when x = 6
by
  sorry

end NUMINAMATH_CALUDE_direct_square_variation_problem_l3716_371607


namespace NUMINAMATH_CALUDE_pencil_transfer_l3716_371637

/-- Given that Gloria has 2 pencils and Lisa has 99 pencils, 
    if Lisa gives all of her pencils to Gloria, 
    then Gloria will have 101 pencils. -/
theorem pencil_transfer (gloria_initial : ℕ) (lisa_initial : ℕ) 
  (h1 : gloria_initial = 2) 
  (h2 : lisa_initial = 99) : 
  gloria_initial + lisa_initial = 101 := by
  sorry

end NUMINAMATH_CALUDE_pencil_transfer_l3716_371637


namespace NUMINAMATH_CALUDE_parabola_reflection_y_axis_l3716_371640

/-- Represents a parabola in the form y = a(x - h)^2 + k --/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Reflects a parabola along the y-axis --/
def reflect_y (p : Parabola) : Parabola :=
  { a := p.a, h := -p.h, k := p.k }

theorem parabola_reflection_y_axis :
  let original := Parabola.mk 2 1 (-4)
  let reflected := reflect_y original
  reflected = Parabola.mk 2 (-1) (-4) := by sorry

end NUMINAMATH_CALUDE_parabola_reflection_y_axis_l3716_371640


namespace NUMINAMATH_CALUDE_fraction_product_theorem_l3716_371608

theorem fraction_product_theorem : 
  (7 / 4 : ℚ) * (8 / 16 : ℚ) * (21 / 14 : ℚ) * (15 / 25 : ℚ) * 
  (28 / 21 : ℚ) * (20 / 40 : ℚ) * (49 / 28 : ℚ) * (25 / 50 : ℚ) = 147 / 320 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_theorem_l3716_371608


namespace NUMINAMATH_CALUDE_oranges_sold_count_l3716_371667

/-- Given information about oranges on a truck -/
structure OrangeTruck where
  bags : Nat
  oranges_per_bag : Nat
  rotten : Nat
  for_juice : Nat

/-- Calculate the number of oranges to be sold -/
def oranges_to_sell (truck : OrangeTruck) : Nat :=
  truck.bags * truck.oranges_per_bag - (truck.rotten + truck.for_juice)

/-- Theorem stating the number of oranges to be sold -/
theorem oranges_sold_count (truck : OrangeTruck) 
  (h1 : truck.bags = 10)
  (h2 : truck.oranges_per_bag = 30)
  (h3 : truck.rotten = 50)
  (h4 : truck.for_juice = 30) :
  oranges_to_sell truck = 220 := by
  sorry

#eval oranges_to_sell { bags := 10, oranges_per_bag := 30, rotten := 50, for_juice := 30 }

end NUMINAMATH_CALUDE_oranges_sold_count_l3716_371667


namespace NUMINAMATH_CALUDE_simplify_expression_l3716_371625

theorem simplify_expression : (2^8 + 4^5) * (2^3 - (-2)^3)^8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3716_371625


namespace NUMINAMATH_CALUDE_flour_weight_range_l3716_371623

/-- Given a bag of flour labeled as 25 ± 0.02kg, prove that its weight m is within the range 24.98kg ≤ m ≤ 25.02kg -/
theorem flour_weight_range (m : ℝ) (h : |m - 25| ≤ 0.02) : 24.98 ≤ m ∧ m ≤ 25.02 := by
  sorry

end NUMINAMATH_CALUDE_flour_weight_range_l3716_371623


namespace NUMINAMATH_CALUDE_fraction_problem_l3716_371611

theorem fraction_problem (x : ℚ) : x * 8 + 2 = 8 → x = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3716_371611


namespace NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l3716_371686

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def toRational (x : RepeatingDecimal) : ℚ :=
  sorry

/-- The repeating decimal 8.137137137... -/
def x : RepeatingDecimal :=
  { integerPart := 8, repeatingPart := 137 }

theorem repeating_decimal_as_fraction :
  toRational x = 2709 / 333 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l3716_371686


namespace NUMINAMATH_CALUDE_divisibility_of_p_l3716_371649

theorem divisibility_of_p (p q r s : ℕ+) 
  (h1 : Nat.gcd p.val q.val = 40)
  (h2 : Nat.gcd q.val r.val = 50)
  (h3 : Nat.gcd r.val s.val = 75)
  (h4 : 80 < Nat.gcd s.val p.val)
  (h5 : Nat.gcd s.val p.val < 120) :
  5 ∣ p.val := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_p_l3716_371649


namespace NUMINAMATH_CALUDE_abc_value_l3716_371638

theorem abc_value (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a + 1/b = 5)
  (h2 : b + 1/c = 2)
  (h3 : c + 1/a = 9/4) :
  a * b * c = (7 + Real.sqrt 21) / 8 := by
sorry

end NUMINAMATH_CALUDE_abc_value_l3716_371638


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainder_two_l3716_371677

theorem smallest_integer_with_remainder_two : ∃! m : ℕ,
  m > 1 ∧
  m % 13 = 2 ∧
  m % 5 = 2 ∧
  m % 3 = 2 ∧
  ∀ n : ℕ, n > 1 ∧ n % 13 = 2 ∧ n % 5 = 2 ∧ n % 3 = 2 → m ≤ n :=
by
  use 197
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainder_two_l3716_371677


namespace NUMINAMATH_CALUDE_completing_square_result_l3716_371651

theorem completing_square_result (x : ℝ) : 
  (x^2 - 6*x - 8 = 0) ↔ ((x - 3)^2 = 17) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_result_l3716_371651


namespace NUMINAMATH_CALUDE_board_number_theorem_l3716_371621

/-- Represents the state of the numbers on the board -/
structure BoardState where
  a : ℤ
  b : ℤ
  c : ℤ

/-- The operation described in the problem -/
def applyOperation (state : BoardState) : BoardState :=
  ⟨state.a, state.b, state.a + state.b - state.c⟩

/-- Checks if the numbers form an arithmetic sequence with difference 6 -/
def isArithmeticSequence (state : BoardState) : Prop :=
  state.b - state.a = 6 ∧ state.c - state.b = 6

/-- The main theorem to be proved -/
theorem board_number_theorem :
  ∃ (n : ℕ) (finalState : BoardState),
    finalState = (applyOperation^[n] ⟨3, 9, 15⟩) ∧
    isArithmeticSequence finalState ∧
    finalState.a = 2013 ∧
    finalState.b = 2019 ∧
    finalState.c = 2025 := by
  sorry

end NUMINAMATH_CALUDE_board_number_theorem_l3716_371621


namespace NUMINAMATH_CALUDE_systematic_sampling_40th_number_l3716_371682

/-- Given a systematic sample of 50 students from 1000, with the first number drawn being 0015,
    prove that the 40th number drawn is 0795. -/
theorem systematic_sampling_40th_number
  (total_students : Nat)
  (sample_size : Nat)
  (first_number : Nat)
  (h1 : total_students = 1000)
  (h2 : sample_size = 50)
  (h3 : first_number = 15)
  : (first_number + (39 * (total_students / sample_size))) % total_students = 795 := by
  sorry

#eval (15 + (39 * (1000 / 50))) % 1000  -- Should output 795

end NUMINAMATH_CALUDE_systematic_sampling_40th_number_l3716_371682


namespace NUMINAMATH_CALUDE_hidden_digit_problem_l3716_371633

theorem hidden_digit_problem :
  ∃! (x : ℕ), x ≠ 0 ∧ x < 10 ∧ ((10 * x + x) + (10 * x + x) + 1) * x = 100 * x + 10 * x + x :=
by
  sorry

end NUMINAMATH_CALUDE_hidden_digit_problem_l3716_371633


namespace NUMINAMATH_CALUDE_files_remaining_l3716_371624

theorem files_remaining (music_files : ℕ) (video_files : ℕ) (deleted_files : ℕ)
  (h1 : music_files = 27)
  (h2 : video_files = 42)
  (h3 : deleted_files = 11) :
  music_files + video_files - deleted_files = 58 :=
by sorry

end NUMINAMATH_CALUDE_files_remaining_l3716_371624


namespace NUMINAMATH_CALUDE_exists_valid_configuration_l3716_371627

/-- Represents a point on the chessboard -/
structure Point where
  x : Fin 8
  y : Fin 8

/-- Checks if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- A configuration of 16 points on the chessboard -/
def Configuration := Fin 16 → Point

/-- Checks if a configuration is valid (no three points are collinear) -/
def valid_configuration (config : Configuration) : Prop :=
  ∀ i j k, i < j → j < k → ¬collinear (config i) (config j) (config k)

/-- Theorem: There exists a valid configuration of 16 points on an 8x8 chessboard -/
theorem exists_valid_configuration : ∃ (config : Configuration), valid_configuration config := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_configuration_l3716_371627


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l3716_371658

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the line
def line (k : ℝ) (x y : ℝ) : Prop := y - 1 = k * (x + 3)

-- Define the point on the parabola
def point_on_parabola (a : ℝ) : Prop := parabola 3 a ∧ (3 - 2)^2 + a^2 = 5^2

-- Theorem statement
theorem parabola_intersection_theorem (k : ℝ) :
  (∃! p : ℝ × ℝ, parabola p.1 p.2 ∧ line k p.1 p.2) ↔ k = 0 ∨ k = -1 ∨ k = 2/3 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l3716_371658


namespace NUMINAMATH_CALUDE_natural_solutions_count_l3716_371681

theorem natural_solutions_count :
  ∃! (s : Finset (ℕ × ℕ)), s.card = 4 ∧ ∀ (x y : ℕ), (x, y) ∈ s ↔ 2 * x + y = 7 := by
  sorry

end NUMINAMATH_CALUDE_natural_solutions_count_l3716_371681


namespace NUMINAMATH_CALUDE_digits_zeros_equality_l3716_371672

/-- Count the number of digits in a natural number -/
def countDigits (n : ℕ) : ℕ := sorry

/-- Count the number of zeros in a natural number -/
def countZeros (n : ℕ) : ℕ := sorry

/-- Sum of digits in a sequence from 1 to n -/
def sumDigits (n : ℕ) : ℕ := (Finset.range n).sum (λ i => countDigits (i + 1))

/-- Sum of zeros in a sequence from 1 to n -/
def sumZeros (n : ℕ) : ℕ := (Finset.range n).sum (λ i => countZeros (i + 1))

/-- Theorem: For any natural number k, the number of all digits in the sequence
    1, 2, 3, ..., 10^k is equal to the number of all zeros in the sequence
    1, 2, 3, ..., 10^(k+1) -/
theorem digits_zeros_equality (k : ℕ) :
  sumDigits (10^k) = sumZeros (10^(k+1)) := by sorry

end NUMINAMATH_CALUDE_digits_zeros_equality_l3716_371672


namespace NUMINAMATH_CALUDE_determinant_of_cubic_roots_l3716_371664

theorem determinant_of_cubic_roots (s p q : ℝ) (a b c : ℝ) : 
  (a^3 - s*a^2 + p*a + q = 0) →
  (b^3 - s*b^2 + p*b + q = 0) →
  (c^3 - s*c^2 + p*c + q = 0) →
  (a + b + c = s) →
  (a*b + b*c + a*c = p) →
  (a*b*c = -q) →
  Matrix.det !![1 + a, 1, 1; 1, 1 + b, 1; 1, 1, 1 + c] = p + 3*s := by
sorry

end NUMINAMATH_CALUDE_determinant_of_cubic_roots_l3716_371664


namespace NUMINAMATH_CALUDE_polygon_area_is_787_5_l3716_371699

/-- The area of a triangle given its vertices -/
def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

/-- The vertices of the polygon -/
def vertices : List (ℝ × ℝ) :=
  [(0, 0), (15, 0), (45, 30), (45, 45), (30, 45), (0, 15)]

/-- The area of the polygon -/
def polygon_area : ℝ :=
  triangle_area 0 0 15 0 0 15 +
  triangle_area 15 0 45 30 0 15 +
  triangle_area 45 30 45 45 30 45

theorem polygon_area_is_787_5 :
  polygon_area = 787.5 := by
  sorry

end NUMINAMATH_CALUDE_polygon_area_is_787_5_l3716_371699


namespace NUMINAMATH_CALUDE_initial_bananas_count_l3716_371629

/-- The number of bananas Raj has eaten -/
def bananas_eaten : ℕ := 70

/-- The number of bananas left on the tree after Raj cut some -/
def bananas_left_on_tree : ℕ := 100

/-- The number of bananas in Raj's basket -/
def bananas_in_basket : ℕ := 2 * bananas_eaten

/-- The total number of bananas Raj cut from the tree -/
def bananas_cut : ℕ := bananas_eaten + bananas_in_basket

/-- The initial number of bananas on the tree -/
def initial_bananas : ℕ := bananas_cut + bananas_left_on_tree

theorem initial_bananas_count : initial_bananas = 310 := by
  sorry

end NUMINAMATH_CALUDE_initial_bananas_count_l3716_371629


namespace NUMINAMATH_CALUDE_amount_ratio_l3716_371697

def total : ℕ := 1210
def r_amount : ℕ := 400

theorem amount_ratio (p q r : ℕ) 
  (h1 : p + q + r = total)
  (h2 : r = r_amount)
  (h3 : 9 * r = 10 * q) :
  5 * q = 4 * p := by sorry

end NUMINAMATH_CALUDE_amount_ratio_l3716_371697


namespace NUMINAMATH_CALUDE_rectangle_to_square_l3716_371693

theorem rectangle_to_square (x y : ℚ) :
  (x - 5 = y + 2) →
  (x * y = (x - 5) * (y + 2)) →
  (x = 25/3 ∧ y = 4/3) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_to_square_l3716_371693


namespace NUMINAMATH_CALUDE_workshop_production_balance_l3716_371696

theorem workshop_production_balance :
  let total_workers : ℕ := 85
  let type_a_rate : ℕ := 16
  let type_b_rate : ℕ := 10
  let set_a_parts : ℕ := 2
  let set_b_parts : ℕ := 3
  let workers_a : ℕ := 25
  let workers_b : ℕ := 60
  (total_workers = workers_a + workers_b) ∧
  ((type_a_rate * workers_a) / set_a_parts = (type_b_rate * workers_b) / set_b_parts) := by
  sorry

end NUMINAMATH_CALUDE_workshop_production_balance_l3716_371696


namespace NUMINAMATH_CALUDE_parabola_equation_l3716_371641

/-- A parabola passing through points (0, 5) and (3, 2) -/
def Parabola (x y : ℝ) : Prop :=
  ∃ (b c : ℝ), y = x^2 + b*x + c ∧ 5 = c ∧ 2 = 9 + 3*b + c

/-- The specific parabola y = x^2 - 4x + 5 -/
def SpecificParabola (x y : ℝ) : Prop :=
  y = x^2 - 4*x + 5

theorem parabola_equation : ∀ x y : ℝ, Parabola x y ↔ SpecificParabola x y :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l3716_371641


namespace NUMINAMATH_CALUDE_min_value_theorem_l3716_371652

theorem min_value_theorem (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^2 + 1) / (y - 2) + (y^2 + 1) / (x - 2) ≥ 20 ∧
  ((x^2 + 1) / (y - 2) + (y^2 + 1) / (x - 2) = 20 ↔ x = 3 ∧ y = 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3716_371652


namespace NUMINAMATH_CALUDE_pyramid_max_volume_l3716_371626

/-- The maximum volume of a pyramid with given base side lengths and angle constraints. -/
theorem pyramid_max_volume (AB AC : ℝ) (sin_BAC : ℝ) (max_lateral_angle : ℝ) :
  AB = 3 →
  AC = 5 →
  sin_BAC = 3/5 →
  max_lateral_angle = 60 * π / 180 →
  ∃ (V : ℝ), V = (5 * Real.sqrt 174) / 4 ∧ 
    ∀ (V' : ℝ), V' ≤ V := by
  sorry

end NUMINAMATH_CALUDE_pyramid_max_volume_l3716_371626


namespace NUMINAMATH_CALUDE_exam_student_count_l3716_371657

theorem exam_student_count (N : ℕ) (average_all : ℝ) (average_excluded : ℝ) (average_remaining : ℝ) 
  (h1 : average_all = 70)
  (h2 : average_excluded = 50)
  (h3 : average_remaining = 90)
  (h4 : N * average_all = 250 + (N - 5) * average_remaining) :
  N = 10 := by sorry

end NUMINAMATH_CALUDE_exam_student_count_l3716_371657


namespace NUMINAMATH_CALUDE_equation_equivalent_to_line_segments_l3716_371632

def satisfies_equation (x y : ℝ) : Prop :=
  3 * |x - 1| + 2 * |y + 2| = 6

def within_rectangle (x y : ℝ) : Prop :=
  -1 ≤ x ∧ x ≤ 3 ∧ -5 ≤ y ∧ y ≤ 1

def on_line_segments (x y : ℝ) : Prop :=
  (3*x + 2*y = 5 ∨ -3*x + 2*y = -1 ∨ 3*x - 2*y = 13 ∨ -3*x - 2*y = 7) ∧ within_rectangle x y

theorem equation_equivalent_to_line_segments :
  ∀ x y : ℝ, satisfies_equation x y ↔ on_line_segments x y :=
sorry

end NUMINAMATH_CALUDE_equation_equivalent_to_line_segments_l3716_371632


namespace NUMINAMATH_CALUDE_parabola_coefficient_ratio_l3716_371620

/-- A parabola with equation y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: Given two points on a parabola with the same y-coordinate and x-coordinates
    equidistant from x = 1, the ratio a/b of the parabola coefficients is -1/2 -/
theorem parabola_coefficient_ratio 
  (p : Parabola) 
  (A B : Point) 
  (h1 : A.x = -1 ∧ A.y = 2) 
  (h2 : B.x = 3 ∧ B.y = 2) 
  (h3 : A.y = p.a * A.x^2 + p.b * A.x + p.c) 
  (h4 : B.y = p.a * B.x^2 + p.b * B.x + p.c) :
  p.a / p.b = -1/2 := by sorry

end NUMINAMATH_CALUDE_parabola_coefficient_ratio_l3716_371620


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_three_l3716_371642

theorem reciprocal_of_negative_three :
  (1 : ℚ) / (-3 : ℚ) = -1/3 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_three_l3716_371642


namespace NUMINAMATH_CALUDE_unique_root_quadratic_l3716_371670

theorem unique_root_quadratic (k : ℝ) : 
  (∃! a : ℝ, (k^2 - 9) * a^2 - 2 * (k + 1) * a + 1 = 0) → 
  (k = 3 ∨ k = -3 ∨ k = -5) :=
by sorry

end NUMINAMATH_CALUDE_unique_root_quadratic_l3716_371670


namespace NUMINAMATH_CALUDE_fraction_equals_121_l3716_371680

theorem fraction_equals_121 : (1100^2 : ℚ) / (260^2 - 240^2) = 121 := by sorry

end NUMINAMATH_CALUDE_fraction_equals_121_l3716_371680


namespace NUMINAMATH_CALUDE_stock_value_ordering_l3716_371684

def initial_investment : ℝ := 200

def alpha_year1 : ℝ := 1.30
def beta_year1 : ℝ := 0.80
def gamma_year1 : ℝ := 1.10
def delta_year1 : ℝ := 0.90

def alpha_year2 : ℝ := 0.85
def beta_year2 : ℝ := 1.30
def gamma_year2 : ℝ := 0.95
def delta_year2 : ℝ := 1.20

def final_alpha : ℝ := initial_investment * alpha_year1 * alpha_year2
def final_beta : ℝ := initial_investment * beta_year1 * beta_year2
def final_gamma : ℝ := initial_investment * gamma_year1 * gamma_year2
def final_delta : ℝ := initial_investment * delta_year1 * delta_year2

theorem stock_value_ordering :
  final_delta < final_beta ∧ final_beta < final_gamma ∧ final_gamma < final_alpha :=
by sorry

end NUMINAMATH_CALUDE_stock_value_ordering_l3716_371684


namespace NUMINAMATH_CALUDE_factorization_cubic_quadratic_l3716_371698

theorem factorization_cubic_quadratic (x y : ℝ) : x^3*y - 4*x*y = x*y*(x-2)*(x+2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_cubic_quadratic_l3716_371698


namespace NUMINAMATH_CALUDE_equation_solution_range_l3716_371661

theorem equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (x + m) / (x - 3) + (3 * m) / (3 - x) = 3) →
  m < 9 / 2 ∧ m ≠ 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_range_l3716_371661


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3716_371613

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt ((3 / x) + 5) = 5/2 → x = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3716_371613


namespace NUMINAMATH_CALUDE_shaded_area_circles_l3716_371692

/-- The area of the shaded region formed by a larger circle and two smaller circles --/
theorem shaded_area_circles (R : ℝ) (h : R = 8) : 
  let r := R / 2
  let large_circle_area := π * R^2
  let small_circle_area := π * r^2
  large_circle_area - 2 * small_circle_area = 32 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_circles_l3716_371692


namespace NUMINAMATH_CALUDE_oil_price_reduction_60_percent_l3716_371610

/-- The percentage reduction in oil price -/
def oil_price_reduction (original_price reduced_price : ℚ) : ℚ :=
  (original_price - reduced_price) / original_price * 100

/-- The amount of oil that can be bought with a fixed amount of money -/
def oil_amount (price : ℚ) (money : ℚ) : ℚ := money / price

theorem oil_price_reduction_60_percent 
  (reduced_price : ℚ) 
  (additional_amount : ℚ) 
  (fixed_money : ℚ) :
  reduced_price = 30 →
  additional_amount = 10 →
  fixed_money = 1500 →
  oil_amount reduced_price fixed_money = oil_amount reduced_price (fixed_money / 2) + additional_amount →
  oil_price_reduction ((fixed_money / 2) / additional_amount) reduced_price = 60 := by
sorry

end NUMINAMATH_CALUDE_oil_price_reduction_60_percent_l3716_371610


namespace NUMINAMATH_CALUDE_intersection_point_product_l3716_371668

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := y^2 / 16 + x^2 / 9 = 1

-- Define the hyperbola
def is_on_hyperbola (x y : ℝ) : Prop := y^2 / 4 - x^2 / 5 = 1

-- Define the common foci
def common_foci (F1 F2 : ℝ × ℝ) : Prop :=
  ∃ (a b c d : ℝ), 
    a^2 / 16 + b^2 / 9 = 1 ∧ 
    c^2 / 4 - d^2 / 5 = 1 ∧
    F1 = (b, a) ∧ F2 = (-b, -a)

-- Define the point of intersection
def is_intersection_point (P : ℝ × ℝ) : Prop :=
  is_on_ellipse P.1 P.2 ∧ is_on_hyperbola P.1 P.2

-- The theorem
theorem intersection_point_product (F1 F2 P : ℝ × ℝ) :
  common_foci F1 F2 → is_intersection_point P →
  (P.1 - F1.1)^2 + (P.2 - F1.2)^2 * ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) = 144 :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_product_l3716_371668


namespace NUMINAMATH_CALUDE_max_notebooks_purchase_l3716_371604

theorem max_notebooks_purchase (available : ℚ) (cost : ℚ) : 
  available = 12 → cost = 1.25 → 
  ⌊available / cost⌋ = 9 := by sorry

end NUMINAMATH_CALUDE_max_notebooks_purchase_l3716_371604


namespace NUMINAMATH_CALUDE_low_key_function_m_range_l3716_371601

def is_t_degree_low_key (f : ℝ → ℝ) (t : ℝ) (C : Set ℝ) : Prop :=
  ∀ x ∈ C, f (x + t) ≤ f x

def f (m : ℝ) (x : ℝ) : ℝ := -|m * x - 3|

theorem low_key_function_m_range :
  ∀ m : ℝ, (is_t_degree_low_key (f m) 6 (Set.Ici 0)) →
    (m ≤ 0 ∨ m ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_low_key_function_m_range_l3716_371601


namespace NUMINAMATH_CALUDE_parabola_equation_l3716_371616

/-- The equation of a parabola with vertex at the origin and focus at (2, 0) -/
theorem parabola_equation : ∀ x y : ℝ, 
  (∃ p : ℝ, p > 0 ∧ x = p ∧ y = 0) →  -- focus at (p, 0)
  (∀ a b : ℝ, (a - x)^2 + (b - y)^2 = (a - 0)^2 + b^2) →  -- definition of parabola
  y^2 = 4 * 2 * x :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l3716_371616


namespace NUMINAMATH_CALUDE_interest_difference_implies_principal_l3716_371655

/-- Proves that given specific interest conditions, if the difference between compound and simple interest is 36, the principal is 3600. -/
theorem interest_difference_implies_principal : 
  let rate : ℝ := 10  -- Interest rate (%)
  let time : ℝ := 2   -- Time period in years
  let diff : ℝ := 36  -- Difference between compound and simple interest
  ∀ principal : ℝ,
    (principal * (1 + rate / 100) ^ time - principal) -  -- Compound interest
    (principal * rate * time / 100) =                    -- Simple interest
    diff →
    principal = 3600 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_implies_principal_l3716_371655


namespace NUMINAMATH_CALUDE_exists_student_won_all_l3716_371671

/-- Represents a competition --/
def Competition := Fin 44

/-- Represents a student --/
structure Student where
  id : ℕ

/-- The set of students who won a given competition --/
def winners : Competition → Finset Student :=
  sorry

/-- The number of competitions a student has won --/
def wins (s : Student) : ℕ :=
  sorry

/-- Statement: There exists a student who won all competitions --/
theorem exists_student_won_all :
  (∀ c : Competition, (winners c).card = 7) →
  (∀ c₁ c₂ : Competition, c₁ ≠ c₂ → ∃! s : Student, s ∈ winners c₁ ∧ s ∈ winners c₂) →
  ∃ s : Student, ∀ c : Competition, s ∈ winners c :=
sorry

end NUMINAMATH_CALUDE_exists_student_won_all_l3716_371671


namespace NUMINAMATH_CALUDE_range_encoding_l3716_371662

/-- Represents a coding scheme for words -/
structure CodeScheme where
  random : Nat
  rand : Nat

/-- Defines the coding for a word given a CodeScheme -/
def encode (scheme : CodeScheme) (word : String) : Nat :=
  sorry

/-- Theorem: Given the coding scheme where 'random' is 123678 and 'rand' is 1236,
    the code for 'range' is 12378 -/
theorem range_encoding (scheme : CodeScheme)
    (h1 : scheme.random = 123678)
    (h2 : scheme.rand = 1236) :
    encode scheme "range" = 12378 :=
  sorry

end NUMINAMATH_CALUDE_range_encoding_l3716_371662


namespace NUMINAMATH_CALUDE_jameson_medals_l3716_371631

theorem jameson_medals (total_medals track_medals : ℕ) 
  (h1 : total_medals = 20)
  (h2 : track_medals = 5)
  (h3 : ∃ swimming_medals : ℕ, swimming_medals = 2 * track_medals) :
  ∃ badminton_medals : ℕ, badminton_medals = total_medals - (track_medals + 2 * track_medals) ∧ badminton_medals = 5 := by
  sorry

end NUMINAMATH_CALUDE_jameson_medals_l3716_371631


namespace NUMINAMATH_CALUDE_rectangle_circle_union_area_l3716_371630

/-- The area of the union of a rectangle and a circle -/
theorem rectangle_circle_union_area :
  let rectangle_width : ℝ := 8
  let rectangle_height : ℝ := 12
  let circle_radius : ℝ := 8
  let rectangle_area : ℝ := rectangle_width * rectangle_height
  let circle_area : ℝ := π * circle_radius^2
  let overlap_area : ℝ := (1/4) * circle_area
  rectangle_area + circle_area - overlap_area = 96 + 48 * π :=
by sorry

end NUMINAMATH_CALUDE_rectangle_circle_union_area_l3716_371630


namespace NUMINAMATH_CALUDE_probability_two_black_balls_l3716_371645

/-- The probability of drawing two black balls from a box containing 8 white balls and 7 black balls, without replacement. -/
theorem probability_two_black_balls (white_balls black_balls : ℕ) (h1 : white_balls = 8) (h2 : black_balls = 7) :
  (black_balls.choose 2 : ℚ) / ((white_balls + black_balls).choose 2) = 1 / 5 := by
  sorry


end NUMINAMATH_CALUDE_probability_two_black_balls_l3716_371645


namespace NUMINAMATH_CALUDE_exponential_inequality_range_l3716_371609

theorem exponential_inequality_range (x : ℝ) : 
  (2 : ℝ) ^ (2 * x - 7) < (2 : ℝ) ^ (x - 3) → x < 4 := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_range_l3716_371609


namespace NUMINAMATH_CALUDE_zunyi_conference_highest_temp_l3716_371635

/-- Given the lowest temperature and maximum temperature difference of a day,
    calculate the highest temperature of that day. -/
def highest_temperature (lowest_temp max_diff : ℝ) : ℝ :=
  lowest_temp + max_diff

/-- Theorem stating that given the specific conditions of the problem,
    the highest temperature of the day is 22°C. -/
theorem zunyi_conference_highest_temp :
  highest_temperature 18 4 = 22 := by
  sorry

end NUMINAMATH_CALUDE_zunyi_conference_highest_temp_l3716_371635


namespace NUMINAMATH_CALUDE_sum_and_simplest_form_l3716_371695

theorem sum_and_simplest_form :
  ∃ (n d : ℕ), n > 0 ∧ d > 0 ∧ (2 : ℚ) / 3 + (7 : ℚ) / 8 = (n : ℚ) / d ∧ 
  ∀ (n' d' : ℕ), n' > 0 → d' > 0 → (n' : ℚ) / d' = (n : ℚ) / d → n' ≥ n ∧ d' ≥ d :=
by
  sorry

end NUMINAMATH_CALUDE_sum_and_simplest_form_l3716_371695


namespace NUMINAMATH_CALUDE_camel_cost_proof_l3716_371665

/-- The cost of a camel in rupees -/
def camel_cost : ℝ := 5200

/-- The cost of a horse in rupees -/
def horse_cost : ℝ := 2166.67

/-- The cost of an ox in rupees -/
def ox_cost : ℝ := 8666.67

/-- The cost of an elephant in rupees -/
def elephant_cost : ℝ := 13000

theorem camel_cost_proof :
  (10 * camel_cost = 24 * horse_cost) ∧
  (16 * horse_cost = 4 * ox_cost) ∧
  (6 * ox_cost = 4 * elephant_cost) ∧
  (10 * elephant_cost = 130000) →
  camel_cost = 5200 := by
sorry

end NUMINAMATH_CALUDE_camel_cost_proof_l3716_371665


namespace NUMINAMATH_CALUDE_product_of_solutions_l3716_371678

theorem product_of_solutions (x : ℝ) : 
  (|18 / x + 4| = 3) → 
  (∃ y : ℝ, (|18 / y + 4| = 3) ∧ x * y = 324 / 7) :=
sorry

end NUMINAMATH_CALUDE_product_of_solutions_l3716_371678


namespace NUMINAMATH_CALUDE_solve_for_D_l3716_371612

theorem solve_for_D : ∃ D : ℤ, 80 - (5 - (6 + 2 * (7 - 8 - D))) = 89 ∧ D = -5 := by sorry

end NUMINAMATH_CALUDE_solve_for_D_l3716_371612


namespace NUMINAMATH_CALUDE_even_function_inequality_l3716_371617

/-- Given a function f(x) = a^(|x+b|) where a > 0, a ≠ 1, b ∈ ℝ, and f is even, prove f(b-3) < f(a+2) -/
theorem even_function_inequality (a b : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(|x + b|)
  (∀ x, f x = f (-x)) →
  f (b - 3) < f (a + 2) := by
sorry

end NUMINAMATH_CALUDE_even_function_inequality_l3716_371617


namespace NUMINAMATH_CALUDE_girls_in_group_l3716_371614

theorem girls_in_group (n : ℕ) : 
  (4 : ℝ) + n > 0 → -- ensure the total number of students is positive
  (((n + 4) * (n + 3) / 2 - 6) / ((n + 4) * (n + 3) / 2) = 5 / 6) →
  n = 5 := by
  sorry


end NUMINAMATH_CALUDE_girls_in_group_l3716_371614


namespace NUMINAMATH_CALUDE_vector_at_negative_two_l3716_371685

/-- A parameterized line in 2D space -/
structure ParameterizedLine where
  /-- The vector on the line at parameter t -/
  vector_at : ℝ → ℝ × ℝ

/-- Theorem: Given a parameterized line with specific points, the vector at t = -2 can be determined -/
theorem vector_at_negative_two
  (line : ParameterizedLine)
  (h1 : line.vector_at 1 = (2, 5))
  (h4 : line.vector_at 4 = (8, -7)) :
  line.vector_at (-2) = (-4, 17) := by
  sorry

end NUMINAMATH_CALUDE_vector_at_negative_two_l3716_371685


namespace NUMINAMATH_CALUDE_car_rental_cost_l3716_371605

theorem car_rental_cost (gas_gallons : ℕ) (gas_price : ℚ) (mile_cost : ℚ) (miles_driven : ℕ) (total_cost : ℚ) :
  gas_gallons = 8 →
  gas_price = 7/2 →
  mile_cost = 1/2 →
  miles_driven = 320 →
  total_cost = 338 →
  (total_cost - (↑gas_gallons * gas_price + ↑miles_driven * mile_cost) : ℚ) = 150 := by
sorry

end NUMINAMATH_CALUDE_car_rental_cost_l3716_371605


namespace NUMINAMATH_CALUDE_trim_length_calculation_oliver_trim_purchase_l3716_371602

theorem trim_length_calculation (table_area : Real) (pi_approx : Real) (extra_trim : Real) : Real :=
  let radius := Real.sqrt (table_area / pi_approx)
  let circumference := 2 * pi_approx * radius
  circumference + extra_trim

theorem oliver_trim_purchase :
  trim_length_calculation 616 (22/7) 5 = 93 :=
by sorry

end NUMINAMATH_CALUDE_trim_length_calculation_oliver_trim_purchase_l3716_371602


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_example_l3716_371689

def is_simplest_quadratic_radical (n : ℝ) : Prop :=
  ∃ (a : ℕ), n = a ∧ ¬∃ (b : ℕ), b * b = a ∧ b > 1

theorem simplest_quadratic_radical_example : 
  ∃ (x : ℝ), is_simplest_quadratic_radical (x + 3) ∧ x = 2 := by
  sorry

#check simplest_quadratic_radical_example

end NUMINAMATH_CALUDE_simplest_quadratic_radical_example_l3716_371689


namespace NUMINAMATH_CALUDE_smallest_z_l3716_371628

theorem smallest_z (x y z : ℤ) : 
  x < y → y < z → 
  (2 * y = x + z) →  -- arithmetic progression
  (z * z = x * y) →  -- geometric progression
  (∀ w : ℤ, (∃ a b c : ℤ, a < b ∧ b < w ∧ 2 * b = a + w ∧ w * w = a * b) → w ≥ z) →
  z = 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_z_l3716_371628


namespace NUMINAMATH_CALUDE_cindys_calculation_l3716_371603

theorem cindys_calculation (x : ℝ) : (x - 10) / 5 = 50 → (x - 5) / 10 = 25.5 := by
  sorry

end NUMINAMATH_CALUDE_cindys_calculation_l3716_371603


namespace NUMINAMATH_CALUDE_max_sections_five_lines_l3716_371691

/- Define a function that calculates the maximum number of sections -/
def max_sections (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else 2 + (n - 1) * n / 2

/- Theorem statement -/
theorem max_sections_five_lines :
  max_sections 5 = 16 :=
by sorry

end NUMINAMATH_CALUDE_max_sections_five_lines_l3716_371691
