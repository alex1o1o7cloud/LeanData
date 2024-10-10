import Mathlib

namespace smallest_square_area_l3609_360904

/-- Represents a rectangle with given width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the area of a square given its side length -/
def square_area (side : ℕ) : ℕ := side * side

/-- Checks if two rectangles can fit side by side within a given length -/
def can_fit_side_by_side (r1 r2 : Rectangle) (length : ℕ) : Prop :=
  r1.width + r2.width ≤ length ∨ r1.height + r2.height ≤ length

/-- Theorem: The smallest possible area of a square containing a 2×3 rectangle and a 4×5 rectangle
    without overlapping and with parallel sides is 49 square units -/
theorem smallest_square_area : ∃ (side : ℕ),
  let r1 : Rectangle := ⟨2, 3⟩
  let r2 : Rectangle := ⟨4, 5⟩
  (∀ (s : ℕ), s < side → ¬(can_fit_side_by_side r1 r2 s)) ∧
  (can_fit_side_by_side r1 r2 side) ∧
  (square_area side = 49) :=
sorry

end smallest_square_area_l3609_360904


namespace nut_raisin_mixture_l3609_360993

/-- The number of pounds of nuts mixed with raisins -/
def pounds_of_nuts : ℝ := 4

/-- The number of pounds of raisins -/
def pounds_of_raisins : ℝ := 3

/-- The ratio of the cost of nuts to the cost of raisins -/
def cost_ratio : ℝ := 4

/-- The ratio of the cost of raisins to the total cost of the mixture -/
def raisin_cost_ratio : ℝ := 0.15789473684210525

theorem nut_raisin_mixture :
  let total_cost := pounds_of_raisins + cost_ratio * pounds_of_nuts
  raisin_cost_ratio * total_cost = pounds_of_raisins :=
by sorry

end nut_raisin_mixture_l3609_360993


namespace extreme_values_and_zero_condition_l3609_360994

/-- The cubic function f(x) = x^3 - x^2 - x - a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x^2 - x - a

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 2*x - 1

theorem extreme_values_and_zero_condition (a : ℝ) :
  (∃ x_max : ℝ, f a x_max = 5/27 - a ∧ ∀ x, f a x ≤ f a x_max) ∧
  (∃ x_min : ℝ, f a x_min = -1 - a ∧ ∀ x, f a x ≥ f a x_min) ∧
  (∃! x, f a x = 0) ↔ (a < -1 ∨ a > 5/27) :=
sorry

end extreme_values_and_zero_condition_l3609_360994


namespace least_common_solution_l3609_360973

theorem least_common_solution : ∃ b : ℕ, b > 0 ∧ 
  b % 7 = 6 ∧ 
  b % 11 = 10 ∧ 
  b % 13 = 12 ∧
  (∀ c : ℕ, c > 0 ∧ c % 7 = 6 ∧ c % 11 = 10 ∧ c % 13 = 12 → b ≤ c) ∧
  b = 1000 := by
  sorry

end least_common_solution_l3609_360973


namespace fraction_increase_l3609_360991

theorem fraction_increase (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (2*x * 2*y) / (2*x + 2*y) = 2 * (x*y / (x + y)) :=
sorry

end fraction_increase_l3609_360991


namespace infinite_solutions_iff_in_solution_set_l3609_360902

/-- A system of two linear equations in two variables with parameters a and b -/
structure LinearSystem (a b : ℝ) where
  eq1 : ∀ x y : ℝ, 3 * (a + b) * x + 12 * y = a
  eq2 : ∀ x y : ℝ, 4 * b * x + (a + b) * b * y = 1

/-- The condition for the system to have infinitely many solutions -/
def HasInfinitelySolutions (a b : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ 3 * (a + b) = 4 * b * k ∧ 12 = (a + b) * b * k ∧ a = k

/-- The set of pairs (a, b) that satisfy the condition -/
def SolutionSet : Set (ℝ × ℝ) :=
  {(1, 3), (3, 1), (-2 - Real.sqrt 7, Real.sqrt 7 - 2), (Real.sqrt 7 - 2, -2 - Real.sqrt 7)}

/-- The main theorem stating the equivalence -/
theorem infinite_solutions_iff_in_solution_set (a b : ℝ) :
  HasInfinitelySolutions a b ↔ (a, b) ∈ SolutionSet := by sorry

end infinite_solutions_iff_in_solution_set_l3609_360902


namespace tank_full_time_l3609_360933

/-- Represents the state of a water tank system -/
structure TankSystem where
  capacity : ℕ
  fill_rate_a : ℕ
  fill_rate_b : ℕ
  drain_rate : ℕ

/-- Calculates the time required to fill the tank -/
def time_to_fill (system : TankSystem) : ℕ :=
  let net_fill_per_cycle := system.fill_rate_a + system.fill_rate_b - system.drain_rate
  let cycles := system.capacity / net_fill_per_cycle
  cycles * 3 - 1

/-- Theorem stating that the tank will be full in 56 minutes -/
theorem tank_full_time (system : TankSystem) 
    (h1 : system.capacity = 950)
    (h2 : system.fill_rate_a = 40)
    (h3 : system.fill_rate_b = 30)
    (h4 : system.drain_rate = 20) :
  time_to_fill system = 56 := by
  sorry

#eval time_to_fill { capacity := 950, fill_rate_a := 40, fill_rate_b := 30, drain_rate := 20 }

end tank_full_time_l3609_360933


namespace nancy_album_pictures_l3609_360964

theorem nancy_album_pictures (total : ℕ) (num_albums : ℕ) (pics_per_album : ℕ) 
  (h1 : total = 51)
  (h2 : num_albums = 8)
  (h3 : pics_per_album = 5) :
  total - (num_albums * pics_per_album) = 11 := by
sorry

end nancy_album_pictures_l3609_360964


namespace integer_root_pairs_l3609_360939

/-- A function that checks if all roots of a quadratic polynomial ax^2 + bx + c are integers -/
def allRootsInteger (a b c : ℤ) : Prop :=
  ∃ x y : ℤ, a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ x ≠ y

/-- The main theorem stating the only valid pairs (p,q) -/
theorem integer_root_pairs :
  ∀ p q : ℤ,
    (allRootsInteger 1 p q ∧ allRootsInteger 1 q p) ↔
    ((p = 4 ∧ q = 4) ∨ (p = 9 ∧ q = 8) ∨ (p = 8 ∧ q = 9)) :=
by sorry

end integer_root_pairs_l3609_360939


namespace distance_is_correct_l3609_360961

def point : ℝ × ℝ × ℝ := (2, 3, 4)
def line_point : ℝ × ℝ × ℝ := (4, 5, 6)
def line_direction : ℝ × ℝ × ℝ := (4, 1, -1)

def distance_to_line (p : ℝ × ℝ × ℝ) (l_point : ℝ × ℝ × ℝ) (l_direction : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_is_correct : 
  distance_to_line point line_point line_direction = (9 * Real.sqrt 2) / 4 :=
sorry

end distance_is_correct_l3609_360961


namespace lesser_number_proof_l3609_360977

theorem lesser_number_proof (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : 
  min x y = 25 := by
sorry

end lesser_number_proof_l3609_360977


namespace gray_eyed_black_haired_count_l3609_360915

theorem gray_eyed_black_haired_count : ∀ (total red_haired black_haired green_eyed gray_eyed green_eyed_red_haired : ℕ),
  total = 60 →
  red_haired + black_haired = total →
  green_eyed + gray_eyed = total →
  green_eyed_red_haired = 20 →
  black_haired = 40 →
  gray_eyed = 25 →
  gray_eyed - (red_haired - green_eyed_red_haired) = 25 := by
  sorry

end gray_eyed_black_haired_count_l3609_360915


namespace ara_height_is_60_l3609_360958

/-- Represents the heights of Shea and Ara --/
structure Heights where
  initial : ℝ  -- Initial height of both Shea and Ara
  shea_current : ℝ  -- Shea's current height
  shea_growth_rate : ℝ  -- Shea's growth rate as a decimal
  ara_growth_difference : ℝ  -- Difference between Shea and Ara's growth in inches

/-- Calculates Ara's current height given the initial conditions --/
def ara_current_height (h : Heights) : ℝ :=
  h.initial + (h.shea_current - h.initial) - h.ara_growth_difference

/-- Theorem stating that Ara's current height is 60 inches --/
theorem ara_height_is_60 (h : Heights)
  (h_shea_current : h.shea_current = 65)
  (h_shea_growth : h.shea_growth_rate = 0.3)
  (h_ara_diff : h.ara_growth_difference = 5) :
  ara_current_height h = 60 := by
  sorry

#eval ara_current_height { initial := 50, shea_current := 65, shea_growth_rate := 0.3, ara_growth_difference := 5 }

end ara_height_is_60_l3609_360958


namespace min_distinct_values_l3609_360918

/-- Represents a list of positive integers -/
def IntegerList := List Nat

/-- Checks if a given value is the unique mode of the list occurring exactly n times -/
def isUniqueMode (list : IntegerList) (mode : Nat) (n : Nat) : Prop :=
  (list.count mode = n) ∧ 
  ∀ x, x ≠ mode → list.count x < n

/-- Theorem: The minimum number of distinct values in a list of 2018 positive integers
    with a unique mode occurring exactly 10 times is 225 -/
theorem min_distinct_values (list : IntegerList) (mode : Nat) :
  list.length = 2018 →
  isUniqueMode list mode 10 →
  list.toFinset.card ≥ 225 :=
sorry

end min_distinct_values_l3609_360918


namespace propositions_true_l3609_360999

theorem propositions_true :
  (∀ a b c : ℝ, c ≠ 0 → a * c^2 > b * c^2 → a > b) ∧
  (∀ a : ℝ, 1 / a > 1 → 0 < a ∧ a < 1) :=
by sorry

end propositions_true_l3609_360999


namespace even_sum_difference_l3609_360912

def sum_even_range (a b : ℕ) : ℕ :=
  let n := (b - a) / 2 + 1
  n * (a + b) / 2

theorem even_sum_difference : sum_even_range 62 110 - sum_even_range 42 90 = 500 := by
  sorry

end even_sum_difference_l3609_360912


namespace all_red_final_state_l3609_360949

/-- Represents the possible colors of chameleons -/
inductive Color
  | Yellow
  | Green
  | Red

/-- Represents the state of chameleons on the island -/
structure ChameleonState where
  yellow : ℕ
  green : ℕ
  red : ℕ

/-- The initial state of chameleons on the island -/
def initial_state : ChameleonState :=
  { yellow := 7, green := 10, red := 17 }

/-- The total number of chameleons on the island -/
def total_chameleons : ℕ := 34

/-- Represents a meeting between two chameleons of different colors -/
def meet (c1 c2 : Color) : Color :=
  match c1, c2 with
  | Color.Yellow, Color.Green => Color.Red
  | Color.Yellow, Color.Red => Color.Green
  | Color.Green, Color.Red => Color.Yellow
  | Color.Green, Color.Yellow => Color.Red
  | Color.Red, Color.Yellow => Color.Green
  | Color.Red, Color.Green => Color.Yellow
  | _, _ => c1  -- If same color, no change

/-- The invariant quantity Delta -/
def Delta (state : ChameleonState) : ℤ :=
  state.red - state.green

/-- Theorem: The only possible final state is all chameleons being red -/
theorem all_red_final_state :
  ∀ (final_state : ChameleonState),
    final_state.yellow + final_state.green + final_state.red = total_chameleons →
    (final_state.yellow = 0 ∧ final_state.green = 0 ∧ final_state.red = total_chameleons) ∨
    (final_state.yellow ≠ 0 ∨ final_state.green ≠ 0 ∨ final_state.red ≠ total_chameleons) :=
sorry


end all_red_final_state_l3609_360949


namespace determinant_zero_implies_y_eq_neg_b_l3609_360984

variable (b y : ℝ)

def matrix (b y : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![y + b, y, y],
    ![y, y + b, y],
    ![y, y, y + b]]

theorem determinant_zero_implies_y_eq_neg_b (h1 : b ≠ 0) 
  (h2 : Matrix.det (matrix b y) = 0) : y = -b := by
  sorry

end determinant_zero_implies_y_eq_neg_b_l3609_360984


namespace sum_two_smallest_prime_factors_l3609_360997

def number : ℕ := 264

-- Define a function to get the prime factors of a number
def prime_factors (n : ℕ) : List ℕ := sorry

-- Define a function to get the two smallest elements from a list
def two_smallest (l : List ℕ) : List ℕ := sorry

theorem sum_two_smallest_prime_factors :
  (two_smallest (prime_factors number)).sum = 5 := by sorry

end sum_two_smallest_prime_factors_l3609_360997


namespace overall_score_calculation_l3609_360980

theorem overall_score_calculation (score1 score2 score3 : ℚ) 
  (problems1 problems2 problems3 : ℕ) : 
  score1 = 60 / 100 →
  score2 = 75 / 100 →
  score3 = 85 / 100 →
  problems1 = 15 →
  problems2 = 25 →
  problems3 = 20 →
  (score1 * problems1 + score2 * problems2 + score3 * problems3) / 
  (problems1 + problems2 + problems3 : ℚ) = 75 / 100 := by
  sorry

end overall_score_calculation_l3609_360980


namespace remainder_b_sixth_l3609_360982

theorem remainder_b_sixth (n : ℕ+) (b : ℤ) (h : b^3 ≡ 1 [ZMOD n]) : b^6 ≡ 1 [ZMOD n] := by
  sorry

end remainder_b_sixth_l3609_360982


namespace jimmy_flour_amount_l3609_360963

/-- The amount of flour Jimmy bought initially -/
def initial_flour (working_hours : ℕ) (minutes_per_pizza : ℕ) (flour_per_pizza : ℚ) (leftover_pizzas : ℕ) : ℚ :=
  let pizzas_per_hour : ℕ := 60 / minutes_per_pizza
  let total_pizzas : ℕ := working_hours * pizzas_per_hour + leftover_pizzas
  total_pizzas * flour_per_pizza

/-- Theorem stating that Jimmy bought 22 kg of flour initially -/
theorem jimmy_flour_amount :
  initial_flour 7 10 (1/2) 2 = 22 := by
  sorry

end jimmy_flour_amount_l3609_360963


namespace lamp_arrangement_l3609_360929

theorem lamp_arrangement (n : ℕ) (k : ℕ) (h : n = 6 ∧ k = 2) :
  (Finset.range (n - k + 1)).card.choose k = 10 := by
  sorry

end lamp_arrangement_l3609_360929


namespace polynomial_factorization_l3609_360935

theorem polynomial_factorization (a : ℝ) : 
  49 * a^3 + 245 * a^2 + 588 * a = 49 * a * (a^2 + 5 * a + 12) := by
  sorry

end polynomial_factorization_l3609_360935


namespace inequality_proof_l3609_360907

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a * b + b * c + c * a = 1) : 
  (1 / (a + b) + 1 / (b + c) + 1 / (c + a)) ≥ 
  Real.sqrt 3 + (a * b / (a + b) + b * c / (b + c) + c * a / (c + a)) := by
  sorry

end inequality_proof_l3609_360907


namespace num_technicians_is_eight_l3609_360914

/-- Represents the number of technicians in the workshop -/
def num_technicians : ℕ := sorry

/-- Represents the total number of workers in the workshop -/
def total_workers : ℕ := 24

/-- Represents the average salary of all workers -/
def avg_salary_all : ℕ := 8000

/-- Represents the average salary of technicians -/
def avg_salary_technicians : ℕ := 12000

/-- Represents the average salary of non-technician workers -/
def avg_salary_others : ℕ := 6000

/-- Theorem stating that the number of technicians is 8 given the workshop conditions -/
theorem num_technicians_is_eight :
  num_technicians = 8 ∧
  num_technicians + (total_workers - num_technicians) = total_workers ∧
  num_technicians * avg_salary_technicians +
    (total_workers - num_technicians) * avg_salary_others =
    total_workers * avg_salary_all :=
by sorry

end num_technicians_is_eight_l3609_360914


namespace regular_polygon_sides_l3609_360906

theorem regular_polygon_sides (n : ℕ) (h1 : n > 2) : 
  (∀ angle : ℝ, angle = 156 ∧ (180 * (n - 2) : ℝ) = n * angle) → n = 15 := by
  sorry

end regular_polygon_sides_l3609_360906


namespace lcm_of_36_and_154_l3609_360966

theorem lcm_of_36_and_154 :
  let a := 36
  let b := 154
  let hcf := 14
  hcf = Nat.gcd a b →
  Nat.lcm a b = 396 := by
sorry

end lcm_of_36_and_154_l3609_360966


namespace unique_n_for_total_digits_l3609_360916

/-- Sum of digits function for a natural number -/
def sumOfDigits (k : ℕ) : ℕ := sorry

/-- Total sum of digits for all numbers from 1 to n -/
def totalSumOfDigits (n : ℕ) : ℕ := 
  (Finset.range n).sum (fun i => sumOfDigits (i + 1))

/-- The theorem statement -/
theorem unique_n_for_total_digits : 
  ∃! n : ℕ, totalSumOfDigits n = 777 := by sorry

end unique_n_for_total_digits_l3609_360916


namespace constant_distance_to_line_l3609_360962

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := y^2 / 2 + x^2 = 1

-- Define the related circle E
def related_circle_E (x y : ℝ) : Prop := x^2 + y^2 = 2/3

-- Define the line l
def line_l (k m x y : ℝ) : Prop := y = k * x + m

-- Theorem statement
theorem constant_distance_to_line
  (k m x1 y1 x2 y2 : ℝ)
  (h1 : ellipse_C x1 y1)
  (h2 : ellipse_C x2 y2)
  (h3 : line_l k m x1 y1)
  (h4 : line_l k m x2 y2)
  (h5 : ∃ (x y : ℝ), related_circle_E x y ∧ line_l k m x y) :
  ∃ (d : ℝ), d = Real.sqrt 6 / 3 ∧
  (∀ (x y : ℝ), line_l k m x y → (x^2 + y^2 = d^2)) :=
sorry

end constant_distance_to_line_l3609_360962


namespace regular_polygon_sides_l3609_360983

/-- A regular polygon with an exterior angle of 10 degrees has 36 sides. -/
theorem regular_polygon_sides (n : ℕ) : n > 0 → (360 : ℝ) / n = 10 → n = 36 := by
  sorry

end regular_polygon_sides_l3609_360983


namespace inscribed_squares_ratio_l3609_360950

/-- A right triangle with sides 5, 12, and 13 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 5
  hb : b = 12
  hc : c = 13
  right_angle : a^2 + b^2 = c^2

/-- A square inscribed in the triangle with one side on the leg of length 5 -/
def inscribed_square_on_leg (t : RightTriangle) (x : ℝ) : Prop :=
  x = t.a

/-- A square inscribed in the triangle with one side on the hypotenuse -/
def inscribed_square_on_hypotenuse (t : RightTriangle) (y : ℝ) : Prop :=
  y / t.c = (t.b - 2*y) / t.b ∧ y / t.c = (t.a - y) / t.a

theorem inscribed_squares_ratio (t : RightTriangle) (x y : ℝ) 
  (hx : inscribed_square_on_leg t x) (hy : inscribed_square_on_hypotenuse t y) : 
  x / y = 18 / 13 := by
  sorry

end inscribed_squares_ratio_l3609_360950


namespace x_plus_y_value_l3609_360951

theorem x_plus_y_value (x y : ℝ) (h : (x - 1)^2 + |2*y + 1| = 0) : x + y = 1/2 := by
  sorry

end x_plus_y_value_l3609_360951


namespace launderette_machines_l3609_360979

/-- Represents the number of quarters in each machine -/
def quarters_per_machine : ℕ := 80

/-- Represents the number of dimes in each machine -/
def dimes_per_machine : ℕ := 100

/-- Represents the total amount of money after emptying all machines (in cents) -/
def total_amount : ℕ := 9000  -- $90 in cents

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Theorem stating that the number of machines in the launderette is 3 -/
theorem launderette_machines : 
  ∃ (n : ℕ), n * (quarters_per_machine * quarter_value + dimes_per_machine * dime_value) = total_amount ∧ n = 3 :=
by sorry

end launderette_machines_l3609_360979


namespace wider_can_radius_l3609_360913

/-- Given two cylindrical cans with the same volume, where the height of one can is double 
    the height of the other, and the radius of the narrower can is 8 units, 
    the radius of the wider can is 8√2 units. -/
theorem wider_can_radius (h : ℝ) (r : ℝ) (h_pos : h > 0) : 
  π * 8^2 * (2*h) = π * r^2 * h → r = 8 * Real.sqrt 2 := by
  sorry

end wider_can_radius_l3609_360913


namespace line_l_theorem_l3609_360926

/-- Definition of line l -/
def line_l (a : ℝ) (x y : ℝ) : Prop := (a + 1) * x + y + 2 - a = 0

/-- Intercepts are equal -/
def equal_intercepts (a : ℝ) : Prop :=
  ∃ k, k = a - 2 ∧ k = (a - 2) / (a + 1)

/-- Line does not pass through second quadrant -/
def not_in_second_quadrant (a : ℝ) : Prop :=
  -(a + 1) ≥ 0 ∧ a - 2 ≤ 0

theorem line_l_theorem :
  (∀ a : ℝ, equal_intercepts a → (a = 2 ∨ a = 0)) ∧
  (∀ a : ℝ, not_in_second_quadrant a ↔ a ≤ -1) :=
sorry

end line_l_theorem_l3609_360926


namespace boat_speed_in_still_water_l3609_360928

/-- The speed of a boat in still water, given its downstream and upstream distances in one hour -/
theorem boat_speed_in_still_water 
  (downstream_distance : ℝ) 
  (upstream_distance : ℝ) 
  (h1 : downstream_distance = 7) 
  (h2 : upstream_distance = 5) : 
  ∃ (boat_speed stream_speed : ℝ), 
    boat_speed + stream_speed = downstream_distance ∧ 
    boat_speed - stream_speed = upstream_distance ∧
    boat_speed = 6 :=
by sorry

end boat_speed_in_still_water_l3609_360928


namespace cube_sum_and_reciprocal_l3609_360938

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = -3) : x^3 + 1/x^3 = -30 := by
  sorry

end cube_sum_and_reciprocal_l3609_360938


namespace quadratic_factorization_l3609_360967

theorem quadratic_factorization (x : ℂ) : 
  2 * x^2 - 4 * x + 5 = (Real.sqrt 2 * x - Real.sqrt 2 + Complex.I * Real.sqrt 3) * 
                        (Real.sqrt 2 * x - Real.sqrt 2 - Complex.I * Real.sqrt 3) := by
  sorry

end quadratic_factorization_l3609_360967


namespace max_value_of_sum_of_roots_l3609_360932

/-- Given positive numbers a, b, c, d with b < d, 
    the maximum value of y = a√(x - b) + c√(d - x) is √((d-b)(a²+c²)) -/
theorem max_value_of_sum_of_roots (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (hbd : b < d) :
  (∀ x, b ≤ x ∧ x ≤ d → a * Real.sqrt (x - b) + c * Real.sqrt (d - x) ≤ Real.sqrt ((d - b) * (a^2 + c^2))) ∧
  (∃ x, b < x ∧ x < d ∧ a * Real.sqrt (x - b) + c * Real.sqrt (d - x) = Real.sqrt ((d - b) * (a^2 + c^2))) :=
by sorry

end max_value_of_sum_of_roots_l3609_360932


namespace oxygen_weight_in_compound_l3609_360930

/-- The atomic weight of hydrogen -/
def hydrogen_weight : ℝ := 1

/-- The atomic weight of chlorine -/
def chlorine_weight : ℝ := 35.5

/-- The total molecular weight of the compound -/
def total_weight : ℝ := 68

/-- The number of hydrogen atoms in the compound -/
def hydrogen_count : ℕ := 1

/-- The number of chlorine atoms in the compound -/
def chlorine_count : ℕ := 1

/-- The number of oxygen atoms in the compound -/
def oxygen_count : ℕ := 2

/-- Theorem: The atomic weight of oxygen in the given compound is 15.75 -/
theorem oxygen_weight_in_compound : 
  ∃ (oxygen_weight : ℝ), 
    (hydrogen_count : ℝ) * hydrogen_weight + 
    (chlorine_count : ℝ) * chlorine_weight + 
    (oxygen_count : ℝ) * oxygen_weight = total_weight ∧ 
    oxygen_weight = 15.75 := by sorry

end oxygen_weight_in_compound_l3609_360930


namespace expression_equality_l3609_360978

theorem expression_equality : 
  (-(-2) + (1 + Real.pi) ^ 0 - |1 - Real.sqrt 2| + Real.sqrt 8 - Real.cos (45 * π / 180)) = 
  2 + 5 / Real.sqrt 2 := by
  sorry

end expression_equality_l3609_360978


namespace xy_zero_necessary_not_sufficient_l3609_360989

theorem xy_zero_necessary_not_sufficient :
  (∀ x y : ℝ, x^2 + y^2 = 0 → x * y = 0) ∧
  (∃ x y : ℝ, x * y = 0 ∧ x^2 + y^2 ≠ 0) :=
by sorry

end xy_zero_necessary_not_sufficient_l3609_360989


namespace arithmetic_sequence_ratio_l3609_360996

/-- An arithmetic sequence with sum of first n terms S_n -/
structure ArithmeticSequence where
  S : ℕ → ℚ  -- Sum function
  is_arithmetic : ∀ n : ℕ, S (n + 1) - S n = S (n + 2) - S (n + 1)

/-- Theorem: If S_2 / S_4 = 1/3, then S_4 / S_8 = 3/10 for an arithmetic sequence -/
theorem arithmetic_sequence_ratio (seq : ArithmeticSequence) 
  (h : seq.S 2 / seq.S 4 = 1 / 3) : 
  seq.S 4 / seq.S 8 = 3 / 10 := by
  sorry

end arithmetic_sequence_ratio_l3609_360996


namespace remaining_area_after_cutting_triangles_l3609_360919

/-- The area of a square with side length n -/
def square_area (n : ℕ) : ℕ := n * n

/-- The area of a rectangle with width w and height h -/
def rectangle_area (w h : ℕ) : ℕ := w * h

theorem remaining_area_after_cutting_triangles :
  let total_area := square_area 6
  let dark_gray_area := rectangle_area 1 3
  let light_gray_area := rectangle_area 2 3
  total_area - (dark_gray_area + light_gray_area) = 27 := by
  sorry

end remaining_area_after_cutting_triangles_l3609_360919


namespace sara_spent_calculation_l3609_360992

/-- Calculates the total amount Sara spent on movies and snacks -/
def sara_total_spent (ticket_price : ℝ) (num_tickets : ℕ) (student_discount : ℝ) 
  (rented_movie_price : ℝ) (purchased_movie_price : ℝ) (snacks_price : ℝ) (sales_tax_rate : ℝ) : ℝ :=
  let discounted_tickets := ticket_price * num_tickets * (1 - student_discount)
  let taxable_items := discounted_tickets + rented_movie_price + purchased_movie_price
  let sales_tax := taxable_items * sales_tax_rate
  discounted_tickets + rented_movie_price + purchased_movie_price + sales_tax + snacks_price

/-- Theorem stating that Sara's total spent is $43.89 -/
theorem sara_spent_calculation : 
  sara_total_spent 10.62 2 0.1 1.59 13.95 7.50 0.05 = 43.89 := by
  sorry

#eval sara_total_spent 10.62 2 0.1 1.59 13.95 7.50 0.05

end sara_spent_calculation_l3609_360992


namespace f_strictly_increasing_l3609_360945

open Real

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_strictly_increasing :
  (∀ x > 0, x^2 * (deriv f x) + 2*x * f x = exp x / x) →
  f 2 = exp 2 / 8 →
  StrictMono f := by sorry

end f_strictly_increasing_l3609_360945


namespace inequality_proof_l3609_360931

def f (a x : ℝ) : ℝ := |x - a|

theorem inequality_proof (a s t : ℝ) (h1 : ∀ x, f a x ≤ 4 ↔ -1 ≤ x ∧ x ≤ 7) 
    (h2 : s > 0) (h3 : t > 0) (h4 : 2*s + t = a) : 
    1/s + 8/t ≥ 6 := by
  sorry

end inequality_proof_l3609_360931


namespace max_xy_value_min_inverse_sum_l3609_360952

-- Part 1
theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 * x + 2 * y = 12) :
  xy ≤ 3 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 3 * x + 2 * y = 12 ∧ x * y = 3 :=
sorry

-- Part 2
theorem min_inverse_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = 3) :
  1 / x + 1 / y ≥ 1 + 2 * Real.sqrt 2 / 3 ∧
  ∃ x y, x > 0 ∧ y > 0 ∧ x + 2 * y = 3 ∧ 1 / x + 1 / y = 1 + 2 * Real.sqrt 2 / 3 :=
sorry

end max_xy_value_min_inverse_sum_l3609_360952


namespace ticket_sales_total_l3609_360908

/-- Calculates the total money collected from ticket sales -/
def total_money_collected (advanced_price : ℕ) (door_price : ℕ) (total_tickets : ℕ) (advanced_tickets : ℕ) : ℕ :=
  advanced_price * advanced_tickets + door_price * (total_tickets - advanced_tickets)

/-- Proves that the total money collected is $1360 given the ticket prices and quantities -/
theorem ticket_sales_total : total_money_collected 8 14 140 100 = 1360 := by
  sorry

#eval total_money_collected 8 14 140 100

end ticket_sales_total_l3609_360908


namespace sum_below_threshold_equals_14_tenths_l3609_360921

def numbers : List ℚ := [14/10, 9/10, 12/10, 5/10, 13/10]
def threshold : ℚ := 11/10

def sum_below_threshold (nums : List ℚ) (t : ℚ) : ℚ :=
  (nums.filter (· ≤ t)).sum

theorem sum_below_threshold_equals_14_tenths :
  sum_below_threshold numbers threshold = 14/10 := by
  sorry

end sum_below_threshold_equals_14_tenths_l3609_360921


namespace population_ratio_l3609_360936

-- Define the populations of cities X, Y, and Z
variable (X Y Z : ℝ)

-- Condition 1: City X has a population 3 times as great as the population of City Y
def condition1 : Prop := X = 3 * Y

-- Condition 2: The ratio of the population of City X to the population of City Z is 6
def condition2 : Prop := X / Z = 6

-- Theorem: The ratio of the population of City Y to the population of City Z is 2
theorem population_ratio (h1 : condition1 X Y) (h2 : condition2 X Z) : Y / Z = 2 := by
  sorry

end population_ratio_l3609_360936


namespace solution_volume_l3609_360905

/-- Given two solutions, one of 6 litres and another of V litres, 
    if 20% of the first solution is mixed with 60% of the second solution,
    and the resulting mixture is 36% of the total volume,
    then V equals 4 litres. -/
theorem solution_volume (V : ℝ) : 
  (0.2 * 6 + 0.6 * V) / (6 + V) = 0.36 → V = 4 := by
  sorry

end solution_volume_l3609_360905


namespace wire_length_l3609_360968

/-- Given two vertical poles on a flat surface, where:
    - The distance between the pole bottoms is 8 feet
    - The height of the first pole is 10 feet
    - The height of the second pole is 4 feet
    This theorem proves that the length of a wire stretched from the top of the taller pole
    to the top of the shorter pole is 10 feet. -/
theorem wire_length (pole1_height pole2_height pole_distance : ℝ) 
  (h1 : pole1_height = 10)
  (h2 : pole2_height = 4)
  (h3 : pole_distance = 8) :
  Real.sqrt ((pole1_height - pole2_height)^2 + pole_distance^2) = 10 := by
  sorry

#check wire_length

end wire_length_l3609_360968


namespace state_tax_calculation_l3609_360910

/-- Calculate the state tax for a partial-year resident -/
theorem state_tax_calculation 
  (months_resident : ℕ) 
  (taxable_income : ℝ) 
  (tax_rate : ℝ) : 
  months_resident = 9 → 
  taxable_income = 42500 → 
  tax_rate = 0.04 → 
  (months_resident : ℝ) / 12 * taxable_income * tax_rate = 1275 := by
  sorry

end state_tax_calculation_l3609_360910


namespace polynomial_uniqueness_l3609_360934

theorem polynomial_uniqueness (Q : ℝ → ℝ) :
  (∀ x, Q x = Q 0 + Q 1 * x + Q 2 * x^2 + Q 3 * x^3) →
  Q (-1) = 2 →
  ∀ x, Q x = x^2 + 1 := by
sorry

end polynomial_uniqueness_l3609_360934


namespace apple_stack_theorem_l3609_360944

/-- Calculates the number of apples in a pyramid-like stack --/
def appleStack (baseWidth : Nat) (baseLength : Nat) : Nat :=
  let layers := min baseWidth baseLength
  List.range layers
  |>.map (fun i => (baseWidth - i) * (baseLength - i))
  |>.sum

/-- Theorem: A pyramid-like stack of apples with a 4x6 base contains 50 apples --/
theorem apple_stack_theorem :
  appleStack 4 6 = 50 := by
  sorry

end apple_stack_theorem_l3609_360944


namespace ray_fish_market_l3609_360970

/-- Calculates the number of tuna needed to serve customers in a fish market -/
def tuna_needed (total_customers : ℕ) (unsatisfied_customers : ℕ) (pounds_per_customer : ℕ) (pounds_per_tuna : ℕ) : ℕ :=
  ((total_customers - unsatisfied_customers) * pounds_per_customer) / pounds_per_tuna

theorem ray_fish_market :
  tuna_needed 100 20 25 200 = 10 := by
  sorry

end ray_fish_market_l3609_360970


namespace fruits_eaten_over_two_meals_l3609_360911

/-- Calculates the total number of fruits eaten over two meals given specific conditions --/
theorem fruits_eaten_over_two_meals : 
  let apples_last_night : ℕ := 3
  let bananas_last_night : ℕ := 1
  let oranges_last_night : ℕ := 4
  let strawberries_last_night : ℕ := 2
  
  let apples_today : ℕ := apples_last_night + 4
  let bananas_today : ℕ := bananas_last_night * 10
  let oranges_today : ℕ := apples_today * 2
  let strawberries_today : ℕ := (oranges_last_night + apples_last_night) * 3
  
  let total_fruits : ℕ := 
    (apples_last_night + apples_today) +
    (bananas_last_night + bananas_today) +
    (oranges_last_night + oranges_today) +
    (strawberries_last_night + strawberries_today)
  
  total_fruits = 62 := by sorry

end fruits_eaten_over_two_meals_l3609_360911


namespace ninth_observation_l3609_360917

theorem ninth_observation (n : ℕ) (original_avg new_avg : ℚ) (decrease : ℚ) :
  n = 8 →
  original_avg = 15 →
  decrease = 2 →
  new_avg = original_avg - decrease →
  (n * original_avg + (n + 1) * new_avg) / (2 * n + 1) - original_avg = -3 :=
by sorry

end ninth_observation_l3609_360917


namespace sum_of_cubes_l3609_360971

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = 1) : a^3 + b^3 = -2 := by
  sorry

end sum_of_cubes_l3609_360971


namespace midpoint_sum_midpoint_sum_specific_l3609_360990

/-- The sum of the coordinates of the midpoint of a line segment with endpoints (3, -1) and (11, 21) is 17. -/
theorem midpoint_sum : ℝ × ℝ → ℝ × ℝ → ℝ
  | (x₁, y₁) => λ (x₂, y₂) => (x₁ + x₂) / 2 + (y₁ + y₂) / 2

#check midpoint_sum (3, -1) (11, 21) = 17

theorem midpoint_sum_specific :
  midpoint_sum (3, -1) (11, 21) = 17 := by
  sorry

end midpoint_sum_midpoint_sum_specific_l3609_360990


namespace rectangle_area_l3609_360975

/-- The area of a rectangle with length 8m and width 50dm is 40 m² -/
theorem rectangle_area : 
  let length : ℝ := 8
  let width_dm : ℝ := 50
  let width_m : ℝ := width_dm / 10
  let area : ℝ := length * width_m
  area = 40 := by sorry

end rectangle_area_l3609_360975


namespace mateo_deducted_salary_l3609_360900

theorem mateo_deducted_salary (weekly_salary : ℝ) (work_days : ℕ) (absent_days : ℕ) : 
  weekly_salary = 791 ∧ work_days = 5 ∧ absent_days = 4 →
  weekly_salary - (weekly_salary / work_days * absent_days) = 158.20 := by
  sorry

end mateo_deducted_salary_l3609_360900


namespace cos_sin_transformation_l3609_360923

theorem cos_sin_transformation (x : Real) : 
  Real.sqrt 2 * Real.cos x = Real.sqrt 2 * Real.sin (2 * (x + Real.pi/4) + Real.pi/4) := by
sorry

end cos_sin_transformation_l3609_360923


namespace hyperbola_line_intersection_l3609_360986

/-- The hyperbola equation x^2 - y^2 = 4 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 4

/-- The line equation y = k(x-1) -/
def line (k x y : ℝ) : Prop := y = k * (x - 1)

/-- The number of intersection points between the line and the hyperbola -/
def intersection_count (k : ℝ) : ℕ := sorry

theorem hyperbola_line_intersection (k : ℝ) :
  (intersection_count k = 2 ↔ k ∈ Set.Ioo (-2 * Real.sqrt 3 / 3) (-1) ∪ 
                            Set.Ioo (-1) 1 ∪ 
                            Set.Ioo 1 (2 * Real.sqrt 3 / 3)) ∧
  (intersection_count k = 1 ↔ k = 1 ∨ k = -1 ∨ k = 2 * Real.sqrt 3 / 3 ∨ k = -2 * Real.sqrt 3 / 3) ∧
  (intersection_count k = 0 ↔ k ∈ Set.Iic (-2 * Real.sqrt 3 / 3) ∪ 
                            Set.Ici (2 * Real.sqrt 3 / 3)) :=
sorry

end hyperbola_line_intersection_l3609_360986


namespace systematic_sampling_probability_l3609_360969

/-- Probability of selecting an individual in systematic sampling -/
theorem systematic_sampling_probability
  (population_size : ℕ)
  (sample_size : ℕ)
  (h1 : population_size = 1001)
  (h2 : sample_size = 50)
  (h3 : population_size > 0)
  (h4 : sample_size ≤ population_size) :
  (sample_size : ℚ) / population_size = 50 / 1001 :=
sorry

end systematic_sampling_probability_l3609_360969


namespace final_price_is_25_92_l3609_360903

/-- The final price observed by the buyer online given the commission rate,
    product cost, and desired profit rate. -/
def final_price (commission_rate : ℝ) (product_cost : ℝ) (profit_rate : ℝ) : ℝ :=
  let profit := product_cost * profit_rate
  let distributor_price := product_cost + profit
  let commission := distributor_price * commission_rate
  distributor_price + commission

/-- Theorem stating that the final price is $25.92 given the specified conditions -/
theorem final_price_is_25_92 :
  final_price 0.2 18 0.2 = 25.92 := by
  sorry

end final_price_is_25_92_l3609_360903


namespace radiator_fluid_calculation_l3609_360987

theorem radiator_fluid_calculation (initial_antifreeze_percentage : Real)
                                   (drain_amount : Real)
                                   (replacement_antifreeze_percentage : Real)
                                   (final_antifreeze_percentage : Real) :
  initial_antifreeze_percentage = 0.10 →
  drain_amount = 2.2857 →
  replacement_antifreeze_percentage = 0.80 →
  final_antifreeze_percentage = 0.50 →
  ∃ x : Real, x = 4 ∧
    initial_antifreeze_percentage * x - 
    initial_antifreeze_percentage * drain_amount + 
    replacement_antifreeze_percentage * drain_amount = 
    final_antifreeze_percentage * x :=
by
  sorry

end radiator_fluid_calculation_l3609_360987


namespace final_water_fraction_l3609_360959

def container_size : ℚ := 25

def initial_water : ℚ := 25

def replacement_volume : ℚ := 5

def third_replacement_water : ℚ := 2

def calculate_final_water_fraction (initial_water : ℚ) (container_size : ℚ) 
  (replacement_volume : ℚ) (third_replacement_water : ℚ) : ℚ :=
  sorry

theorem final_water_fraction :
  calculate_final_water_fraction initial_water container_size replacement_volume third_replacement_water
  = 14.8 / 25 :=
sorry

end final_water_fraction_l3609_360959


namespace marathon_remainder_yards_l3609_360927

/-- Represents the distance of a marathon in miles and yards -/
structure MarathonDistance :=
  (miles : ℕ)
  (yards : ℕ)

/-- Represents a total distance in miles and yards -/
structure TotalDistance :=
  (miles : ℕ)
  (yards : ℕ)

def marathon_distance : MarathonDistance :=
  { miles := 26, yards := 385 }

def yards_per_mile : ℕ := 1760

def num_marathons : ℕ := 15

theorem marathon_remainder_yards :
  ∃ (m : ℕ) (y : ℕ), 
    y < yards_per_mile ∧
    TotalDistance.yards (
      {miles := m, 
       yards := y} : TotalDistance
    ) = 495 ∧
    m * yards_per_mile + y = 
      num_marathons * (marathon_distance.miles * yards_per_mile + marathon_distance.yards) :=
by sorry

end marathon_remainder_yards_l3609_360927


namespace hcf_problem_l3609_360937

theorem hcf_problem (a b : ℕ+) (h1 : a * b = 2562) (h2 : Nat.lcm a b = 183) :
  Nat.gcd a b = 14 := by
  sorry

end hcf_problem_l3609_360937


namespace last_two_nonzero_digits_70_factorial_l3609_360956

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_nonzero_digits (n : ℕ) : ℕ :=
  let m := n % 100
  if m ≠ 0 then m else last_two_nonzero_digits (n / 10)

theorem last_two_nonzero_digits_70_factorial :
  ∃ n : ℕ, last_two_nonzero_digits (factorial 70) = n ∧ n < 100 := by
  sorry

#eval last_two_nonzero_digits (factorial 70)

end last_two_nonzero_digits_70_factorial_l3609_360956


namespace base12_addition_correct_l3609_360954

/-- Represents a digit in base 12 --/
inductive Digit12 : Type
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B

/-- Converts a Digit12 to its decimal value --/
def toDecimal (d : Digit12) : Nat :=
  match d with
  | Digit12.D0 => 0
  | Digit12.D1 => 1
  | Digit12.D2 => 2
  | Digit12.D3 => 3
  | Digit12.D4 => 4
  | Digit12.D5 => 5
  | Digit12.D6 => 6
  | Digit12.D7 => 7
  | Digit12.D8 => 8
  | Digit12.D9 => 9
  | Digit12.A => 10
  | Digit12.B => 11

/-- Represents a number in base 12 --/
def Base12 := List Digit12

/-- Converts a Base12 number to its decimal value --/
def base12ToDecimal (n : Base12) : Nat :=
  n.foldr (fun d acc => toDecimal d + 12 * acc) 0

/-- Addition in base 12 --/
def addBase12 (a b : Base12) : Base12 :=
  sorry -- Implementation details omitted

theorem base12_addition_correct :
  addBase12 [Digit12.D8, Digit12.A, Digit12.D2] [Digit12.D3, Digit12.B, Digit12.D7] =
  [Digit12.D1, Digit12.D0, Digit12.D9, Digit12.D9] :=
by sorry

end base12_addition_correct_l3609_360954


namespace f_2_3_neg1_eq_5_3_l3609_360988

-- Define the function f
def f (a b c : ℚ) : ℚ := (a + b) / (a - c)

-- State the theorem
theorem f_2_3_neg1_eq_5_3 : f 2 3 (-1) = 5/3 := by
  sorry

end f_2_3_neg1_eq_5_3_l3609_360988


namespace min_value_of_function_min_value_achieved_l3609_360920

theorem min_value_of_function (x : ℝ) (hx : x < 0) :
  (1 - 2*x - 3/x) ≥ 1 + 2*Real.sqrt 6 := by
  sorry

theorem min_value_achieved (x : ℝ) (hx : x < 0) :
  ∃ x₀, x₀ < 0 ∧ (1 - 2*x₀ - 3/x₀) = 1 + 2*Real.sqrt 6 := by
  sorry

end min_value_of_function_min_value_achieved_l3609_360920


namespace number_operation_result_l3609_360901

theorem number_operation_result : 
  let n : ℚ := 55
  (n / 5 + 10) = 21 := by sorry

end number_operation_result_l3609_360901


namespace ones_digit_of_large_power_l3609_360985

theorem ones_digit_of_large_power : ∃ n : ℕ, n < 10 ∧ 34^(34 * 17^17) ≡ n [ZMOD 10] ∧ n = 6 := by
  sorry

end ones_digit_of_large_power_l3609_360985


namespace parabola_vertex_l3609_360965

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := y = -3 * (x - 1)^2 + 4

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (1, 4)

/-- Theorem: The vertex of the parabola y = -3(x-1)^2 + 4 is at the point (1,4) -/
theorem parabola_vertex :
  ∀ x y : ℝ, parabola_equation x y → (x, y) = vertex :=
sorry

end parabola_vertex_l3609_360965


namespace street_crossing_time_l3609_360924

/-- Proves that a person walking at 5.4 km/h takes 12 minutes to cross a 1080 m street -/
theorem street_crossing_time :
  let street_length : ℝ := 1080  -- length in meters
  let speed_kmh : ℝ := 5.4       -- speed in km/h
  let speed_mpm : ℝ := speed_kmh * 1000 / 60  -- speed in meters per minute
  let time_minutes : ℝ := street_length / speed_mpm
  time_minutes = 12 := by sorry

end street_crossing_time_l3609_360924


namespace michael_tom_flying_robots_ratio_l3609_360941

theorem michael_tom_flying_robots_ratio : 
  ∀ (michael_robots tom_robots : ℕ), 
    michael_robots = 12 → 
    tom_robots = 3 → 
    (michael_robots : ℚ) / (tom_robots : ℚ) = 4 := by
  sorry

end michael_tom_flying_robots_ratio_l3609_360941


namespace matrix_product_50_l3609_360976

def matrix_product (n : ℕ) : Matrix (Fin 2) (Fin 2) ℕ :=
  (List.range n).foldl
    (fun acc k => acc * !![1, 2*(k+1); 0, 1])
    !![1, 0; 0, 1]

theorem matrix_product_50 :
  matrix_product 50 = !![1, 2550; 0, 1] := by sorry

end matrix_product_50_l3609_360976


namespace smallest_integer_with_remainders_l3609_360953

theorem smallest_integer_with_remainders : ∃! n : ℕ, 
  n > 0 ∧
  n % 4 = 3 ∧
  n % 5 = 4 ∧
  n % 6 = 5 ∧
  n % 7 = 6 ∧
  ∀ m : ℕ, m > 0 ∧ m % 4 = 3 ∧ m % 5 = 4 ∧ m % 6 = 5 ∧ m % 7 = 6 → n ≤ m :=
by sorry

end smallest_integer_with_remainders_l3609_360953


namespace polynomial_composition_difference_l3609_360925

theorem polynomial_composition_difference (f : Polynomial ℝ) :
  ∃ (g h : Polynomial ℝ), f = g.comp h - h.comp g := by
  sorry

end polynomial_composition_difference_l3609_360925


namespace friends_walking_problem_l3609_360942

/-- Two friends walking on a trail problem -/
theorem friends_walking_problem (v : ℝ) (h : v > 0) :
  let trail_length : ℝ := 22
  let speed_ratio : ℝ := 1.2
  let d : ℝ := trail_length / (1 + speed_ratio)
  trail_length - d = 12 := by sorry

end friends_walking_problem_l3609_360942


namespace sin_intersection_sum_l3609_360955

open Real

theorem sin_intersection_sum (f : ℝ → ℝ) (x₁ x₂ x₃ : ℝ) :
  (∀ x ∈ Set.Icc 0 (7 * π / 6), f x = Real.sin (2 * x + π / 6)) →
  x₁ < x₂ →
  x₂ < x₃ →
  x₁ ∈ Set.Icc 0 (7 * π / 6) →
  x₂ ∈ Set.Icc 0 (7 * π / 6) →
  x₃ ∈ Set.Icc 0 (7 * π / 6) →
  f x₁ = f x₂ →
  f x₂ = f x₃ →
  x₁ + 2 * x₂ + x₃ = 5 * π / 3 :=
by sorry

end sin_intersection_sum_l3609_360955


namespace runners_passing_count_l3609_360940

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ  -- speed in meters per minute
  radius : ℝ  -- radius of the track in meters
  direction : ℤ  -- 1 for clockwise, -1 for counterclockwise

/-- Calculates the number of times two runners pass each other -/
def passingCount (r1 r2 : Runner) (duration : ℝ) : ℕ :=
  sorry

theorem runners_passing_count :
  let odell : Runner := { speed := 260, radius := 55, direction := 1 }
  let kershaw : Runner := { speed := 280, radius := 65, direction := -1 }
  passingCount odell kershaw 30 = 126 :=
sorry

end runners_passing_count_l3609_360940


namespace betty_needs_five_more_l3609_360972

def wallet_cost : ℕ := 100
def betty_initial_savings : ℕ := wallet_cost / 2
def parents_contribution : ℕ := 15
def grandparents_contribution : ℕ := 2 * parents_contribution

theorem betty_needs_five_more :
  wallet_cost - (betty_initial_savings + parents_contribution + grandparents_contribution) = 5 := by
  sorry

end betty_needs_five_more_l3609_360972


namespace min_n_for_constant_term_l3609_360909

theorem min_n_for_constant_term (n : ℕ) : 
  (∃ k : ℕ, (2 * n = 3 * k) ∧ (k ≤ n)) ↔ n ≥ 3 :=
by sorry

end min_n_for_constant_term_l3609_360909


namespace expression_equivalence_l3609_360948

theorem expression_equivalence (x y : ℝ) (h : x * y ≠ 0) :
  ((x^4 + 1) / x^2) * ((y^4 + 1) / y^2) + ((x^4 - 1) / y^2) * ((y^4 - 1) / x^2) = 2 * x^2 * y^2 + 2 / (x^2 * y^2) := by
  sorry

end expression_equivalence_l3609_360948


namespace expression_evaluation_l3609_360981

theorem expression_evaluation (x : ℝ) (h : x = -1) :
  (((x - 2) / x - x / (x + 2)) / ((x + 2) / (x^2 + 4*x + 4))) = 4 := by
  sorry

end expression_evaluation_l3609_360981


namespace team_can_have_odd_and_even_points_l3609_360922

/-- Represents a football team in the tournament -/
structure Team :=
  (id : Nat)
  (points : Nat)

/-- Represents the football tournament -/
structure Tournament :=
  (teams : Finset Team)
  (num_teams : Nat)
  (points_for_win : Nat)
  (points_for_draw : Nat)
  (bonus_points : Nat)

/-- Definition of the specific tournament conditions -/
def specific_tournament : Tournament :=
  { teams := sorry,
    num_teams := 10,
    points_for_win := 3,
    points_for_draw := 1,
    bonus_points := 5 }

/-- Theorem stating that a team can end with both odd and even points -/
theorem team_can_have_odd_and_even_points (t : Tournament) 
  (h1 : t.num_teams = 10)
  (h2 : t.points_for_win = 3)
  (h3 : t.points_for_draw = 1)
  (h4 : t.bonus_points = 5) :
  ∃ (team1 team2 : Team), 
    team1 ∈ t.teams ∧ 
    team2 ∈ t.teams ∧ 
    Odd team1.points ∧ 
    Even team2.points :=
sorry

end team_can_have_odd_and_even_points_l3609_360922


namespace min_value_f_neg_three_range_of_a_l3609_360995

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Theorem 1: Minimum value of f when a = -3
theorem min_value_f_neg_three :
  ∃ (m : ℝ), m = 4 ∧ ∀ (x : ℝ), f (-3) x ≥ m :=
sorry

-- Theorem 2: Range of a when f(x) ≤ 2a + 2|x-1| for all x
theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ), f a x ≤ 2 * a + 2 * |x - 1|) → a ≥ 1/3 :=
sorry

end min_value_f_neg_three_range_of_a_l3609_360995


namespace fourth_quadrant_a_range_l3609_360957

-- Define the complex number z
def z (a : ℝ) : ℂ := (1 - 2*Complex.I) * (a + Complex.I)

-- Define the point M
def M (a : ℝ) : ℝ × ℝ := (a + 2, 1 - 2*a)

-- Theorem statement
theorem fourth_quadrant_a_range (a : ℝ) :
  (M a).1 > 0 ∧ (M a).2 < 0 → a > 1/2 := by sorry

end fourth_quadrant_a_range_l3609_360957


namespace chocolate_boxes_given_away_l3609_360943

theorem chocolate_boxes_given_away (total_boxes : ℕ) (pieces_per_box : ℕ) (remaining_pieces : ℕ) : 
  total_boxes = 14 → pieces_per_box = 6 → remaining_pieces = 54 → 
  (total_boxes * pieces_per_box - remaining_pieces) / pieces_per_box = 5 := by
  sorry

end chocolate_boxes_given_away_l3609_360943


namespace min_ratio_of_integers_with_mean_l3609_360946

theorem min_ratio_of_integers_with_mean (x y : ℤ) : 
  10 ≤ x ∧ x ≤ 150 → 
  10 ≤ y ∧ y ≤ 150 → 
  (x + y) / 2 = 75 → 
  ∃ (x' y' : ℤ), 
    10 ≤ x' ∧ x' ≤ 150 ∧ 
    10 ≤ y' ∧ y' ≤ 150 ∧ 
    (x' + y') / 2 = 75 ∧ 
    x' / y' ≤ x / y ∧
    x' / y' = 1 / 14 :=
by sorry

end min_ratio_of_integers_with_mean_l3609_360946


namespace ellipse_equation_l3609_360960

/-- Definition of an ellipse with given focal distance and major axis length -/
structure Ellipse :=
  (focal_distance : ℝ)
  (major_axis_length : ℝ)

/-- Standard equation of an ellipse -/
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), 
    a = e.major_axis_length / 2 ∧
    b^2 = a^2 - (e.focal_distance / 2)^2 ∧
    x^2 / a^2 + y^2 / b^2 = 1

/-- Theorem stating the standard equation of the given ellipse -/
theorem ellipse_equation (e : Ellipse) 
  (h1 : e.focal_distance = 8)
  (h2 : e.major_axis_length = 10) :
  standard_equation e x y ↔ x^2 / 25 + y^2 / 9 = 1 :=
sorry

end ellipse_equation_l3609_360960


namespace no_whole_number_57_times_less_l3609_360974

theorem no_whole_number_57_times_less : ¬ ∃ (N : ℕ) (n : ℕ) (a : Fin 10),
  N ≥ 10 ∧ 
  a.val ≠ 0 ∧
  N = a.val * 10^n + (N / 57) :=
sorry

end no_whole_number_57_times_less_l3609_360974


namespace f_max_min_values_l3609_360947

-- Define the function
def f (x : ℝ) : ℝ := 3 * x - x^3

-- State the theorem
theorem f_max_min_values :
  (∃ x : ℝ, f x = 2 ∧ ∀ y : ℝ, f y ≤ 2) ∧
  (∃ x : ℝ, f x = -2 ∧ ∀ y : ℝ, f y ≥ -2) := by
  sorry

end f_max_min_values_l3609_360947


namespace certain_number_l3609_360998

theorem certain_number : ∃ x : ℝ, x + 0.675 = 0.8 ∧ x = 0.125 := by
  sorry

end certain_number_l3609_360998
