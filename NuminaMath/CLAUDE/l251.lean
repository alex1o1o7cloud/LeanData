import Mathlib

namespace systematic_sampling_constant_difference_l251_25109

/-- Represents a sequence of 5 numbers -/
structure Sequence :=
  (numbers : Fin 5 → Nat)

/-- Checks if a sequence has a constant difference between consecutive elements -/
def hasConstantDifference (s : Sequence) (d : Nat) : Prop :=
  ∀ i : Fin 4, s.numbers (i.succ) - s.numbers i = d

/-- Systematic sampling function -/
def systematicSample (totalStudents : Nat) (sampleSize : Nat) : Sequence :=
  sorry

theorem systematic_sampling_constant_difference :
  let totalStudents : Nat := 55
  let sampleSize : Nat := 5
  let sampledSequence := systematicSample totalStudents sampleSize
  hasConstantDifference sampledSequence (totalStudents / sampleSize) :=
by
  sorry

end systematic_sampling_constant_difference_l251_25109


namespace min_sum_of_prime_factors_l251_25140

theorem min_sum_of_prime_factors (x : ℕ) : 
  let sequence_sum := 25 * (x + 12)
  ∃ (p₁ p₂ p₃ : ℕ), Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ sequence_sum = p₁ * p₂ * p₃ →
  ∀ (q₁ q₂ q₃ : ℕ), Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ sequence_sum = q₁ * q₂ * q₃ →
  q₁ + q₂ + q₃ ≥ 23 :=
by sorry

end min_sum_of_prime_factors_l251_25140


namespace sandals_sold_l251_25113

theorem sandals_sold (shoes : ℕ) (sandals : ℕ) : 
  (shoes : ℚ) / sandals = 15 / 8 → shoes = 135 → sandals = 72 := by
sorry

end sandals_sold_l251_25113


namespace minutes_before_noon_l251_25128

/-- 
Given that 20 minutes ago it was 3 times as many minutes after 9 am, 
and there are 180 minutes between 9 am and 12 noon, 
prove that it is 130 minutes before 12 noon.
-/
theorem minutes_before_noon : 
  ∀ x : ℕ, 
  (x + 20 = 3 * (180 - x)) → 
  x = 130 := by
sorry

end minutes_before_noon_l251_25128


namespace no_eight_roots_for_composite_quadratics_l251_25123

/-- A quadratic trinomial is a polynomial of degree 2 -/
def QuadraticTrinomial (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem no_eight_roots_for_composite_quadratics :
  ¬ ∃ (f g h : ℝ → ℝ),
    QuadraticTrinomial f ∧ QuadraticTrinomial g ∧ QuadraticTrinomial h ∧
    (∀ x, f (g (h x)) = 0 ↔ x ∈ ({1, 2, 3, 4, 5, 6, 7, 8} : Set ℝ)) :=
by sorry

end no_eight_roots_for_composite_quadratics_l251_25123


namespace right_triangle_angle_sum_l251_25150

theorem right_triangle_angle_sum (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C)
  (h4 : A + B + C = 180) (h5 : A + B = C) : C = 90 := by
  sorry

end right_triangle_angle_sum_l251_25150


namespace ferris_wheel_capacity_l251_25192

theorem ferris_wheel_capacity (total_seats broken_seats people_riding : ℕ) 
  (h1 : total_seats = 18)
  (h2 : broken_seats = 10)
  (h3 : people_riding = 120) :
  people_riding / (total_seats - broken_seats) = 15 := by
  sorry

end ferris_wheel_capacity_l251_25192


namespace garden_vegetables_theorem_l251_25129

/-- Represents the quantities of vegetables in a garden -/
structure GardenVegetables where
  tomatoes : ℕ
  potatoes : ℕ
  cabbages : ℕ
  eggplants : ℕ

/-- Calculates the final quantities of vegetables after changes -/
def finalQuantities (initial : GardenVegetables) 
  (tomatoesPicked potatoes_sold cabbagesBought eggplantsPlanted : ℕ) : GardenVegetables :=
  { tomatoes := initial.tomatoes - min initial.tomatoes tomatoesPicked,
    potatoes := initial.potatoes - min initial.potatoes potatoes_sold,
    cabbages := initial.cabbages + cabbagesBought,
    eggplants := initial.eggplants + eggplantsPlanted }

theorem garden_vegetables_theorem (initial : GardenVegetables) 
  (tomatoesPicked potatoes_sold cabbagesBought eggplantsPlanted : ℕ) :
  initial.tomatoes = 177 → 
  initial.potatoes = 12 → 
  initial.cabbages = 25 → 
  initial.eggplants = 10 → 
  tomatoesPicked = 53 → 
  potatoes_sold = 15 → 
  cabbagesBought = 32 → 
  eggplantsPlanted = 18 → 
  finalQuantities initial tomatoesPicked potatoes_sold cabbagesBought eggplantsPlanted = 
    { tomatoes := 124, potatoes := 0, cabbages := 57, eggplants := 28 } := by
  sorry

end garden_vegetables_theorem_l251_25129


namespace geometric_sequence_product_property_l251_25160

/-- A sequence is geometric if there exists a non-zero common ratio between consecutive terms. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

/-- The property that a_m * a_n = a_p * a_q for specific m, n, p, q. -/
def HasProductProperty (a : ℕ → ℝ) (m n p q : ℕ) : Prop :=
  a m * a n = a p * a q

theorem geometric_sequence_product_property 
  (a : ℕ → ℝ) (m n p q : ℕ) 
  (hm : m > 0) (hn : n > 0) (hp : p > 0) (hq : q > 0)
  (h_sum : m + n = p + q) :
  IsGeometricSequence a → HasProductProperty a m n p q :=
by
  sorry


end geometric_sequence_product_property_l251_25160


namespace sqrt_sum_quotient_l251_25121

theorem sqrt_sum_quotient : 
  (Real.sqrt 1.21) / (Real.sqrt 0.81) + (Real.sqrt 1.44) / (Real.sqrt 0.49) = 185/63 := by
  sorry

end sqrt_sum_quotient_l251_25121


namespace initial_milk_water_ratio_l251_25101

/-- Proves that the initial ratio of milk to water is 3:1 given the conditions -/
theorem initial_milk_water_ratio 
  (total_volume : ℝ) 
  (added_water : ℝ) 
  (final_ratio : ℝ) :
  total_volume = 50 →
  added_water = 100 →
  final_ratio = 1/3 →
  ∃ (initial_milk initial_water : ℝ),
    initial_milk + initial_water = total_volume ∧
    initial_milk / (initial_water + added_water) = final_ratio ∧
    initial_milk / initial_water = 3 := by
  sorry

end initial_milk_water_ratio_l251_25101


namespace ellipse_circle_dot_product_range_l251_25132

-- Define the ellipse
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 9) + (P.2^2 / 8) = 1

-- Define the circle
def is_on_circle (P : ℝ × ℝ) : Prop :=
  (P.1 - 1)^2 + P.2^2 = 1

-- Define a diameter of the circle
def is_diameter (A B : ℝ × ℝ) : Prop :=
  is_on_circle A ∧ is_on_circle B ∧ (A.1 + B.1 = 2) ∧ (A.2 + B.2 = 0)

-- Define the dot product
def dot_product (P A B : ℝ × ℝ) : ℝ :=
  (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2)

theorem ellipse_circle_dot_product_range :
  ∀ (P A B : ℝ × ℝ),
    is_on_ellipse P →
    is_diameter A B →
    3 ≤ dot_product P A B ∧ dot_product P A B ≤ 15 :=
by sorry

end ellipse_circle_dot_product_range_l251_25132


namespace integral_sqrt_a_squared_minus_x_squared_l251_25162

theorem integral_sqrt_a_squared_minus_x_squared (a : ℝ) (ha : a > 0) :
  ∫ x in -a..a, Real.sqrt (a^2 - x^2) = (1/2) * π * a^2 := by
  sorry

end integral_sqrt_a_squared_minus_x_squared_l251_25162


namespace a_greater_than_b_l251_25119

theorem a_greater_than_b (m : ℝ) (h : m > 1) : 
  (Real.sqrt m - Real.sqrt (m - 1)) > (Real.sqrt (m + 1) - Real.sqrt m) := by
  sorry

end a_greater_than_b_l251_25119


namespace valid_outfit_count_l251_25155

/-- The number of shirts available. -/
def num_shirts : ℕ := 7

/-- The number of pants available. -/
def num_pants : ℕ := 5

/-- The number of hats available. -/
def num_hats : ℕ := 7

/-- The number of colors available for pants. -/
def num_pants_colors : ℕ := 5

/-- The number of colors available for shirts and hats. -/
def num_shirt_hat_colors : ℕ := 7

/-- The number of valid outfit choices. -/
def num_valid_outfits : ℕ := num_shirts * num_pants * num_hats - num_pants_colors

theorem valid_outfit_count : num_valid_outfits = 240 := by
  sorry

end valid_outfit_count_l251_25155


namespace function_properties_l251_25196

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x / 4 + a / x - Real.log x - 3 / 2

-- Define the derivative of f(x)
def f' (a : ℝ) (x : ℝ) : ℝ := 1 / 4 - a / (x^2) - 1 / x

theorem function_properties (a : ℝ) :
  (∀ x > 0, f a x = x / 4 + a / x - Real.log x - 3 / 2) →
  (f' a 1 = -2) →
  ∃ (x_min : ℝ),
    (a = 5 / 4) ∧
    (x_min = 5) ∧
    (∀ x ∈ Set.Ioo 0 x_min, (f' (5/4)) x < 0) ∧
    (∀ x ∈ Set.Ioi x_min, (f' (5/4)) x > 0) ∧
    (∀ x > 0, f (5/4) x ≥ f (5/4) x_min) ∧
    (f (5/4) x_min = -Real.log 5) :=
by sorry

end

end function_properties_l251_25196


namespace diagonals_properties_l251_25148

/-- Number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem diagonals_properties :
  (∀ n : ℕ, n ≥ 4 → num_diagonals n = n * (n - 3) / 2) →
  num_diagonals 4 = 2 →
  num_diagonals 5 = 5 ∧
  num_diagonals 6 - num_diagonals 5 = 4 ∧
  ∀ n : ℕ, n ≥ 4 → num_diagonals (n + 1) - num_diagonals n = n - 1 :=
by sorry

end diagonals_properties_l251_25148


namespace village_population_l251_25145

/-- Given that 40% of a village's population is 23040, prove that the total population is 57600. -/
theorem village_population (population : ℕ) (h : (40 : ℕ) * population = 100 * 23040) : population = 57600 := by
  sorry

end village_population_l251_25145


namespace no_rectangle_solution_l251_25182

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem no_rectangle_solution : ¬∃ (x y : ℕ), 
  is_prime x ∧ is_prime y ∧ 
  x < y ∧ y < 6 ∧ 
  2 * (x + y) = 21 ∧ 
  x * y = 45 :=
sorry

end no_rectangle_solution_l251_25182


namespace f_negative_m_value_l251_25139

noncomputable def f (x : ℝ) : ℝ := (x^3 + 3*x^2 + x + 9) / (x^2 + 3)

theorem f_negative_m_value (m : ℝ) (h : f m = 10) : f (-m) = -4 := by
  sorry

end f_negative_m_value_l251_25139


namespace smallest_perimeter_l251_25131

/-- Triangle PQR with positive integer side lengths, PQ = PR, and J is the intersection of angle bisectors of ∠Q and ∠R with QJ = 10 -/
structure IsoscelesTriangle where
  PQ : ℕ+
  QR : ℕ+
  J : ℝ × ℝ
  QJ_length : ℝ
  qj_eq_10 : QJ_length = 10

/-- The smallest possible perimeter of triangle PQR is 96 -/
theorem smallest_perimeter (t : IsoscelesTriangle) : 
  ∃ (min_perimeter : ℕ), min_perimeter = 96 ∧ 
  ∀ (perimeter : ℕ), perimeter ≥ min_perimeter :=
by sorry

end smallest_perimeter_l251_25131


namespace toms_next_birthday_l251_25185

theorem toms_next_birthday (sally tom jenny : ℝ) 
  (h1 : sally = 1.25 * tom)  -- Sally is 25% older than Tom
  (h2 : tom = 0.7 * jenny)   -- Tom is 30% younger than Jenny
  (h3 : sally + tom + jenny = 30)  -- Sum of ages is 30
  : ⌊tom⌋ + 1 = 9 := by
  sorry

end toms_next_birthday_l251_25185


namespace set_A_properties_l251_25106

def A : Set ℕ := {n : ℕ | ∃ k : ℕ, n = 2^k ∧ k ≥ 1}

theorem set_A_properties :
  (∀ a ∈ A, ∀ b : ℕ, b > 0 → b < 2*a - 1 → ¬(2*a ∣ b*(b+1))) ∧
  (∀ a : ℕ, a > 1 → a ∉ A → ∃ b : ℕ, b > 0 ∧ b < 2*a - 1 ∧ (2*a ∣ b*(b+1))) :=
by sorry

end set_A_properties_l251_25106


namespace prove_complex_circle_theorem_l251_25146

def complex_circle_theorem (z : ℂ) : Prop :=
  Complex.abs (z - Complex.I) = Real.sqrt 5 →
  ∃ (center : ℂ) (radius : ℝ),
    center = Complex.mk 0 1 ∧
    radius = Real.sqrt 5 ∧
    Complex.abs (z - center) = radius

theorem prove_complex_circle_theorem :
  ∀ z : ℂ, complex_circle_theorem z :=
by
  sorry

end prove_complex_circle_theorem_l251_25146


namespace mode_of_data_set_l251_25136

def data_set : List Int := [-1, 0, 2, -1, 3]

def mode (l : List α) [DecidableEq α] : Option α :=
  l.argmax (fun x => l.count x)

theorem mode_of_data_set :
  mode data_set = some (-1) := by
  sorry

end mode_of_data_set_l251_25136


namespace P_in_third_quadrant_iff_m_less_than_two_l251_25103

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the third quadrant -/
def is_in_third_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The point P with coordinates (-1, -2+m) -/
def P (m : ℝ) : Point :=
  ⟨-1, -2+m⟩

/-- Theorem stating the condition for P to be in the third quadrant -/
theorem P_in_third_quadrant_iff_m_less_than_two (m : ℝ) :
  is_in_third_quadrant (P m) ↔ m < 2 := by
  sorry

end P_in_third_quadrant_iff_m_less_than_two_l251_25103


namespace greatest_value_of_fraction_l251_25133

theorem greatest_value_of_fraction (y : ℝ) : 
  (∀ θ : ℝ, y ≥ 14 / (5 + 3 * Real.sin θ)) → y = 7 := by
  sorry

end greatest_value_of_fraction_l251_25133


namespace haley_extra_tickets_l251_25125

/-- The number of extra concert tickets Haley bought -/
def extra_tickets (ticket_price : ℕ) (tickets_for_friends : ℕ) (total_spent : ℕ) : ℕ :=
  (total_spent / ticket_price) - tickets_for_friends

/-- Theorem: Haley bought 5 extra tickets -/
theorem haley_extra_tickets :
  extra_tickets 4 3 32 = 5 := by
  sorry

end haley_extra_tickets_l251_25125


namespace line_direction_vector_k_l251_25181

/-- A line passing through two points with a specific direction vector form -/
def Line (p1 p2 : ℝ × ℝ) (k : ℝ) : Prop :=
  let dir := (p2.1 - p1.1, p2.2 - p1.2)
  ∃ (t : ℝ), dir = (3 * t, k * t)

/-- The main theorem stating that k = -3 for the given line -/
theorem line_direction_vector_k (k : ℝ) : 
  Line (2, -1) (-4, 5) k → k = -3 := by
  sorry

end line_direction_vector_k_l251_25181


namespace elizabeth_lost_bottles_l251_25118

/-- The number of water bottles Elizabeth lost at school -/
def bottles_lost_at_school : ℕ := 2

theorem elizabeth_lost_bottles (initial_bottles : ℕ) (stolen_bottle : ℕ) (stickers_per_bottle : ℕ) (total_stickers : ℕ) :
  initial_bottles = 10 →
  stolen_bottle = 1 →
  stickers_per_bottle = 3 →
  total_stickers = 21 →
  stickers_per_bottle * (initial_bottles - bottles_lost_at_school - stolen_bottle) = total_stickers →
  bottles_lost_at_school = 2 := by
sorry

end elizabeth_lost_bottles_l251_25118


namespace inner_square_prob_10x10_l251_25143

/-- Represents a square checkerboard -/
structure Checkerboard where
  size : ℕ
  total_squares : ℕ
  edge_squares : ℕ
  inner_squares : ℕ

/-- Calculates the probability of choosing an inner square -/
def inner_square_probability (board : Checkerboard) : ℚ :=
  board.inner_squares / board.total_squares

/-- Properties of a 10x10 checkerboard -/
def board_10x10 : Checkerboard :=
  { size := 10
  , total_squares := 100
  , edge_squares := 36
  , inner_squares := 64 }

/-- Theorem: The probability of choosing an inner square on a 10x10 board is 16/25 -/
theorem inner_square_prob_10x10 :
  inner_square_probability board_10x10 = 16 / 25 := by
  sorry

end inner_square_prob_10x10_l251_25143


namespace malfunctioning_clock_correct_time_l251_25117

/-- Represents a 12-hour digital clock with a malfunction where '2' is displayed as '5' -/
structure MalfunctioningClock where
  /-- The number of hours in the clock (12) -/
  total_hours : ℕ
  /-- The number of minutes per hour (60) -/
  minutes_per_hour : ℕ
  /-- The number of hours affected by the malfunction -/
  incorrect_hours : ℕ
  /-- The number of minutes per hour affected by the malfunction -/
  incorrect_minutes : ℕ

/-- The fraction of the day a malfunctioning clock shows the correct time -/
def correct_time_fraction (clock : MalfunctioningClock) : ℚ :=
  ((clock.total_hours - clock.incorrect_hours : ℚ) / clock.total_hours) *
  ((clock.minutes_per_hour - clock.incorrect_minutes : ℚ) / clock.minutes_per_hour)

theorem malfunctioning_clock_correct_time :
  ∃ (clock : MalfunctioningClock),
    clock.total_hours = 12 ∧
    clock.minutes_per_hour = 60 ∧
    clock.incorrect_hours = 2 ∧
    clock.incorrect_minutes = 15 ∧
    correct_time_fraction clock = 5 / 8 := by
  sorry

end malfunctioning_clock_correct_time_l251_25117


namespace find_a_value_l251_25176

theorem find_a_value (a : ℝ) : 
  let A := {x : ℝ | x^2 - a*x + a^2 - 19 = 0}
  let B := {x : ℝ | x^2 - 5*x + 6 = 0}
  let C := {x : ℝ | x^2 + 2*x - 8 = 0}
  (∃ x, x ∈ A ∩ B) ∧ (A ∩ C = ∅) → a = -2 := by
  sorry

end find_a_value_l251_25176


namespace intersection_in_fourth_quadrant_l251_25116

def line1 (x : ℝ) : ℝ := -x
def line2 (x : ℝ) : ℝ := 2*x - 1

def intersection_point : ℝ × ℝ :=
  let x := 1
  let y := -1
  (x, y)

theorem intersection_in_fourth_quadrant :
  let (x, y) := intersection_point
  x > 0 ∧ y < 0 ∧ line1 x = y ∧ line2 x = y :=
sorry

end intersection_in_fourth_quadrant_l251_25116


namespace base5_multiplication_l251_25186

/-- Converts a base 5 number to its decimal equivalent -/
def base5ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a decimal number to its base 5 representation -/
def decimalToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem base5_multiplication (a b : List Nat) :
  decimalToBase5 (base5ToDecimal a * base5ToDecimal b) = [2, 1, 3, 4] :=
  by sorry

end base5_multiplication_l251_25186


namespace nuts_in_third_box_l251_25183

theorem nuts_in_third_box (A B C : ℝ) 
  (h1 : A = B + C - 6) 
  (h2 : B = A + C - 10) : C = 8 := by
  sorry

end nuts_in_third_box_l251_25183


namespace cricketer_score_l251_25173

theorem cricketer_score : ∀ (total_score : ℝ),
  (12 * 4 + 2 * 6 : ℝ) + 0.55223880597014926 * total_score = total_score →
  total_score = 134 := by
  sorry

end cricketer_score_l251_25173


namespace negation_of_universal_proposition_l251_25141

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, 0 < x ∧ x < π/2 → x > Real.sin x) ↔
  ∃ x : ℝ, 0 < x ∧ x < π/2 ∧ x ≤ Real.sin x :=
by sorry

end negation_of_universal_proposition_l251_25141


namespace perpendicular_vectors_m_value_l251_25167

/-- Given two vectors a and b in R², prove that if they are perpendicular
    and a = (1, -2) and b = (m, m+2), then m = -4. -/
theorem perpendicular_vectors_m_value
  (a b : ℝ × ℝ)
  (h1 : a = (1, -2))
  (h2 : ∃ m : ℝ, b = (m, m + 2))
  (h3 : a.1 * b.1 + a.2 * b.2 = 0) :
  ∃ m : ℝ, b = (m, m + 2) ∧ m = -4 :=
by sorry

end perpendicular_vectors_m_value_l251_25167


namespace square_screen_diagonal_l251_25104

theorem square_screen_diagonal (d : ℝ) : 
  d > 0 → 
  (d / Real.sqrt 2) ^ 2 = 20 ^ 2 + 42 → 
  d = Real.sqrt 884 := by
  sorry

end square_screen_diagonal_l251_25104


namespace medium_birdhouse_price_l251_25126

/-- The price of a large birdhouse -/
def large_price : ℕ := 22

/-- The price of a small birdhouse -/
def small_price : ℕ := 7

/-- The number of large birdhouses sold -/
def large_sold : ℕ := 2

/-- The number of medium birdhouses sold -/
def medium_sold : ℕ := 2

/-- The number of small birdhouses sold -/
def small_sold : ℕ := 3

/-- The total amount made from all birdhouses -/
def total_amount : ℕ := 97

/-- The price of a medium birdhouse -/
def medium_price : ℕ := 16

theorem medium_birdhouse_price : 
  large_price * large_sold + medium_price * medium_sold + small_price * small_sold = total_amount :=
by sorry

end medium_birdhouse_price_l251_25126


namespace f_value_at_2_l251_25177

def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

theorem f_value_at_2 (a b : ℝ) :
  f a b (-2) = 10 → f a b 2 = -26 := by
  sorry

end f_value_at_2_l251_25177


namespace log_inequality_l251_25174

theorem log_inequality (a b c : ℝ) (ha : a = Real.log 6 / Real.log 4)
  (hb : b = Real.log 0.2 / Real.log 4) (hc : c = Real.log 3 / Real.log 2) :
  c > a ∧ a > b :=
by sorry

end log_inequality_l251_25174


namespace proportion_problem_l251_25115

/-- Given that a, d, b, c are in proportion, where a = 3 cm, b = 4 cm, and c = 6 cm, prove that d = 9/2 cm. -/
theorem proportion_problem (a d b c : ℚ) : 
  a = 3 → b = 4 → c = 6 → (a / d = b / c) → d = 9 / 2 := by
  sorry

end proportion_problem_l251_25115


namespace complex_arithmetic_result_l251_25124

theorem complex_arithmetic_result :
  let A : ℂ := 5 - 2*I
  let M : ℂ := -3 + 3*I
  let S : ℂ := 2*I
  let P : ℝ := (1/2 : ℝ)
  A - M + S - (P : ℂ) = 7.5 - 3*I :=
by sorry

end complex_arithmetic_result_l251_25124


namespace distance_to_origin_l251_25158

theorem distance_to_origin : Real.sqrt (3^2 + (-2)^2) = Real.sqrt 13 := by sorry

end distance_to_origin_l251_25158


namespace carol_initial_cupcakes_l251_25191

/-- The number of cupcakes Carol initially made -/
def initial_cupcakes : ℕ := 30

/-- The number of cupcakes Carol sold -/
def sold_cupcakes : ℕ := 9

/-- The number of additional cupcakes Carol made -/
def additional_cupcakes : ℕ := 28

/-- The total number of cupcakes Carol had at the end -/
def total_cupcakes : ℕ := 49

/-- Theorem: Carol initially made 30 cupcakes -/
theorem carol_initial_cupcakes : 
  initial_cupcakes = total_cupcakes - additional_cupcakes + sold_cupcakes :=
by sorry

end carol_initial_cupcakes_l251_25191


namespace bird_migration_difference_l251_25189

/-- The number of bird families living near the mountain -/
def mountain_families : ℕ := 38

/-- The number of bird families that flew to Africa -/
def africa_families : ℕ := 47

/-- The number of bird families that flew to Asia -/
def asia_families : ℕ := 94

/-- Theorem: The difference between the number of bird families that flew to Asia
    and the number of bird families that flew to Africa is 47 -/
theorem bird_migration_difference :
  asia_families - africa_families = 47 := by
  sorry

end bird_migration_difference_l251_25189


namespace probability_three_kings_or_ace_value_l251_25161

/-- Represents a standard deck of cards --/
structure Deck :=
  (total_cards : ℕ)
  (queens : ℕ)
  (kings : ℕ)
  (aces : ℕ)

/-- The probability of drawing either three Kings or at least one Ace --/
def probability_three_kings_or_ace (d : Deck) : ℚ :=
  let p_three_kings := (d.kings : ℚ) / d.total_cards * (d.kings - 1) / (d.total_cards - 1) * (d.kings - 2) / (d.total_cards - 2)
  let p_no_aces := (d.total_cards - d.aces : ℚ) / d.total_cards * (d.total_cards - d.aces - 1) / (d.total_cards - 1) * (d.total_cards - d.aces - 2) / (d.total_cards - 2)
  p_three_kings + (1 - p_no_aces)

/-- The theorem to be proved --/
theorem probability_three_kings_or_ace_value :
  let d : Deck := ⟨52, 4, 4, 4⟩
  probability_three_kings_or_ace d = 961 / 4420 := by
  sorry


end probability_three_kings_or_ace_value_l251_25161


namespace market_prices_l251_25195

/-- The cost of one pound of rice in dollars -/
def rice_cost : ℝ := 0.33

/-- The number of eggs that cost the same as one pound of rice -/
def eggs_per_rice : ℕ := 1

/-- The number of eggs that cost the same as half a liter of kerosene -/
def eggs_per_half_liter : ℕ := 8

/-- The cost of one liter of kerosene in cents -/
def kerosene_cost : ℕ := 528

theorem market_prices :
  (rice_cost = rice_cost / eggs_per_rice) ∧
  (kerosene_cost = 2 * eggs_per_half_liter * rice_cost * 100) := by
sorry

end market_prices_l251_25195


namespace alice_coins_value_l251_25164

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a half-dollar in cents -/
def half_dollar_value : ℕ := 50

/-- The number of pennies Alice has -/
def num_pennies : ℕ := 2

/-- The number of nickels Alice has -/
def num_nickels : ℕ := 3

/-- The number of dimes Alice has -/
def num_dimes : ℕ := 4

/-- The number of half-dollars Alice has -/
def num_half_dollars : ℕ := 1

/-- The total value of Alice's coins in cents -/
def total_cents : ℕ :=
  num_pennies * penny_value +
  num_nickels * nickel_value +
  num_dimes * dime_value +
  num_half_dollars * half_dollar_value

/-- The value of one dollar in cents -/
def dollar_in_cents : ℕ := 100

theorem alice_coins_value :
  (total_cents : ℚ) / (dollar_in_cents : ℚ) = 107 / 100 := by
  sorry

end alice_coins_value_l251_25164


namespace optimal_pole_is_twelve_l251_25171

/-- Represents the number of intervals in the path -/
def intervals : ℕ := 28

/-- Represents Dodson's walking time for one interval (in minutes) -/
def dodson_walk_time : ℕ := 9

/-- Represents Williams' walking time for one interval (in minutes) -/
def williams_walk_time : ℕ := 11

/-- Represents the riding time on Bolivar for one interval (in minutes) -/
def bolivar_ride_time : ℕ := 3

/-- Calculates Dodson's total travel time given the pole number -/
def dodson_total_time (pole : ℕ) : ℚ :=
  (pole * bolivar_ride_time + (intervals - pole) * dodson_walk_time) / intervals

/-- Calculates Williams' total travel time given the pole number -/
def williams_total_time (pole : ℕ) : ℚ :=
  (pole * williams_walk_time + (intervals - pole) * bolivar_ride_time) / intervals

/-- Theorem stating that the 12th pole is the optimal point to tie Bolivar -/
theorem optimal_pole_is_twelve :
  ∃ (pole : ℕ), pole = 12 ∧
  ∀ (k : ℕ), 1 ≤ k ∧ k ≤ intervals →
    max (dodson_total_time pole) (williams_total_time pole) ≤
    max (dodson_total_time k) (williams_total_time k) :=
by sorry

end optimal_pole_is_twelve_l251_25171


namespace mary_hospital_time_l251_25179

/-- Given the conditions of Mary's ambulance ride and Don's drive to the hospital,
    prove that Mary reaches the hospital in 15 minutes. -/
theorem mary_hospital_time (ambulance_speed : ℝ) (don_speed : ℝ) (don_time : ℝ) :
  ambulance_speed = 60 →
  don_speed = 30 →
  don_time = 0.5 →
  (don_speed * don_time) / ambulance_speed = 0.25 := by
  sorry

#check mary_hospital_time

end mary_hospital_time_l251_25179


namespace no_snow_probability_l251_25166

theorem no_snow_probability (p1 p2 p3 : ℝ) 
  (h1 : p1 = 2/3)
  (h2 : p2 = 2/3)
  (h3 : p3 = 3/5)
  (h_independent : True)  -- Representing independence of events
  : (1 - p1) * (1 - p2) * (1 - p3) = 2/45 := by
  sorry

end no_snow_probability_l251_25166


namespace ferry_tourist_sum_l251_25130

/-- The number of trips made by the ferry -/
def num_trips : ℕ := 15

/-- The initial number of tourists -/
def initial_tourists : ℕ := 100

/-- The decrease in number of tourists per trip -/
def tourist_decrease : ℕ := 2

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (n : ℕ) (a : ℤ) (d : ℤ) : ℤ :=
  (n : ℤ) * (2 * a + (n - 1) * d) / 2

theorem ferry_tourist_sum :
  arithmetic_sum num_trips initial_tourists (-tourist_decrease) = 1290 :=
sorry

end ferry_tourist_sum_l251_25130


namespace tower_construction_modulo_l251_25107

/-- Represents the number of towers that can be built using cubes up to size n -/
def T : ℕ → ℕ
| 0 => 1
| 1 => 1
| 2 => 2
| (n + 1) => if n ≥ 2 then 4 * T n else 3 * T n

/-- The problem statement -/
theorem tower_construction_modulo :
  T 10 % 1000 = 304 := by
  sorry

end tower_construction_modulo_l251_25107


namespace hyperbola_equation_l251_25138

/-- The standard equation of a hyperbola with given foci and real axis length -/
theorem hyperbola_equation (x y : ℝ) : 
  let foci_distance : ℝ := 8
  let real_axis_length : ℝ := 4
  let a : ℝ := real_axis_length / 2
  let c : ℝ := foci_distance / 2
  let b_squared : ℝ := c^2 - a^2
  x^2 / a^2 - y^2 / b_squared = 1 := by
  sorry

end hyperbola_equation_l251_25138


namespace pencil_case_cost_solution_l251_25193

/-- Calculates the amount spent on a pencil case given the initial amount,
    the amount spent on a toy truck, and the remaining amount. -/
def pencil_case_cost (initial : ℝ) (toy_truck : ℝ) (remaining : ℝ) : ℝ :=
  initial - toy_truck - remaining

theorem pencil_case_cost_solution :
  pencil_case_cost 10 3 5 = 2 := by
  sorry

end pencil_case_cost_solution_l251_25193


namespace intersection_passes_through_center_l251_25184

-- Define the cube
def Cube : Type := Unit

-- Define a point in 3D space
def Point : Type := Unit

-- Define a plane
def Plane : Type := Unit

-- Define a hexagon
structure Hexagon :=
  (A B C D E F : Point)

-- Define the intersection of a cube and a plane
def intersection (c : Cube) (p : Plane) : Hexagon := sorry

-- Define the center of a cube
def center (c : Cube) : Point := sorry

-- Define a function to check if three lines intersect at a point
def intersect_at (p1 p2 p3 p4 p5 p6 : Point) (O : Point) : Prop := sorry

-- Theorem statement
theorem intersection_passes_through_center (c : Cube) (p : Plane) :
  let h := intersection c p
  intersect_at h.A h.D h.B h.E h.C h.F (center c) := by sorry

end intersection_passes_through_center_l251_25184


namespace division_simplification_l251_25187

theorem division_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  6 * x^3 * y^2 / (-3 * x * y) = -2 * x^2 * y := by
  sorry

end division_simplification_l251_25187


namespace floor_of_3_point_9_l251_25135

theorem floor_of_3_point_9 : ⌊(3.9 : ℝ)⌋ = 3 := by sorry

end floor_of_3_point_9_l251_25135


namespace weights_division_impossibility_l251_25165

theorem weights_division_impossibility : 
  let weights : List Nat := List.range 23
  let total_sum : Nat := (weights.sum + 23) - 21
  ¬ ∃ (half : Nat), 2 * half = total_sum
  := by sorry

end weights_division_impossibility_l251_25165


namespace right_angle_vector_condition_l251_25198

/-- Given two vectors OA and OB in a Cartesian coordinate plane, 
    if the angle ABO is 90 degrees, then the t-coordinate of OA is 5. -/
theorem right_angle_vector_condition (t : ℝ) : 
  let OA : ℝ × ℝ := (-1, t)
  let OB : ℝ × ℝ := (2, 2)
  (OB.1 * (OB.1 - OA.1) + OB.2 * (OB.2 - OA.2) = 0) →
  t = 5 := by
sorry

end right_angle_vector_condition_l251_25198


namespace pencils_in_drawer_l251_25188

/-- The total number of pencils after adding more -/
def total_pencils (initial : ℕ) (added : ℕ) : ℕ := initial + added

/-- Proof that the total number of pencils is 72 -/
theorem pencils_in_drawer : total_pencils 27 45 = 72 := by
  sorry

end pencils_in_drawer_l251_25188


namespace simultaneous_inequalities_l251_25110

theorem simultaneous_inequalities (a b : ℝ) :
  (a > b ∧ 1 / a > 1 / b) ↔ (a > 0 ∧ 0 > b) :=
sorry

end simultaneous_inequalities_l251_25110


namespace triangle_geometric_sequence_l251_25199

theorem triangle_geometric_sequence (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  -- a, b, c form a geometric sequence
  b^2 = a * c →
  -- cos B = 1/3
  Real.cos B = 1/3 →
  -- a/c = 1/2
  a / c = 1/2 →
  -- k is the first term of the geometric sequence
  ∃ k : ℝ, k > 0 ∧ a = k ∧ b = 2*k ∧ c = 4*k ∧ a + c = 5*k := by
sorry

end triangle_geometric_sequence_l251_25199


namespace triangle_side_length_l251_25153

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  C = 4 * A ∧  -- Given condition
  a = 30 ∧  -- Given side length
  c = 48 ∧  -- Given side length
  a / Real.sin A = c / Real.sin C ∧  -- Law of Sines
  b / Real.sin B = a / Real.sin A ∧  -- Law of Sines
  ∃ x : ℝ, 4 * x^3 - 4 * x - 8 / 5 = 0 ∧ x = Real.cos A  -- Equation for cosA
  →
  b = 30 * (5 - 20 * Real.sin A ^ 2 + 16 * Real.sin A ^ 4) :=
by sorry

end triangle_side_length_l251_25153


namespace subset_intersection_union_equivalence_l251_25144

theorem subset_intersection_union_equivalence (A B C : Set α) :
  (B ⊆ A ∧ C ⊆ A) ↔ ((A ∩ B) ∪ (A ∩ C) = B ∪ C) := by
  sorry

end subset_intersection_union_equivalence_l251_25144


namespace fish_aquarium_problem_l251_25127

theorem fish_aquarium_problem (x y : ℕ) :
  x + y = 100 ∧ x - 30 = y - 40 → x = 45 ∧ y = 55 := by
  sorry

end fish_aquarium_problem_l251_25127


namespace foil_covered_prism_width_l251_25111

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular prism -/
def volume (d : PrismDimensions) : ℝ := d.length * d.width * d.height

/-- Represents the properties of the inner prism -/
structure InnerPrism where
  dimensions : PrismDimensions
  cubeCount : ℕ

/-- Represents the properties of the foil-covered prism -/
structure FoilCoveredPrism where
  innerPrism : InnerPrism
  foilThickness : ℝ

theorem foil_covered_prism_width
  (p : FoilCoveredPrism)
  (h1 : p.innerPrism.cubeCount = 128)
  (h2 : p.innerPrism.dimensions.width = 2 * p.innerPrism.dimensions.length)
  (h3 : p.innerPrism.dimensions.width = 2 * p.innerPrism.dimensions.height)
  (h4 : volume p.innerPrism.dimensions = p.innerPrism.cubeCount)
  (h5 : p.foilThickness = 1) :
  p.innerPrism.dimensions.width + 2 * p.foilThickness = 10 := by
  sorry


end foil_covered_prism_width_l251_25111


namespace infinite_geometric_series_sum_problem_solution_l251_25163

def geometric_series (a : ℝ) (r : ℝ) : ℕ → ℝ := λ n => a * r^n

theorem infinite_geometric_series_sum (a : ℝ) (r : ℝ) (h : |r| < 1) :
  ∑' n, geometric_series a r n = a / (1 - r) :=
sorry

theorem problem_solution :
  ∑' n, geometric_series (1/4) (1/3) n = 3/8 :=
sorry

end infinite_geometric_series_sum_problem_solution_l251_25163


namespace negation_of_universal_proposition_l251_25190

open Real

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, 0 < x ∧ x < π / 2 → x < tan x)) ↔
  (∃ x : ℝ, 0 < x ∧ x < π / 2 ∧ x ≥ tan x) :=
by sorry

end negation_of_universal_proposition_l251_25190


namespace jelly_bean_difference_l251_25100

theorem jelly_bean_difference (total : ℕ) (vanilla : ℕ) (grape : ℕ) 
  (h1 : total = 770)
  (h2 : vanilla = 120)
  (h3 : total = grape + vanilla)
  (h4 : grape > 5 * vanilla) :
  grape - 5 * vanilla = 50 := by
  sorry

end jelly_bean_difference_l251_25100


namespace correct_first_grade_sample_size_l251_25122

/-- Represents a school with stratified sampling -/
structure School where
  total_students : ℕ
  first_grade_students : ℕ
  sample_size : ℕ

/-- Calculates the number of first-grade students to be selected in a stratified sample -/
def stratified_sample_size (school : School) : ℕ :=
  (school.first_grade_students * school.sample_size) / school.total_students

/-- Theorem stating the correct number of first-grade students to be selected -/
theorem correct_first_grade_sample_size (school : School) 
  (h1 : school.total_students = 2000)
  (h2 : school.first_grade_students = 400)
  (h3 : school.sample_size = 200) :
  stratified_sample_size school = 40 := by
  sorry

#eval stratified_sample_size { total_students := 2000, first_grade_students := 400, sample_size := 200 }

end correct_first_grade_sample_size_l251_25122


namespace starting_lineup_combinations_l251_25112

def total_players : ℕ := 15
def preselected_players : ℕ := 3
def lineup_size : ℕ := 5

theorem starting_lineup_combinations :
  Nat.choose (total_players - preselected_players) (lineup_size - preselected_players) = 66 :=
by sorry

end starting_lineup_combinations_l251_25112


namespace complement_of_A_l251_25194

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4}

-- Define set A
def A : Finset Nat := {1, 3}

-- Theorem statement
theorem complement_of_A :
  (U \ A) = {2, 4} := by sorry

end complement_of_A_l251_25194


namespace symmetry_line_l251_25154

/-- A circle in the 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a line in the form ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point is on a circle --/
def onCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Check if a point is on a line --/
def onLine (p : ℝ × ℝ) (l : Line) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- Check if two circles are symmetric with respect to a line --/
def symmetric (c1 c2 : Circle) (l : Line) : Prop :=
  ∀ p : ℝ × ℝ, onCircle p c1 → 
    ∃ q : ℝ × ℝ, onCircle q c2 ∧ onLine ((p.1 + q.1) / 2, (p.2 + q.2) / 2) l

/-- The main theorem --/
theorem symmetry_line : 
  let c1 : Circle := { center := (0, 0), radius := 3 }
  let c2 : Circle := { center := (2, -2), radius := 3 }
  let l : Line := { a := 1, b := -1, c := -2 }
  symmetric c1 c2 l := by sorry

end symmetry_line_l251_25154


namespace watermelon_problem_l251_25120

theorem watermelon_problem (selling_price : ℕ) (total_profit : ℕ) (watermelons_left : ℕ) :
  selling_price = 3 →
  total_profit = 105 →
  watermelons_left = 18 →
  selling_price * ((total_profit / selling_price) + watermelons_left) = 53 * selling_price :=
by sorry

end watermelon_problem_l251_25120


namespace six_eight_ten_pythagorean_triple_l251_25159

/-- A Pythagorean triple is a set of three positive integers (a, b, c) that satisfies a² + b² = c² --/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

/-- The set (6, 8, 10) is a Pythagorean triple --/
theorem six_eight_ten_pythagorean_triple : is_pythagorean_triple 6 8 10 := by
  sorry

end six_eight_ten_pythagorean_triple_l251_25159


namespace remainder_of_sum_mod_11_l251_25180

theorem remainder_of_sum_mod_11 : (100001 + 100002 + 100003 + 100004 + 100005 + 100006 + 100007) % 11 = 2 := by
  sorry

end remainder_of_sum_mod_11_l251_25180


namespace compound_molecular_weight_l251_25197

/-- The atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.01

/-- The atomic weight of Bromine in g/mol -/
def atomic_weight_Br : ℝ := 79.90

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Hydrogen atoms in the compound -/
def num_H : ℕ := 1

/-- The number of Bromine atoms in the compound -/
def num_Br : ℕ := 1

/-- The number of Oxygen atoms in the compound -/
def num_O : ℕ := 3

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 
  num_H * atomic_weight_H + num_Br * atomic_weight_Br + num_O * atomic_weight_O

/-- Theorem stating that the molecular weight of the compound is 128.91 g/mol -/
theorem compound_molecular_weight : molecular_weight = 128.91 := by
  sorry

end compound_molecular_weight_l251_25197


namespace base4_multiplication_division_l251_25178

-- Define a function to convert from base 4 to base 10
def base4ToBase10 (n : ℕ) : ℕ := sorry

-- Define a function to convert from base 10 to base 4
def base10ToBase4 (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem base4_multiplication_division :
  base10ToBase4 (base4ToBase10 132 * base4ToBase10 22 / base4ToBase10 3) = 154 := by sorry

end base4_multiplication_division_l251_25178


namespace adams_account_balance_l251_25105

theorem adams_account_balance :
  let initial_savings : ℚ := 1579.37
  let monday_earnings : ℚ := 21.85
  let tuesday_earnings : ℚ := 33.28
  let wednesday_spending : ℚ := 87.41
  let final_balance : ℚ := initial_savings + monday_earnings + tuesday_earnings - wednesday_spending
  final_balance = 1547.09 := by sorry

end adams_account_balance_l251_25105


namespace smallest_divisor_exponent_l251_25172

def polynomial (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_divisor_exponent :
  ∃ (k : ℕ), k > 0 ∧ 
  (∀ (z : ℂ), polynomial z = 0 → z^k = 1) ∧
  (∀ (m : ℕ), m > 0 → m < k → ∃ (w : ℂ), polynomial w = 0 ∧ w^m ≠ 1) ∧
  k = 120 :=
sorry

end smallest_divisor_exponent_l251_25172


namespace cube_root_floor_product_limit_l251_25152

def cube_root_floor_product (n : ℕ) : ℚ :=
  (Finset.range n).prod (λ i => ⌊(3 * i + 1 : ℚ)^(1/3)⌋) /
  (Finset.range n).prod (λ i => ⌊(3 * i + 2 : ℚ)^(1/3)⌋)

theorem cube_root_floor_product_limit : 
  cube_root_floor_product 167 = 1/8 := by sorry

end cube_root_floor_product_limit_l251_25152


namespace square_of_1035_l251_25134

theorem square_of_1035 : (1035 : ℕ)^2 = 1071225 := by
  sorry

end square_of_1035_l251_25134


namespace pebbles_distribution_l251_25137

/-- The number of pebbles in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of pebbles Janice had -/
def janice_dozens : ℕ := 3

/-- The total number of pebbles Janice had -/
def total_pebbles : ℕ := janice_dozens * dozen

/-- The number of friends who received pebbles -/
def num_friends : ℕ := 9

/-- The number of pebbles each friend received -/
def pebbles_per_friend : ℕ := total_pebbles / num_friends

theorem pebbles_distribution :
  pebbles_per_friend = 4 :=
sorry

end pebbles_distribution_l251_25137


namespace complex_number_in_second_quadrant_l251_25170

/-- Proves that the complex number z = (-8 - 7i)(-3i) is located in the second quadrant of the complex plane. -/
theorem complex_number_in_second_quadrant : 
  let z : ℂ := (-8 - 7*I) * (-3*I)
  (z.re < 0 ∧ z.im > 0) := by sorry

end complex_number_in_second_quadrant_l251_25170


namespace cos_B_value_l251_25157

-- Define the angle B
def B : ℝ := sorry

-- Define the conditions
def B_in_third_quadrant : 3 * π / 2 < B ∧ B < 2 * π := sorry
def sin_B : Real.sin B = -5/13 := sorry

-- Theorem to prove
theorem cos_B_value : Real.cos B = -12/13 := by sorry

end cos_B_value_l251_25157


namespace flies_needed_for_week_l251_25156

/-- The number of flies a frog eats per day -/
def flies_per_day : ℕ := 2

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of flies Betty caught in total -/
def flies_caught : ℕ := 11

/-- The number of flies that escaped -/
def flies_escaped : ℕ := 1

/-- Theorem stating how many more flies Betty needs for a week -/
theorem flies_needed_for_week : 
  flies_per_day * days_in_week - (flies_caught - flies_escaped) = 4 := by
  sorry

end flies_needed_for_week_l251_25156


namespace distance_between_circle_centers_l251_25151

/-- The distance between the centers of two circles with polar equations ρ = 2cos(θ) and ρ = 4sin(θ) is √5. -/
theorem distance_between_circle_centers :
  let circle1 : ℝ → ℝ := fun θ ↦ 2 * Real.cos θ
  let circle2 : ℝ → ℝ := fun θ ↦ 4 * Real.sin θ
  let center1 : ℝ × ℝ := (1, 0)
  let center2 : ℝ × ℝ := (0, 2)
  (center1.1 - center2.1)^2 + (center1.2 - center2.2)^2 = 5 := by
  sorry

#check distance_between_circle_centers

end distance_between_circle_centers_l251_25151


namespace least_years_to_double_l251_25102

-- Define the interest rate
def interest_rate : ℝ := 0.5

-- Define the function for the amount after t years
def amount (t : ℕ) : ℝ := (1 + interest_rate) ^ t

-- Theorem statement
theorem least_years_to_double :
  ∀ t : ℕ, t < 2 → amount t ≤ 2 ∧ 2 < amount 2 :=
by sorry

end least_years_to_double_l251_25102


namespace smallest_undefined_inverse_l251_25108

theorem smallest_undefined_inverse (a : ℕ) : a > 0 ∧ 
  (∀ b : ℕ, b < a → (Nat.gcd b 60 = 1 ∨ Nat.gcd b 75 = 1)) ∧
  Nat.gcd a 60 ≠ 1 ∧ Nat.gcd a 75 ≠ 1 → a = 15 := by
  sorry

end smallest_undefined_inverse_l251_25108


namespace polynomial_sum_l251_25169

-- Define the polynomial g(x)
def g (a b c d : ℝ) (x : ℂ) : ℂ := x^4 + a*x^3 + b*x^2 + c*x + d

-- State the theorem
theorem polynomial_sum (a b c d : ℝ) :
  g a b c d (1 + I) = 0 → g a b c d (3*I) = 0 → a + b + c + d = 27 := by
  sorry

end polynomial_sum_l251_25169


namespace remaining_movies_to_watch_l251_25168

theorem remaining_movies_to_watch (total_movies watched_movies : ℕ) : 
  total_movies = 8 → watched_movies = 4 → total_movies - watched_movies = 4 :=
by sorry

end remaining_movies_to_watch_l251_25168


namespace selected_students_l251_25175

-- Define the set of students
inductive Student : Type
| A | B | C | D | E

-- Define a type for the selection of students
def Selection := Student → Prop

-- Define the conditions
def valid_selection (s : Selection) : Prop :=
  -- 3 students are selected
  (∃ (x y z : Student), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ s x ∧ s y ∧ s z ∧
    ∀ (w : Student), s w → (w = x ∨ w = y ∨ w = z)) ∧
  -- If A is selected, then B is selected and E is not selected
  (s Student.A → s Student.B ∧ ¬s Student.E) ∧
  -- If B or E is selected, then D is not selected
  ((s Student.B ∨ s Student.E) → ¬s Student.D) ∧
  -- At least one of C or D must be selected
  (s Student.C ∨ s Student.D)

-- Theorem statement
theorem selected_students (s : Selection) :
  valid_selection s → s Student.A → s Student.B ∧ s Student.C :=
by sorry

end selected_students_l251_25175


namespace set_B_equality_l251_25114

def A : Set ℤ := {-1, 0, 1}

def B : Set ℤ := {x | ∃ t ∈ A, x = t^2}

theorem set_B_equality : B = {0, 1} := by sorry

end set_B_equality_l251_25114


namespace fractional_inequality_solution_set_l251_25149

theorem fractional_inequality_solution_set (x : ℝ) : 
  (x - 1) / (x + 2) > 1 ↔ x < -2 := by
  sorry

end fractional_inequality_solution_set_l251_25149


namespace polynomial_property_l251_25147

theorem polynomial_property (P : ℤ → ℤ) (h_poly : ∀ a b : ℤ, ∃ c : ℤ, P a - P b = c * (a - b)) :
  P 1 = 2019 →
  P 2019 = 1 →
  ∃ k : ℤ, P k = k →
  k = 1010 :=
sorry

end polynomial_property_l251_25147


namespace triangle_equilateral_l251_25142

theorem triangle_equilateral (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)
  (condition : a^2 + b^2 + 2*c^2 - 2*a*c - 2*b*c = 0) :
  a = b ∧ b = c := by
  sorry

end triangle_equilateral_l251_25142
