import Mathlib

namespace ratio_of_distances_l1279_127949

/-- Given four points P, Q, R, and S on a line (in that order), with distances PQ = 3, QR = 7, and PS = 22,
    prove that the ratio of PR to QS is 10/19. -/
theorem ratio_of_distances (P Q R S : ℝ) (h_order : P < Q ∧ Q < R ∧ R < S) 
  (h_PQ : Q - P = 3) (h_QR : R - Q = 7) (h_PS : S - P = 22) : 
  (R - P) / (S - Q) = 10 / 19 := by
  sorry

end ratio_of_distances_l1279_127949


namespace set_A_proof_l1279_127935

def U : Set ℕ := {1, 3, 5, 7, 9}

theorem set_A_proof (A B : Set ℕ) 
  (h1 : A ⊆ U)
  (h2 : B ⊆ U)
  (h3 : A ∩ B = {3, 5})
  (h4 : A ∩ (U \ B) = {9}) :
  A = {3, 5, 9} := by
  sorry


end set_A_proof_l1279_127935


namespace perpendicular_lines_b_value_l1279_127979

-- Define the slopes of the two lines
def slope1 : ℚ := 3 / 4
def slope2 (b : ℚ) : ℚ := -b / 2

-- Define the perpendicularity condition
def perpendicular (b : ℚ) : Prop := slope1 * slope2 b = -1

-- Theorem statement
theorem perpendicular_lines_b_value :
  ∃ b : ℚ, perpendicular b ∧ b = 8 / 3 :=
sorry

end perpendicular_lines_b_value_l1279_127979


namespace martha_children_count_l1279_127928

theorem martha_children_count (total_cakes : ℕ) (cakes_per_child : ℕ) (h1 : total_cakes = 18) (h2 : cakes_per_child = 6) : 
  total_cakes / cakes_per_child = 3 :=
by sorry

end martha_children_count_l1279_127928


namespace frog_expected_returns_l1279_127909

/-- Represents the probability of moving in a certain direction or getting eaten -/
def move_probability : ℚ := 1 / 3

/-- Represents the frog's position on the number line -/
def Position : Type := ℤ

/-- Calculates the probability of returning to the starting position from a given position -/
noncomputable def prob_return_to_start (pos : Position) : ℝ :=
  sorry

/-- Calculates the expected number of returns to the starting position before getting eaten -/
noncomputable def expected_returns : ℝ :=
  sorry

/-- The main theorem stating the expected number of returns -/
theorem frog_expected_returns :
  expected_returns = (3 * Real.sqrt 5 - 5) / 5 := by
  sorry

end frog_expected_returns_l1279_127909


namespace solution_difference_l1279_127908

theorem solution_difference (r s : ℝ) : 
  ((r - 4) * (r + 4) = 24 * r - 96) →
  ((s - 4) * (s + 4) = 24 * s - 96) →
  r ≠ s →
  r > s →
  r - s = 16 := by
sorry

end solution_difference_l1279_127908


namespace matrix_power_four_l1279_127960

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_power_four :
  A^4 = !![0, -9; 9, -9] := by sorry

end matrix_power_four_l1279_127960


namespace kath_group_cost_l1279_127991

/-- Calculates the total cost for a group watching a movie with early showing discount -/
def total_cost (standard_price : ℕ) (discount : ℕ) (group_size : ℕ) : ℕ :=
  (standard_price - discount) * group_size

/-- Theorem: The total cost for Kath's group is $30 -/
theorem kath_group_cost :
  let standard_price : ℕ := 8
  let early_discount : ℕ := 3
  let group_size : ℕ := 6
  total_cost standard_price early_discount group_size = 30 := by
  sorry

end kath_group_cost_l1279_127991


namespace quadratic_roots_sum_l1279_127975

theorem quadratic_roots_sum (α β : ℝ) : 
  α^2 + 2*α - 2024 = 0 → 
  β^2 + 2*β - 2024 = 0 → 
  α^2 + 3*α + β = 2022 := by
sorry

end quadratic_roots_sum_l1279_127975


namespace decimal_88_to_base5_base5_323_to_decimal_decimal_88_equals_base5_323_l1279_127980

/-- Converts a natural number to its base-5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

/-- Converts a list of digits in base-5 to its decimal representation -/
def fromBase5 (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => d + 5 * acc) 0

theorem decimal_88_to_base5 :
  toBase5 88 = [3, 2, 3] :=
sorry

theorem base5_323_to_decimal :
  fromBase5 [3, 2, 3] = 88 :=
sorry

/-- The base-5 representation of 88 is 323 -/
theorem decimal_88_equals_base5_323 :
  toBase5 88 = [3, 2, 3] ∧ fromBase5 [3, 2, 3] = 88 :=
sorry

end decimal_88_to_base5_base5_323_to_decimal_decimal_88_equals_base5_323_l1279_127980


namespace gcd_1963_1891_l1279_127942

theorem gcd_1963_1891 : Nat.gcd 1963 1891 = 1 := by
  sorry

end gcd_1963_1891_l1279_127942


namespace altitude_least_integer_l1279_127998

theorem altitude_least_integer (a b : ℝ) (h : a = 5 ∧ b = 12) : 
  ∃ (L : ℝ), L = (a * b) / (2 * Real.sqrt (a^2 + b^2)) ∧ 
  (∀ (n : ℤ), (n : ℝ) > L → n ≥ 5) ∧ (4 : ℝ) < L :=
sorry

end altitude_least_integer_l1279_127998


namespace min_sum_squares_l1279_127963

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*y = 0

-- Define the line
def line (x y : ℝ) : Prop := y = x - 1

-- Define a point on the line
structure PointOnLine where
  x : ℝ
  y : ℝ
  on_line : line x y

-- Define the diameter AB
structure Diameter where
  A : ℝ × ℝ
  B : ℝ × ℝ
  is_diameter : ∀ (x y : ℝ), circle_C x y → 
    (x - A.1)^2 + (y - A.2)^2 + (x - B.1)^2 + (y - B.2)^2 = 4

-- Theorem statement
theorem min_sum_squares (d : Diameter) :
  ∃ (min : ℝ), min = 6 ∧ 
  ∀ (P : PointOnLine), 
    (P.x - d.A.1)^2 + (P.y - d.A.2)^2 + (P.x - d.B.1)^2 + (P.y - d.B.2)^2 ≥ min :=
sorry

end min_sum_squares_l1279_127963


namespace external_diagonals_invalid_l1279_127995

theorem external_diagonals_invalid (a b c : ℝ) : 
  a = 4 ∧ b = 6 ∧ c = 8 →
  ¬(a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ a^2 + c^2 > b^2) :=
by sorry

end external_diagonals_invalid_l1279_127995


namespace max_value_expression_l1279_127983

theorem max_value_expression (x₁ x₂ x₃ x₄ : ℝ)
  (h₁ : 0 ≤ x₁ ∧ x₁ ≤ 1)
  (h₂ : 0 ≤ x₂ ∧ x₂ ≤ 1)
  (h₃ : 0 ≤ x₃ ∧ x₃ ≤ 1)
  (h₄ : 0 ≤ x₄ ∧ x₄ ≤ 1) :
  (∀ y₁ y₂ y₃ y₄ : ℝ,
    0 ≤ y₁ ∧ y₁ ≤ 1 →
    0 ≤ y₂ ∧ y₂ ≤ 1 →
    0 ≤ y₃ ∧ y₃ ≤ 1 →
    0 ≤ y₄ ∧ y₄ ≤ 1 →
    1 - (1 - y₁) * (1 - y₂) * (1 - y₃) * (1 - y₄) ≤ 1 - (1 - x₁) * (1 - x₂) * (1 - x₃) * (1 - x₄)) ∧
  1 - (1 - x₁) * (1 - x₂) * (1 - x₃) * (1 - x₄) = 1 :=
by sorry

end max_value_expression_l1279_127983


namespace absolute_value_at_two_l1279_127956

/-- A third-degree polynomial with real coefficients -/
def ThirdDegreePolynomial : Type := ℝ → ℝ

/-- Condition that g is a third-degree polynomial with specific absolute values -/
def SatisfiesCondition (g : ThirdDegreePolynomial) : Prop :=
  ∃ (a b c d : ℝ), ∀ x, g x = a * x^3 + b * x^2 + c * x + d ∧
  (|g 0| = 10) ∧ (|g 1| = 10) ∧ (|g 3| = 10) ∧
  (|g 4| = 10) ∧ (|g 5| = 10) ∧ (|g 8| = 10)

/-- Theorem stating that if g satisfies the condition, then |g(2)| = 20 -/
theorem absolute_value_at_two
  (g : ThirdDegreePolynomial)
  (h : SatisfiesCondition g) :
  |g 2| = 20 := by
  sorry

end absolute_value_at_two_l1279_127956


namespace symmetric_probability_l1279_127934

/-- Represents a standard six-sided die -/
def StandardDie : Type := Fin 6

/-- The number of dice rolled -/
def numDice : Nat := 8

/-- The sum we're comparing to -/
def givenSum : Nat := 15

/-- The sum we're proving has the same probability -/
def symmetricSum : Nat := 41

/-- Function to calculate the probability of a specific sum when rolling n dice -/
noncomputable def probability (n : Nat) (sum : Nat) : Real := sorry

theorem symmetric_probability : 
  probability numDice givenSum = probability numDice symmetricSum := by sorry

end symmetric_probability_l1279_127934


namespace bianca_albums_l1279_127930

theorem bianca_albums (total_pics : ℕ) (main_album_pics : ℕ) (pics_per_album : ℕ) : 
  total_pics = 33 → main_album_pics = 27 → pics_per_album = 2 → 
  (total_pics - main_album_pics) / pics_per_album = 3 := by
  sorry

end bianca_albums_l1279_127930


namespace icosahedron_edges_l1279_127948

/-- A regular icosahedron is a convex polyhedron with 20 congruent equilateral triangular faces -/
def RegularIcosahedron : Type := sorry

/-- The number of edges in a polyhedron -/
def num_edges (p : RegularIcosahedron) : ℕ := sorry

/-- Theorem: A regular icosahedron has 30 edges -/
theorem icosahedron_edges :
  ∀ (i : RegularIcosahedron), num_edges i = 30 := by sorry

end icosahedron_edges_l1279_127948


namespace triangle_count_l1279_127920

def stick_lengths : List ℕ := [1, 2, 3, 4, 5]

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def count_valid_triangles (lengths : List ℕ) : ℕ :=
  (lengths.toFinset.powerset.filter (fun s => s.card = 3)).card

theorem triangle_count : count_valid_triangles stick_lengths = 22 := by
  sorry

end triangle_count_l1279_127920


namespace mod_congruence_unique_n_l1279_127939

theorem mod_congruence_unique_n (a b : ℤ) 
  (ha : a ≡ 22 [ZMOD 50])
  (hb : b ≡ 78 [ZMOD 50]) :
  ∃! n : ℤ, 150 ≤ n ∧ n ≤ 201 ∧ (a - b) ≡ n [ZMOD 50] ∧ n = 194 :=
sorry

end mod_congruence_unique_n_l1279_127939


namespace cooldrink_mixture_l1279_127992

/-- Amount of Cool-drink B added to create a mixture with 10% jasmine water -/
theorem cooldrink_mixture (total_volume : ℝ) (cooldrink_a_volume : ℝ) (jasmine_water_added : ℝ) (fruit_juice_added : ℝ)
  (cooldrink_a_jasmine_percent : ℝ) (cooldrink_a_fruit_percent : ℝ)
  (cooldrink_b_jasmine_percent : ℝ) (cooldrink_b_fruit_percent : ℝ)
  (final_jasmine_percent : ℝ) :
  total_volume = 150 →
  cooldrink_a_volume = 80 →
  jasmine_water_added = 8 →
  fruit_juice_added = 20 →
  cooldrink_a_jasmine_percent = 0.12 →
  cooldrink_a_fruit_percent = 0.88 →
  cooldrink_b_jasmine_percent = 0.05 →
  cooldrink_b_fruit_percent = 0.95 →
  final_jasmine_percent = 0.10 →
  ∃ cooldrink_b_volume : ℝ,
    cooldrink_b_volume = 136 ∧
    (cooldrink_a_volume * cooldrink_a_jasmine_percent + cooldrink_b_volume * cooldrink_b_jasmine_percent + jasmine_water_added) / 
    (cooldrink_a_volume + cooldrink_b_volume + jasmine_water_added + fruit_juice_added) = final_jasmine_percent :=
by
  sorry

end cooldrink_mixture_l1279_127992


namespace left_handed_jazz_lovers_count_l1279_127921

/-- Represents a club with members and their characteristics -/
structure Club where
  total_members : ℕ
  left_handed : ℕ
  jazz_lovers : ℕ
  right_handed_non_jazz : ℕ

/-- The number of left-handed jazz lovers in the club -/
def left_handed_jazz_lovers (c : Club) : ℕ :=
  c.total_members + c.right_handed_non_jazz - c.left_handed - c.jazz_lovers

/-- Theorem stating the number of left-handed jazz lovers in the given club -/
theorem left_handed_jazz_lovers_count (c : Club) 
  (h1 : c.total_members = 30)
  (h2 : c.left_handed = 12)
  (h3 : c.jazz_lovers = 22)
  (h4 : c.right_handed_non_jazz = 4)
  (h5 : c.left_handed + (c.total_members - c.left_handed) = c.total_members) :
  left_handed_jazz_lovers c = 8 := by
  sorry

#eval left_handed_jazz_lovers ⟨30, 12, 22, 4⟩

end left_handed_jazz_lovers_count_l1279_127921


namespace train_crossing_time_l1279_127927

/-- Calculates the time taken for a train to cross a platform -/
theorem train_crossing_time (train_length : Real) (signal_cross_time : Real) (platform_length : Real) :
  train_length = 300 ∧ 
  signal_cross_time = 16 ∧ 
  platform_length = 431.25 →
  (train_length + platform_length) / (train_length / signal_cross_time) = 39 := by
  sorry

#check train_crossing_time

end train_crossing_time_l1279_127927


namespace total_savings_theorem_l1279_127910

/-- The amount of money saved per month in dollars -/
def monthly_savings : ℕ := 4000

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- Theorem: If Abigail saves $4,000 every month for an entire year, 
    the total amount saved will be $48,000 -/
theorem total_savings_theorem : 
  monthly_savings * months_in_year = 48000 := by
  sorry

end total_savings_theorem_l1279_127910


namespace rectangular_box_surface_area_l1279_127933

theorem rectangular_box_surface_area 
  (a b c : ℝ) 
  (h1 : 4 * a + 4 * b + 4 * c = 180) 
  (h2 : Real.sqrt (a^2 + b^2 + c^2) = 25) 
  (h3 : a = 10) : 
  2 * (a * b + b * c + c * a) = 1400 := by
sorry

end rectangular_box_surface_area_l1279_127933


namespace emily_bought_two_skirts_l1279_127976

def cost_of_art_supplies : ℕ := 20
def total_spent : ℕ := 50
def cost_per_skirt : ℕ := 15

def number_of_skirts : ℕ := (total_spent - cost_of_art_supplies) / cost_per_skirt

theorem emily_bought_two_skirts : number_of_skirts = 2 := by
  sorry

end emily_bought_two_skirts_l1279_127976


namespace bad_carrots_count_l1279_127901

theorem bad_carrots_count (faye_carrots : ℕ) (mom_carrots : ℕ) (good_carrots : ℕ) :
  faye_carrots = 23 →
  mom_carrots = 5 →
  good_carrots = 12 →
  faye_carrots + mom_carrots - good_carrots = 16 :=
by
  sorry

end bad_carrots_count_l1279_127901


namespace red_light_probability_l1279_127925

/-- Represents the duration of each traffic light color in seconds -/
structure TrafficLightDuration where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the probability of seeing a red light -/
def probabilityRedLight (d : TrafficLightDuration) : ℚ :=
  d.red / (d.red + d.yellow + d.green)

/-- Theorem: The probability of seeing a red light is 2/5 for the given durations -/
theorem red_light_probability (d : TrafficLightDuration) 
  (h1 : d.red = 30) 
  (h2 : d.yellow = 5) 
  (h3 : d.green = 40) : 
  probabilityRedLight d = 2/5 := by
  sorry

#eval probabilityRedLight ⟨30, 5, 40⟩

end red_light_probability_l1279_127925


namespace min_product_of_three_l1279_127944

def S : Finset Int := {-8, -6, -4, 0, 3, 5, 7}

theorem min_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hdiff : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ (x y z : Int), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  x * y * z = -280 ∧ 
  ∀ (p q r : Int), p ∈ S → q ∈ S → r ∈ S → p ≠ q → q ≠ r → p ≠ r → 
  p * q * r ≥ -280 := by
sorry

end min_product_of_three_l1279_127944


namespace only_45_increases_ninefold_l1279_127943

/-- A function that inserts a zero between the tens and units digits of a natural number -/
def insertZero (n : ℕ) : ℕ :=
  10 * (n / 10) * 10 + n % 10

/-- The property that a number increases ninefold when a zero is inserted between its digits -/
def increasesNinefold (n : ℕ) : Prop :=
  insertZero n = 9 * n

theorem only_45_increases_ninefold :
  ∀ n : ℕ, n ≠ 0 → (increasesNinefold n ↔ n = 45) :=
sorry

end only_45_increases_ninefold_l1279_127943


namespace theater_seat_count_l1279_127951

/-- Represents a theater with an arithmetic progression of seats per row -/
structure Theater where
  first_row_seats : ℕ
  seat_increase : ℕ
  last_row_seats : ℕ

/-- Calculates the number of rows in the theater -/
def number_of_rows (t : Theater) : ℕ :=
  (t.last_row_seats - t.first_row_seats) / t.seat_increase + 1

/-- Calculates the total number of seats in the theater -/
def total_seats (t : Theater) : ℕ :=
  let n := number_of_rows t
  n * (t.first_row_seats + t.last_row_seats) / 2

/-- The theater described in the problem -/
def problem_theater : Theater :=
  { first_row_seats := 15
  , seat_increase := 2
  , last_row_seats := 53 }

theorem theater_seat_count :
  total_seats problem_theater = 680 := by
  sorry

end theater_seat_count_l1279_127951


namespace profit_equation_l1279_127911

/-- The profit function for a commodity -/
def profit (x : ℝ) : ℝ :=
  let cost_price : ℝ := 30
  let quantity_sold : ℝ := 200 - x
  (x - cost_price) * quantity_sold

theorem profit_equation (x : ℝ) : profit x = -x^2 + 230*x - 6000 := by
  sorry

end profit_equation_l1279_127911


namespace matrix_equation_solution_l1279_127957

theorem matrix_equation_solution : 
  let N : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, 2]
  N^3 - 3 • N^2 + 2 • N = !![2, 6; 3, 1] :=
by sorry

end matrix_equation_solution_l1279_127957


namespace problem_statement_l1279_127931

open Real

variable (a b : ℝ)

theorem problem_statement (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a + 1/a) * (b + 1/b) > 4 ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → sqrt (1 + a) + sqrt (1 + b) ≤ sqrt 6) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → (a * b + 4 * a + b) / (4 * a + b) ≤ 10 / 9) :=
by sorry

end problem_statement_l1279_127931


namespace product_of_odd_negative_integers_l1279_127907

def odd_negative_integers : List ℤ := sorry

theorem product_of_odd_negative_integers :
  let product := (List.prod odd_negative_integers)
  (product < 0) ∧ (product % 10 = -5) := by
  sorry

end product_of_odd_negative_integers_l1279_127907


namespace number_forms_and_products_l1279_127986

theorem number_forms_and_products (n m : ℕ) :
  -- Part 1: Any number not divisible by 2 or 3 is of the form 6n + 1 or 6n + 5
  (∀ k : ℤ, (¬(2 ∣ k) ∧ ¬(3 ∣ k)) → (∃ n : ℕ, k = 6*n + 1 ∨ k = 6*n + 5)) ∧
  
  -- Part 2: Product of two numbers of form 6n + 1 or 6n + 5 is of form 6m + 1
  ((6*n + 1) * (6*m + 1) ≡ 1 [MOD 6]) ∧
  ((6*n + 5) * (6*m + 5) ≡ 1 [MOD 6]) ∧
  
  -- Part 3: Product of 6n + 1 and 6n + 5 is of form 6m + 5
  ((6*n + 1) * (6*m + 5) ≡ 5 [MOD 6]) :=
by sorry


end number_forms_and_products_l1279_127986


namespace power_mod_thirteen_l1279_127913

theorem power_mod_thirteen : (6 ^ 1234 : ℕ) % 13 = 10 := by sorry

end power_mod_thirteen_l1279_127913


namespace problem_statement_l1279_127915

theorem problem_statement (x : ℝ) (h : x + 1/x = 7) :
  (x - 3)^2 + 36/(x - 3)^2 = 12.375 := by sorry

end problem_statement_l1279_127915


namespace tshirt_packages_l1279_127904

theorem tshirt_packages (total_tshirts : ℕ) (tshirts_per_package : ℕ) 
  (h1 : total_tshirts = 56)
  (h2 : tshirts_per_package = 2) :
  total_tshirts / tshirts_per_package = 28 :=
by
  sorry

end tshirt_packages_l1279_127904


namespace lcm_12_18_l1279_127953

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l1279_127953


namespace tom_tim_ratio_l1279_127964

structure TypingSpeed where
  tim : ℝ
  tom : ℝ

def combined_speed (s : TypingSpeed) : ℝ := s.tim + s.tom

def increased_speed (s : TypingSpeed) : ℝ := s.tim + 1.4 * s.tom

theorem tom_tim_ratio (s : TypingSpeed) 
  (h1 : combined_speed s = 20)
  (h2 : increased_speed s = 24) : 
  s.tom / s.tim = 1 := by
  sorry

end tom_tim_ratio_l1279_127964


namespace scientific_notation_pm25_l1279_127924

theorem scientific_notation_pm25 :
  ∃ (a : ℝ) (n : ℤ), 0.000042 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 4.2 ∧ n = -5 :=
sorry

end scientific_notation_pm25_l1279_127924


namespace bus_time_is_ten_l1279_127981

/-- Represents the travel times and conditions for Xiaoming's journey --/
structure TravelTimes where
  total : ℕ  -- Total travel time
  transfer : ℕ  -- Transfer time
  subway_only : ℕ  -- Time if only taking subway
  bus_only : ℕ  -- Time if only taking bus

/-- Calculates the time spent on the bus given the travel times --/
def time_on_bus (t : TravelTimes) : ℕ :=
  let actual_travel_time := t.total - t.transfer
  let extra_time := actual_travel_time - t.subway_only
  let time_unit := extra_time / (t.bus_only / 10 - t.subway_only / 10)
  (t.bus_only / 10) * time_unit

/-- Theorem stating that given the specific travel times, the time spent on the bus is 10 minutes --/
theorem bus_time_is_ten : 
  let t : TravelTimes := { 
    total := 40, 
    transfer := 6, 
    subway_only := 30, 
    bus_only := 50 
  }
  time_on_bus t = 10 := by
  sorry


end bus_time_is_ten_l1279_127981


namespace decimal_addition_l1279_127958

theorem decimal_addition : (0.9 : ℝ) + 0.99 = 1.89 := by
  sorry

end decimal_addition_l1279_127958


namespace smallest_t_is_four_l1279_127969

def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def smallest_valid_t : ℕ → Prop
  | t => is_valid_triangle 7.5 11 (t : ℝ) ∧ 
         ∀ k : ℕ, k < t → ¬is_valid_triangle 7.5 11 (k : ℝ)

theorem smallest_t_is_four : smallest_valid_t 4 := by
  sorry

end smallest_t_is_four_l1279_127969


namespace inheritance_tax_problem_l1279_127926

theorem inheritance_tax_problem (x : ℝ) : 
  0.25 * x + 0.12 * (0.75 * x) = 13600 → x = 40000 := by
  sorry

end inheritance_tax_problem_l1279_127926


namespace stop_duration_l1279_127988

/-- Calculates the duration of a stop given the total distance, speed, and total travel time. -/
theorem stop_duration (distance : ℝ) (speed : ℝ) (total_time : ℝ) 
  (h1 : distance = 360) 
  (h2 : speed = 60) 
  (h3 : total_time = 7) :
  total_time - distance / speed = 1 := by
  sorry

end stop_duration_l1279_127988


namespace divisible_number_is_six_l1279_127946

/-- The number of three-digit numbers divisible by the specific number -/
def divisible_count : ℕ := 150

/-- The lower bound of three-digit numbers -/
def lower_bound : ℕ := 100

/-- The upper bound of three-digit numbers -/
def upper_bound : ℕ := 999

/-- The total count of three-digit numbers -/
def total_count : ℕ := upper_bound - lower_bound + 1

theorem divisible_number_is_six :
  ∃ (n : ℕ), n = 6 ∧
  (∀ k : ℕ, lower_bound ≤ k ∧ k ≤ upper_bound →
    (divisible_count * n = total_count)) :=
sorry

end divisible_number_is_six_l1279_127946


namespace sum_of_fifth_terms_l1279_127916

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem sum_of_fifth_terms (a b : ℕ → ℝ) :
  is_arithmetic_sequence a →
  is_geometric_sequence b →
  a 1 + b 1 = 3 →
  a 2 + b 2 = 7 →
  a 3 + b 3 = 15 →
  a 4 + b 4 = 35 →
  a 5 + b 5 = 91 :=
by sorry

end sum_of_fifth_terms_l1279_127916


namespace expression_equals_36_l1279_127962

theorem expression_equals_36 (k : ℚ) : k = 13 → k * (3 - 3 / k) = 36 := by
  sorry

end expression_equals_36_l1279_127962


namespace triangle_area_l1279_127978

theorem triangle_area (a b c : ℝ) (h1 : a = 7) (h2 : b = 24) (h3 : c = 25) :
  (1/2) * a * b = 84 := by
  sorry

end triangle_area_l1279_127978


namespace max_prime_difference_l1279_127918

theorem max_prime_difference (a b c d : ℕ) : 
  a.Prime ∧ b.Prime ∧ c.Prime ∧ d.Prime ∧
  (a + b + c + 18 + d).Prime ∧ (a + b + c + 18 - d).Prime ∧
  (b + c).Prime ∧ (c + d).Prime ∧
  (a + b + c = 2010) ∧
  (a ≠ 3 ∧ b ≠ 3 ∧ c ≠ 3 ∧ d ≠ 3) ∧
  (d ≤ 50) →
  (∃ (p q : ℕ), (p.Prime ∧ q.Prime ∧ 
    (p = a ∨ p = b ∨ p = c ∨ p = d ∨ 
     p = a + b + c + 18 + d ∨ p = a + b + c + 18 - d ∨
     p = b + c ∨ p = c + d) ∧
    (q = a ∨ q = b ∨ q = c ∨ q = d ∨ 
     q = a + b + c + 18 + d ∨ q = a + b + c + 18 - d ∨
     q = b + c ∨ q = c + d) ∧
    p - q ≤ 2067) ∧
   ∀ (r s : ℕ), (r.Prime ∧ s.Prime ∧ 
    (r = a ∨ r = b ∨ r = c ∨ r = d ∨ 
     r = a + b + c + 18 + d ∨ r = a + b + c + 18 - d ∨
     r = b + c ∨ r = c + d) ∧
    (s = a ∨ s = b ∨ s = c ∨ s = d ∨ 
     s = a + b + c + 18 + d ∨ s = a + b + c + 18 - d ∨
     s = b + c ∨ s = c + d) →
    r - s ≤ 2067)) :=
by sorry

end max_prime_difference_l1279_127918


namespace tan_70_plus_tan_50_minus_sqrt3_tan_70_tan_50_l1279_127985

theorem tan_70_plus_tan_50_minus_sqrt3_tan_70_tan_50 :
  Real.tan (70 * π / 180) + Real.tan (50 * π / 180) - Real.sqrt 3 * Real.tan (70 * π / 180) * Real.tan (50 * π / 180) = -Real.sqrt 3 := by
  sorry

end tan_70_plus_tan_50_minus_sqrt3_tan_70_tan_50_l1279_127985


namespace equation_solution_l1279_127919

theorem equation_solution : 
  ∃! x : ℚ, (x - 15) / 3 = (3 * x + 11) / 8 :=
by
  use -153
  sorry

end equation_solution_l1279_127919


namespace book_arrangement_count_l1279_127982

def number_of_books : ℕ := 6
def number_of_identical_pairs : ℕ := 2
def books_per_pair : ℕ := 2

theorem book_arrangement_count :
  (number_of_books.factorial) / ((books_per_pair.factorial) ^ number_of_identical_pairs) = 180 := by
  sorry

end book_arrangement_count_l1279_127982


namespace bus_speed_and_interval_l1279_127922

/-- The speed of buses and interval between departures in a traffic scenario --/
theorem bus_speed_and_interval (a b c : ℝ) (hc : c > b) (hb : b > 0) (ha : a > 0) :
  ∃ (x t : ℝ),
    (a + x) * b = t * x ∧
    (x - a) * c = t * x ∧
    x = a * (c + b) / (c - b) ∧
    t = 2 * b * c / (b + c) := by
  sorry

end bus_speed_and_interval_l1279_127922


namespace inequality_equivalence_l1279_127993

theorem inequality_equivalence (x : ℝ) : 
  (∀ y : ℝ, y > 0 → (4 * (x * y^2 + x^2 * y + 4 * y^2 + 4 * x * y)) / (x + y) > 3 * x^2 * y) ↔ 
  (x > 0 ∧ x < 4) := by
sorry

end inequality_equivalence_l1279_127993


namespace vera_doll_count_l1279_127945

theorem vera_doll_count (aida sophie vera : ℕ) : 
  aida = 2 * sophie →
  sophie = 2 * vera →
  aida + sophie + vera = 140 →
  vera = 20 :=
by sorry

end vera_doll_count_l1279_127945


namespace cost_of_dozen_pens_l1279_127973

/-- Given the cost of 3 pens and 5 pencils, and the cost ratio of pen to pencil,
    prove the cost of one dozen pens -/
theorem cost_of_dozen_pens (cost_3pens_5pencils : ℕ) (ratio_pen_pencil : ℚ) :
  cost_3pens_5pencils = 240 →
  ratio_pen_pencil = 5 / 1 →
  (12 : ℕ) * (5 * (cost_3pens_5pencils / (3 * 5 + 5))) = 720 := by
sorry

end cost_of_dozen_pens_l1279_127973


namespace min_value_ratio_l1279_127914

theorem min_value_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∃ d₁ : ℝ, ∀ n, a (n + 1) = a n + d₁) →
  (∃ d₂ : ℝ, ∀ n, Real.sqrt (S (n + 1)) = Real.sqrt (S n) + d₂) →
  (∀ n, S n = (n * (a 1 + a n)) / 2) →
  (∀ n, (S (n + 10)) / (a n) ≥ 21) ∧
  (∃ n, (S (n + 10)) / (a n) = 21) :=
sorry

end min_value_ratio_l1279_127914


namespace fourth_root_of_four_sixes_l1279_127994

theorem fourth_root_of_four_sixes : 
  (4^6 + 4^6 + 4^6 + 4^6 : ℝ)^(1/4) = 8 * Real.sqrt 2 := by
  sorry

end fourth_root_of_four_sixes_l1279_127994


namespace max_y_coordinate_sin_3theta_l1279_127968

/-- The maximum y-coordinate of a point on the curve r = sin 3θ is 4√3/9 -/
theorem max_y_coordinate_sin_3theta :
  let r : ℝ → ℝ := λ θ => Real.sin (3 * θ)
  let y : ℝ → ℝ := λ θ => r θ * Real.sin θ
  ∃ (max_y : ℝ), (∀ θ, y θ ≤ max_y) ∧ (max_y = 4 * Real.sqrt 3 / 9) := by
  sorry

end max_y_coordinate_sin_3theta_l1279_127968


namespace triangle_side_length_l1279_127906

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a + b + c = 20 →
  (1/2) * b * c * Real.sin A = 10 * Real.sqrt 3 →
  A = π / 3 →
  a = 7 :=
sorry

end triangle_side_length_l1279_127906


namespace corner_cut_length_l1279_127959

/-- Given a rectangular sheet of dimensions 48 m x 36 m, if squares of side length x
    are cut from each corner to form an open box with volume 5120 m³,
    then x = 8 m. -/
theorem corner_cut_length (x : ℝ) : 
  x > 0 ∧ x < 18 ∧ (48 - 2*x) * (36 - 2*x) * x = 5120 → x = 8 := by
  sorry

end corner_cut_length_l1279_127959


namespace max_tan_alpha_l1279_127996

theorem max_tan_alpha (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2) 
  (h3 : Real.tan (α + β) = 9 * Real.tan β) : 
  ∃ (max_tan_α : Real), max_tan_α = 4/3 ∧ 
    ∀ (γ : Real), (0 < γ ∧ γ < π/2 ∧ ∃ (δ : Real), (0 < δ ∧ δ < π/2 ∧ Real.tan (γ + δ) = 9 * Real.tan δ)) 
      → Real.tan γ ≤ max_tan_α :=
sorry

end max_tan_alpha_l1279_127996


namespace radioactive_balls_identification_l1279_127997

/-- Represents a strategy for identifying radioactive balls -/
structure Strategy where
  num_tests : ℕ
  -- Other fields omitted for simplicity

/-- Represents the outcome of applying a strategy -/
inductive Outcome
  | IdentifiedBoth
  | NotIdentified

/-- Applies a strategy to a set of balls and returns the outcome -/
def apply_strategy (s : Strategy) (total_balls : ℕ) (radioactive_balls : ℕ) : Outcome :=
  sorry

theorem radioactive_balls_identification
  (total_balls : ℕ)
  (radioactive_balls : ℕ)
  (h_total : total_balls = 11)
  (h_radioactive : radioactive_balls = 2) :
  (∀ s : Strategy, s.num_tests < 7 → ∃ outcome, outcome = Outcome.NotIdentified) ∧
  (∃ s : Strategy, s.num_tests = 7 ∧ apply_strategy s total_balls radioactive_balls = Outcome.IdentifiedBoth) :=
sorry

end radioactive_balls_identification_l1279_127997


namespace factorization_proof_l1279_127974

theorem factorization_proof (x : ℝ) : x^2 - 4*x + 4 = (x - 2)^2 := by
  sorry

end factorization_proof_l1279_127974


namespace rectangle_segment_sum_l1279_127917

theorem rectangle_segment_sum (a b : ℝ) (n : ℕ) (h1 : a = 4) (h2 : b = 3) (h3 : n = 168) :
  let diagonal := Real.sqrt (a^2 + b^2)
  let segment_sum := n * diagonal
  segment_sum = 840 := by
  sorry

end rectangle_segment_sum_l1279_127917


namespace tangent_line_at_one_unique_solution_l1279_127987

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a - 1/x - Real.log x

-- Part 1: Tangent line when a = 2
theorem tangent_line_at_one (x : ℝ) :
  let a : ℝ := 2
  let f' : ℝ → ℝ := λ x => (1 - x) / (x^2)
  (f' 1 = 0) ∧ (f a 1 = 1) → (λ y => y = 1) = (λ y => y = f' 1 * (x - 1) + f a 1) :=
sorry

-- Part 2: Unique solution when a = 1
theorem unique_solution :
  (∃! x : ℝ, f 1 x = 0) ∧ (∀ a : ℝ, a ≠ 1 → ¬(∃! x : ℝ, f a x = 0)) :=
sorry

end

end tangent_line_at_one_unique_solution_l1279_127987


namespace stratified_sample_second_year_l1279_127967

/-- Represents the number of students in each year of high school -/
structure HighSchool where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ

/-- Calculates the total number of students in the high school -/
def total_students (hs : HighSchool) : ℕ :=
  hs.first_year + hs.second_year + hs.third_year

/-- Calculates the number of students from a specific year in a stratified sample -/
def stratified_sample_size (hs : HighSchool) (year_size : ℕ) (sample_size : ℕ) : ℕ :=
  (year_size * sample_size) / total_students hs

/-- Theorem: In a stratified sample of 100 students from a high school with 1000 first-year,
    800 second-year, and 700 third-year students, the number of second-year students
    in the sample is 32. -/
theorem stratified_sample_second_year :
  let hs : HighSchool := ⟨1000, 800, 700⟩
  stratified_sample_size hs hs.second_year 100 = 32 := by
  sorry

end stratified_sample_second_year_l1279_127967


namespace nine_sided_polygon_odd_spanning_diagonals_l1279_127984

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  -- We don't need to define the specifics of a regular polygon for this problem

/-- The number of diagonals in a regular polygon that span an odd number of vertices between their endpoints -/
def oddSpanningDiagonals (p : RegularPolygon n) : ℕ :=
  sorry  -- Definition to be implemented

/-- Theorem stating that a regular nine-sided polygon has 18 diagonals spanning an odd number of vertices -/
theorem nine_sided_polygon_odd_spanning_diagonals :
  ∀ (p : RegularPolygon 9), oddSpanningDiagonals p = 18 :=
by sorry

end nine_sided_polygon_odd_spanning_diagonals_l1279_127984


namespace cylinder_volume_relation_l1279_127912

/-- Given two cylinders A and B, where the radius of A equals the height of B,
    and the height of A equals the radius of B, if the volume of A is three times
    the volume of B, then the volume of A can be expressed as 9πh^3,
    where h is the height of A. -/
theorem cylinder_volume_relation (h r : ℝ) : 
  h > 0 → r > 0 → 
  (π * r^2 * h) = 3 * (π * h^2 * r) → 
  ∃ (N : ℝ), π * r^2 * h = N * π * h^3 ∧ N = 9 := by
  sorry

end cylinder_volume_relation_l1279_127912


namespace x_difference_is_22_l1279_127902

theorem x_difference_is_22 (x : ℝ) (h : (x + 3)^2 / (3*x + 65) = 2) :
  ∃ (x₁ x₂ : ℝ), ((x₁ + 3)^2 / (3*x₁ + 65) = 2) ∧
                 ((x₂ + 3)^2 / (3*x₂ + 65) = 2) ∧
                 (x₁ ≠ x₂) ∧
                 (x₁ - x₂ = 22 ∨ x₂ - x₁ = 22) :=
by sorry

end x_difference_is_22_l1279_127902


namespace point_transformation_l1279_127971

def rotate90CCW (x y : ℝ) : ℝ × ℝ := (-y, x)

def reflectAboutYeqX (x y : ℝ) : ℝ × ℝ := (y, x)

theorem point_transformation (a b : ℝ) :
  let (x₁, y₁) := rotate90CCW a b
  let (x₂, y₂) := reflectAboutYeqX x₁ y₁
  (x₂ = 3 ∧ y₂ = -7) → b - a = 4 := by sorry

end point_transformation_l1279_127971


namespace max_popsicles_with_budget_l1279_127955

def single_price : ℚ := 3/2
def box3_price : ℚ := 3
def box7_price : ℚ := 5
def budget : ℚ := 12

def max_popsicles (s p3 p7 : ℕ) : ℕ := s + 3 * p3 + 7 * p7

def valid_purchase (s p3 p7 : ℕ) : Prop :=
  single_price * s + box3_price * p3 + box7_price * p7 ≤ budget

theorem max_popsicles_with_budget :
  ∃ (s p3 p7 : ℕ), valid_purchase s p3 p7 ∧
    max_popsicles s p3 p7 = 15 ∧
    ∀ (s' p3' p7' : ℕ), valid_purchase s' p3' p7' →
      max_popsicles s' p3' p7' ≤ 15 := by sorry

end max_popsicles_with_budget_l1279_127955


namespace complex_number_imaginary_part_l1279_127990

theorem complex_number_imaginary_part (a : ℝ) :
  let z : ℂ := (1 + a * Complex.I) / (1 - Complex.I)
  Complex.im z = 1 → a = 1 := by
sorry

end complex_number_imaginary_part_l1279_127990


namespace deepak_age_l1279_127999

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    prove Deepak's current age -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 5 / 2 →
  rahul_age + 6 = 26 →
  deepak_age = 8 := by
sorry

end deepak_age_l1279_127999


namespace percent_relation_l1279_127977

theorem percent_relation (x y z : ℝ) 
  (h1 : x = y * 1.2)  -- x is 20 percent more than y
  (h2 : y = z * 0.7)  -- y is 30 percent less than z
  : x = z * 0.84 :=   -- x is 84 percent of z
by sorry

end percent_relation_l1279_127977


namespace prob_at_least_three_same_l1279_127952

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 4

/-- The probability of rolling a specific value on a single die -/
def probSingleDie : ℚ := 1 / numSides

/-- The probability that at least three out of four fair six-sided dice show the same value -/
def probAtLeastThreeSame : ℚ := 7 / 72

/-- Theorem stating that the probability of at least three out of four fair six-sided dice 
    showing the same value is 7/72 -/
theorem prob_at_least_three_same :
  probAtLeastThreeSame = 
    (1 * probSingleDie ^ 3) + -- Probability of all four dice showing same value
    (4 * probSingleDie ^ 2 * (1 - probSingleDie)) -- Probability of exactly three dice showing same value
  := by sorry

end prob_at_least_three_same_l1279_127952


namespace all_zeros_assignment_l1279_127954

/-- Represents a vertex in the triangular grid -/
structure Vertex (n : ℕ) where
  x : Fin (n + 1)
  y : Fin (n + 1)
  h : x.val + y.val ≤ n

/-- Represents an assignment of real numbers to vertices -/
def Assignment (n : ℕ) := Vertex n → ℝ

/-- Checks if three vertices form a triangle parallel to the sides of the main triangle -/
def is_parallel_triangle (n : ℕ) (v1 v2 v3 : Vertex n) : Prop :=
  ∃ (dx dy : Fin (n + 1)), 
    (v2.x = v1.x + dx ∧ v2.y = v1.y) ∧
    (v3.x = v1.x ∧ v3.y = v1.y + dy)

/-- The main theorem -/
theorem all_zeros_assignment {n : ℕ} (h : n ≥ 3) 
  (f : Assignment n) 
  (sum_zero : ∀ (v1 v2 v3 : Vertex n), 
    is_parallel_triangle n v1 v2 v3 → f v1 + f v2 + f v3 = 0) :
  ∀ v : Vertex n, f v = 0 := by sorry

end all_zeros_assignment_l1279_127954


namespace min_value_a2_b2_l1279_127900

/-- Given that (ax^2 + b/x)^6 has a coefficient of 20 for x^3, 
    the minimum value of a^2 + b^2 is 2 -/
theorem min_value_a2_b2 (a b : ℝ) : 
  (∃ c : ℝ, c = 20 ∧ 
   c = (Nat.choose 6 3 : ℝ) * a^3 * b^3) → 
  ∀ x y : ℝ, x^2 + y^2 ≥ 2 ∧ (x^2 + y^2 = 2 → x = 1 ∧ y = 1) :=
by sorry

end min_value_a2_b2_l1279_127900


namespace a_power_sum_l1279_127989

theorem a_power_sum (a x : ℝ) (ha : a > 0) (hx : a^(x/2) + a^(-x/2) = 5) : 
  a^x + a^(-x) = 23 := by
sorry

end a_power_sum_l1279_127989


namespace marts_income_percentage_l1279_127937

/-- Given that Tim's income is 60 percent less than Juan's income
    and Mart's income is 64 percent of Juan's income,
    prove that Mart's income is 60 percent more than Tim's income. -/
theorem marts_income_percentage (juan tim mart : ℝ) 
  (h1 : tim = juan - 0.60 * juan)
  (h2 : mart = 0.64 * juan) :
  mart = tim + 0.60 * tim := by
  sorry

end marts_income_percentage_l1279_127937


namespace sum_of_binary_digits_300_l1279_127965

/-- Given a natural number n, returns the sum of digits in its binary representation -/
def sumOfBinaryDigits (n : ℕ) : ℕ :=
  -- Implementation details omitted
  sorry

/-- Theorem: The sum of digits in the binary representation of 300 is 4 -/
theorem sum_of_binary_digits_300 : sumOfBinaryDigits 300 = 4 := by
  sorry

end sum_of_binary_digits_300_l1279_127965


namespace box_max_volume_l1279_127929

/-- The volume of the box as a function of the side length of the cut squares -/
def boxVolume (x : ℝ) : ℝ := (10 - 2*x) * (16 - 2*x) * x

/-- The maximum volume of the box -/
def maxVolume : ℝ := 144

theorem box_max_volume :
  ∃ (x : ℝ), 0 < x ∧ x < 5 ∧ 
  (∀ (y : ℝ), 0 < y ∧ y < 5 → boxVolume y ≤ boxVolume x) ∧
  boxVolume x = maxVolume :=
sorry

end box_max_volume_l1279_127929


namespace lottery_probabilities_l1279_127923

/-- Represents the total number of lottery tickets -/
def total_tickets : ℕ := 12

/-- Represents the number of winning tickets -/
def winning_tickets : ℕ := 2

/-- Represents the number of people -/
def num_people : ℕ := 4

/-- Represents the probability of giving 2 winning tickets to different people -/
def prob_different_people : ℚ := 9/11

/-- Represents the probability of giving 1 winning ticket to A and 1 to B -/
def prob_A_and_B : ℚ := 3/22

/-- Theorem stating the probabilities for the lottery ticket distribution -/
theorem lottery_probabilities :
  (prob_different_people = 9/11) ∧ (prob_A_and_B = 3/22) :=
sorry

end lottery_probabilities_l1279_127923


namespace line_bisected_by_M_l1279_127966

-- Define the lines and point
def l₁ (x y : ℝ) : Prop := x - 3 * y + 10 = 0
def l₂ (x y : ℝ) : Prop := 2 * x + y - 8 = 0
def M : ℝ × ℝ := (0, 1)

-- Define the line we want to prove
def target_line (x y : ℝ) : Prop := y = -1/3 * x + 1

-- Theorem statement
theorem line_bisected_by_M :
  ∃ (A B : ℝ × ℝ),
    l₁ A.1 A.2 ∧
    l₂ B.1 B.2 ∧
    target_line A.1 A.2 ∧
    target_line B.1 B.2 ∧
    target_line M.1 M.2 ∧
    M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) :=
  sorry


end line_bisected_by_M_l1279_127966


namespace tiktok_house_theorem_l1279_127932

/-- Represents a 3x3 grid of bloggers --/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Represents a day's arrangement of bloggers --/
def DailyArrangement := Fin 9 → Fin 3 × Fin 3

/-- Represents the three days of arrangements --/
def ThreeDayArrangements := Fin 3 → DailyArrangement

/-- Checks if two positions in the grid are adjacent --/
def are_adjacent (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p2.2 = p1.2 + 1)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p2.1 = p1.1 + 1))

/-- Counts the number of unique pairs formed over three days --/
def count_unique_pairs (arrangements : ThreeDayArrangements) : ℕ :=
  sorry

/-- The main theorem to be proved --/
theorem tiktok_house_theorem (arrangements : ThreeDayArrangements) :
  count_unique_pairs arrangements < (9 * 8) / 2 := by
  sorry

end tiktok_house_theorem_l1279_127932


namespace inverse_g_75_l1279_127940

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^3 - 6

-- State the theorem
theorem inverse_g_75 : g⁻¹ 75 = 3 := by sorry

end inverse_g_75_l1279_127940


namespace pumpkin_weight_problem_l1279_127961

/-- Given two pumpkins with a total weight of 12.7 pounds, 
    if one pumpkin weighs 4 pounds, then the other pumpkin weighs 8.7 pounds. -/
theorem pumpkin_weight_problem (total_weight : ℝ) (pumpkin1_weight : ℝ) (pumpkin2_weight : ℝ) :
  total_weight = 12.7 →
  pumpkin1_weight = 4 →
  total_weight = pumpkin1_weight + pumpkin2_weight →
  pumpkin2_weight = 8.7 := by
  sorry

end pumpkin_weight_problem_l1279_127961


namespace cosine_in_special_triangle_l1279_127950

/-- 
In a triangle ABC, given that:
1. The side lengths a, b, c form a geometric sequence
2. c = 2a
Then, cos B = 3/4
-/
theorem cosine_in_special_triangle (a b c : ℝ) (h_positive : a > 0) 
  (h_geometric : b^2 = a * c) (h_relation : c = 2 * a) :
  let cos_B := (a^2 + c^2 - b^2) / (2 * a * c)
  cos_B = 3/4 := by sorry

end cosine_in_special_triangle_l1279_127950


namespace closest_integer_to_k_l1279_127972

theorem closest_integer_to_k : ∃ (k : ℝ), 
  k = Real.sqrt 2 * ((Real.sqrt 5 + Real.sqrt 3) * (Real.sqrt 5 - Real.sqrt 3)) ∧
  ∀ (n : ℤ), |k - 3| ≤ |k - n| :=
by sorry

end closest_integer_to_k_l1279_127972


namespace profit_share_difference_l1279_127947

theorem profit_share_difference (a b c : ℕ) (b_profit : ℕ) : 
  a = 8000 → b = 10000 → c = 12000 → b_profit = 1400 →
  ∃ (a_profit c_profit : ℕ), 
    a_profit * b = b_profit * a ∧ 
    c_profit * b = b_profit * c ∧ 
    c_profit - a_profit = 560 := by
  sorry

end profit_share_difference_l1279_127947


namespace h_shape_perimeter_is_44_l1279_127905

/-- The perimeter of a rectangle with length l and width w -/
def rectanglePerimeter (l w : ℝ) : ℝ := 2 * l + 2 * w

/-- The perimeter of an H shape formed by two vertical rectangles and one horizontal rectangle -/
def hShapePerimeter (v_length v_width h_length h_width : ℝ) : ℝ :=
  2 * (rectanglePerimeter v_length v_width) + 
  (rectanglePerimeter h_length h_width) - 
  2 * (2 * h_width)

theorem h_shape_perimeter_is_44 : 
  hShapePerimeter 6 3 6 2 = 44 := by sorry

end h_shape_perimeter_is_44_l1279_127905


namespace product_of_roots_is_root_of_sextic_l1279_127938

theorem product_of_roots_is_root_of_sextic (a b c d : ℝ) : 
  a^4 + a^3 - 1 = 0 → 
  b^4 + b^3 - 1 = 0 → 
  c^4 + c^3 - 1 = 0 → 
  d^4 + d^3 - 1 = 0 → 
  (a * b)^6 + (a * b)^4 + (a * b)^3 - (a * b)^2 - 1 = 0 :=
by sorry

end product_of_roots_is_root_of_sextic_l1279_127938


namespace circle_properties_l1279_127970

/-- Represents a circle in the 2D plane -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- Calculates the center of a circle given its equation -/
def circle_center (c : Circle) : ℝ × ℝ := sorry

/-- Calculates the length of the shortest chord passing through a given point -/
def shortest_chord_length (c : Circle) (p : ℝ × ℝ) : ℝ := sorry

/-- The main theorem about the circle and its properties -/
theorem circle_properties :
  let c : Circle := { equation := fun x y => x^2 + y^2 - 6*x - 8*y = 0 }
  let p : ℝ × ℝ := (3, 5)
  circle_center c = (3, 4) ∧
  shortest_chord_length c p = 4 * Real.sqrt 6 := by sorry

end circle_properties_l1279_127970


namespace log_inequality_relation_l1279_127936

-- Define the logarithm function (base 10)
noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_inequality_relation :
  (∀ x y : ℝ, x > 0 → y > 0 → (log x < log y → x < y)) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x < y ∧ ¬(log x < log y)) :=
sorry

end log_inequality_relation_l1279_127936


namespace unique_solution_l1279_127941

/-- Given denominations 3, n, and n+2, returns true if m cents can be formed -/
def can_form_postage (n : ℕ) (m : ℕ) : Prop :=
  ∃ (a b c : ℕ), m = 3 * a + n * b + (n + 2) * c

/-- Returns true if n satisfies the problem conditions -/
def satisfies_conditions (n : ℕ) : Prop :=
  (∀ m > 63, can_form_postage n m) ∧
  ¬(can_form_postage n 63)

theorem unique_solution :
  ∃! n : ℕ, n > 0 ∧ satisfies_conditions n ∧ n = 30 :=
sorry

end unique_solution_l1279_127941


namespace triangle_side_length_l1279_127903

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < A ∧ A < pi ∧ 0 < B ∧ B < pi ∧ 0 < C ∧ C < pi ∧
  A + B + C = pi →
  -- Given conditions
  a = 2 →
  B = pi / 3 →
  b = Real.sqrt 7 →
  -- Conclusion
  c = 3 := by
  sorry

end triangle_side_length_l1279_127903
