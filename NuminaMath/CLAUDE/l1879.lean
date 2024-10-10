import Mathlib

namespace min_value_x_plus_y_l1879_187911

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 1) (hy : y > 2) (h : (x - 1) * (y - 2) = 4) :
  ∀ a b : ℝ, a > 1 → b > 2 → (a - 1) * (b - 2) = 4 → x + y ≤ a + b ∧ ∃ x y : ℝ, x > 1 ∧ y > 2 ∧ (x - 1) * (y - 2) = 4 ∧ x + y = 7 := by
  sorry

end min_value_x_plus_y_l1879_187911


namespace night_flying_hours_l1879_187908

theorem night_flying_hours (total_required : ℕ) (day_flying : ℕ) (cross_country : ℕ) (monthly_hours : ℕ) (months : ℕ) : 
  total_required = 1500 →
  day_flying = 50 →
  cross_country = 121 →
  monthly_hours = 220 →
  months = 6 →
  total_required - (day_flying + cross_country) - (monthly_hours * months) = 9 := by
  sorry

end night_flying_hours_l1879_187908


namespace grandmother_pill_duration_l1879_187963

/-- Calculates the duration in months for a given pill supply -/
def pillDuration (pillSupply : ℕ) (pillFraction : ℚ) (daysPerDose : ℕ) (daysPerMonth : ℕ) : ℚ :=
  (pillSupply : ℚ) * daysPerDose / pillFraction / daysPerMonth

theorem grandmother_pill_duration :
  pillDuration 60 (1/3) 3 30 = 18 := by
  sorry

end grandmother_pill_duration_l1879_187963


namespace range_of_function_l1879_187906

theorem range_of_function (x : ℝ) : 
  (1/2 : ℝ) ≤ ((14 * Real.cos (2 * x) + 28 * Real.sin x + 15) * Real.pi / 108) ∧ 
  ((14 * Real.cos (2 * x) + 28 * Real.sin x + 15) * Real.pi / 108) ≤ 1 := by
  sorry

end range_of_function_l1879_187906


namespace complex_polynomial_equality_l1879_187995

theorem complex_polynomial_equality (z : ℂ) (h : z = 2 - I) :
  z^6 - 3*z^5 + z^4 + 5*z^3 + 2 = (z^2 - 4*z + 5)*(z^4 + z^3) + 2 := by
  sorry

end complex_polynomial_equality_l1879_187995


namespace x_in_terms_of_y_l1879_187907

theorem x_in_terms_of_y (x y : ℝ) :
  (x + 1) / (x - 2) = (y^2 + 3*y - 2) / (y^2 + 3*y - 5) →
  x = (y^2 + 3*y - 1) / 7 :=
by sorry

end x_in_terms_of_y_l1879_187907


namespace readers_of_both_l1879_187992

def total_readers : ℕ := 400
def science_fiction_readers : ℕ := 250
def literary_works_readers : ℕ := 230

theorem readers_of_both : ℕ := by
  sorry

end readers_of_both_l1879_187992


namespace ratio_equality_l1879_187931

theorem ratio_equality (a b c x y z : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : a^2 + b^2 + c^2 = 49)
  (h2 : x^2 + y^2 + z^2 = 64)
  (h3 : a*x + b*y + c*z = 56) :
  (a + b + c) / (x + y + z) = 7/8 := by
sorry

end ratio_equality_l1879_187931


namespace max_value_of_f_on_S_l1879_187955

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the constraint set
def S : Set ℝ := {x : ℝ | x^4 + 36 ≤ 13*x^2}

-- Theorem statement
theorem max_value_of_f_on_S :
  ∃ (M : ℝ), M = 18 ∧ ∀ x ∈ S, f x ≤ M :=
sorry

end max_value_of_f_on_S_l1879_187955


namespace base_r_problem_l1879_187959

/-- Represents a number in base r -/
def BaseRNumber (r : ℕ) (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => d + r * acc) 0

/-- The problem statement -/
theorem base_r_problem (r : ℕ) : 
  (r > 1) →
  (BaseRNumber r [0, 0, 0, 1] = 1000) →
  (BaseRNumber r [0, 4, 4] = 440) →
  (BaseRNumber r [0, 4, 3] = 340) →
  (1000 - 440 = 340) →
  r = 8 := by
sorry

end base_r_problem_l1879_187959


namespace imaginary_part_of_z_l1879_187953

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 2) : 
  Complex.im z = -1 := by sorry

end imaginary_part_of_z_l1879_187953


namespace gcd_930_868_l1879_187979

theorem gcd_930_868 : Nat.gcd 930 868 = 62 := by
  sorry

end gcd_930_868_l1879_187979


namespace x_convergence_bound_l1879_187927

def x : ℕ → ℚ
  | 0 => 7
  | n + 1 => (x n ^ 2 + 7 * x n + 12) / (x n + 8)

theorem x_convergence_bound : 
  ∃ m : ℕ, 243 ≤ m ∧ m ≤ 728 ∧ 
    x m ≤ 6 + 1 / 2^18 ∧ 
    ∀ k < m, x k > 6 + 1 / 2^18 :=
sorry

end x_convergence_bound_l1879_187927


namespace mystical_village_population_l1879_187975

theorem mystical_village_population : ∃ (x y z : ℕ), 
  (x > 0) ∧ (y > 0) ∧ (z > 0) ∧
  (y^2 = x^2 + 200) ∧
  (z^2 = y^2 + 180) ∧
  (∃ (k : ℕ), x^2 = 5 * k) :=
by sorry

end mystical_village_population_l1879_187975


namespace tony_total_payment_l1879_187940

def lego_price : ℕ := 250
def sword_price : ℕ := 120
def dough_price : ℕ := 35

def lego_quantity : ℕ := 3
def sword_quantity : ℕ := 7
def dough_quantity : ℕ := 10

def total_cost : ℕ := lego_price * lego_quantity + sword_price * sword_quantity + dough_price * dough_quantity

theorem tony_total_payment : total_cost = 1940 := by
  sorry

end tony_total_payment_l1879_187940


namespace square_sum_equals_43_l1879_187916

theorem square_sum_equals_43 (x y : ℝ) 
  (h1 : y + 6 = (x - 3)^2) 
  (h2 : x + 6 = (y - 3)^2) 
  (h3 : x ≠ y) : 
  x^2 + y^2 = 43 := by
  sorry

end square_sum_equals_43_l1879_187916


namespace hyperbola_point_coordinate_l1879_187989

theorem hyperbola_point_coordinate :
  ∀ x : ℝ,
  (Real.sqrt ((x - 5)^2 + 4^2) - Real.sqrt ((x + 5)^2 + 4^2) = 6) →
  x = -3 * Real.sqrt 2 := by
sorry

end hyperbola_point_coordinate_l1879_187989


namespace melies_money_left_l1879_187943

/-- The amount of money Méliès has left after buying meat -/
def money_left (initial_money meat_quantity meat_price : ℝ) : ℝ :=
  initial_money - meat_quantity * meat_price

/-- Theorem: Méliès has $16 left after buying meat -/
theorem melies_money_left :
  let initial_money : ℝ := 180
  let meat_quantity : ℝ := 2
  let meat_price : ℝ := 82
  money_left initial_money meat_quantity meat_price = 16 := by
  sorry

end melies_money_left_l1879_187943


namespace intersection_complement_theorem_l1879_187952

def M : Set ℝ := {x | ∃ y, y = Real.log (1 - 2/x)}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 1)}

theorem intersection_complement_theorem : N ∩ (Set.univ \ M) = Set.Icc 1 2 := by
  sorry

end intersection_complement_theorem_l1879_187952


namespace polygonal_number_formula_l1879_187985

def N (n k : ℕ) : ℚ :=
  match k with
  | 3 => (n^2 + n) / 2
  | 4 => n^2
  | 5 => (3*n^2 - n) / 2
  | 6 => 2*n^2 - n
  | _ => 0

theorem polygonal_number_formula (n k : ℕ) (h : k ≥ 3) :
  N n k = ((k - 2 : ℚ) / 2) * n^2 + ((4 - k : ℚ) / 2) * n :=
by sorry

end polygonal_number_formula_l1879_187985


namespace ricardo_coin_difference_l1879_187932

/-- The total number of coins Ricardo has -/
def total_coins : ℕ := 2020

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The number of pennies Ricardo has -/
def num_pennies : ℕ → ℕ := λ p => p

/-- The number of nickels Ricardo has -/
def num_nickels : ℕ → ℕ := λ p => total_coins - p

/-- The total value of Ricardo's coins in cents -/
def total_value : ℕ → ℕ := λ p => 
  penny_value * num_pennies p + nickel_value * num_nickels p

/-- The constraint that Ricardo has at least one penny and one nickel -/
def valid_distribution : ℕ → Prop := λ p => 
  1 ≤ num_pennies p ∧ 1 ≤ num_nickels p

theorem ricardo_coin_difference : 
  ∃ (max_p min_p : ℕ), 
    valid_distribution max_p ∧ 
    valid_distribution min_p ∧ 
    (∀ p, valid_distribution p → total_value p ≤ total_value max_p) ∧
    (∀ p, valid_distribution p → total_value min_p ≤ total_value p) ∧
    total_value max_p - total_value min_p = 8072 := by
  sorry

end ricardo_coin_difference_l1879_187932


namespace sum_fifth_sixth_l1879_187918

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q
  sum_first_two : a 1 + a 2 = 3
  sum_third_fourth : a 3 + a 4 = 12

/-- The sum of the fifth and sixth terms equals 48 -/
theorem sum_fifth_sixth (seq : GeometricSequence) : seq.a 5 + seq.a 6 = 48 := by
  sorry

end sum_fifth_sixth_l1879_187918


namespace divisor_count_l1879_187933

theorem divisor_count (m n : ℕ+) (h_coprime : Nat.Coprime m n) 
  (h_divisors : (Nat.divisors (m^3 * n^5)).card = 209) : 
  (Nat.divisors (m^5 * n^3)).card = 217 := by
  sorry

end divisor_count_l1879_187933


namespace find_k_value_l1879_187961

/-- Given functions f and g, prove the value of k when f(3) - g(3) = 6 -/
theorem find_k_value (f g : ℝ → ℝ) (k : ℝ) : 
  (∀ x, f x = 7 * x^2 - 2 / x + 5) →
  (∀ x, g x = x^2 - k) →
  f 3 - g 3 = 6 →
  k = -157 / 3 := by
sorry

end find_k_value_l1879_187961


namespace decimal_addition_l1879_187937

theorem decimal_addition : (0.8 : ℝ) + 0.02 = 0.82 := by
  sorry

end decimal_addition_l1879_187937


namespace rent_is_5000_l1879_187951

/-- Calculates the monthly rent for John's computer business -/
def calculate_rent (component_cost : ℝ) (markup : ℝ) (computers_sold : ℕ) 
                   (extra_expenses : ℝ) (profit : ℝ) : ℝ :=
  let selling_price := component_cost * markup
  let total_revenue := selling_price * computers_sold
  let total_component_cost := component_cost * computers_sold
  total_revenue - total_component_cost - extra_expenses - profit

/-- Proves that the monthly rent is $5000 given the specified conditions -/
theorem rent_is_5000 : 
  calculate_rent 800 1.4 60 3000 11200 = 5000 := by
  sorry

end rent_is_5000_l1879_187951


namespace cheese_division_theorem_l1879_187998

/-- Represents the weight of cheese pieces -/
structure CheesePair :=
  (larger : ℕ)
  (smaller : ℕ)

/-- Divides cheese by taking a piece equal to the smaller portion from the larger -/
def divide_cheese (pair : CheesePair) : CheesePair :=
  CheesePair.mk pair.smaller (pair.larger - pair.smaller)

/-- The initial weight of the cheese -/
def initial_weight : ℕ := 850

/-- The final weight of each piece of cheese -/
def final_piece_weight : ℕ := 25

theorem cheese_division_theorem :
  let final_state := CheesePair.mk final_piece_weight final_piece_weight
  let third_division := divide_cheese (divide_cheese (divide_cheese (CheesePair.mk initial_weight 0)))
  third_division = final_state :=
sorry

end cheese_division_theorem_l1879_187998


namespace smallest_maximal_arrangement_l1879_187900

/-- Represents a chessboard -/
structure Chessboard :=
  (size : ℕ)
  (total_squares : ℕ := size * size)

/-- Represents a Γ piece -/
structure GammaPiece :=
  (squares_covered : ℕ := 3)

/-- Represents an arrangement of Γ pieces on a chessboard -/
structure Arrangement (board : Chessboard) :=
  (pieces : ℕ)
  (is_maximal : Bool)

/-- The theorem stating the smallest number of Γ pieces in a maximal arrangement -/
theorem smallest_maximal_arrangement (board : Chessboard) (piece : GammaPiece) :
  board.size = 8 →
  ∃ (arr : Arrangement board), 
    arr.pieces = 16 ∧ 
    arr.is_maximal = true ∧
    ∀ (arr' : Arrangement board), arr'.is_maximal = true → arr'.pieces ≥ 16 := by
  sorry

end smallest_maximal_arrangement_l1879_187900


namespace quadratic_root_difference_l1879_187999

theorem quadratic_root_difference (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (∀ x y : ℝ, x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 ∧ x ≠ y → |x - y| = 2) →
  p = 2 * Real.sqrt (q + 1) := by
sorry

end quadratic_root_difference_l1879_187999


namespace librarian_books_taken_l1879_187974

theorem librarian_books_taken (total_books : ℕ) (books_per_shelf : ℕ) (shelves_needed : ℕ) : 
  total_books - (books_per_shelf * shelves_needed) = 10 :=
by
  sorry

#check librarian_books_taken 46 4 9

end librarian_books_taken_l1879_187974


namespace boat_stream_speed_ratio_l1879_187926

theorem boat_stream_speed_ratio 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (distance : ℝ) 
  (h1 : boat_speed > stream_speed) 
  (h2 : stream_speed > 0) 
  (h3 : distance > 0) 
  (h4 : distance / (boat_speed - stream_speed) = 2 * (distance / (boat_speed + stream_speed))) :
  boat_speed / stream_speed = 3 := by
sorry

end boat_stream_speed_ratio_l1879_187926


namespace oldies_requests_l1879_187929

/-- Represents the number of song requests for each genre --/
structure SongRequests where
  total : ℕ
  electropop : ℕ
  dance : ℕ
  rock : ℕ
  oldies : ℕ
  dj_choice : ℕ
  rap : ℕ

/-- Theorem stating the number of oldies requests given the conditions --/
theorem oldies_requests (sr : SongRequests) : sr.oldies = 5 :=
  by
  have h1 : sr.total = 30 := by sorry
  have h2 : sr.electropop = sr.total / 2 := by sorry
  have h3 : sr.dance = sr.electropop / 3 := by sorry
  have h4 : sr.rock = 5 := by sorry
  have h5 : sr.dj_choice = sr.oldies / 2 := by sorry
  have h6 : sr.rap = 2 := by sorry
  have h7 : sr.total = sr.electropop + sr.rock + sr.oldies + sr.dj_choice + sr.rap := by sorry
  sorry

end oldies_requests_l1879_187929


namespace remainder_h_x_10_divided_by_h_x_l1879_187973

-- Define the polynomial h(x)
def h (x : ℝ) : ℝ := x^6 + x^5 + x^4 + x^3 + x^2 + x + 1

-- State the theorem
theorem remainder_h_x_10_divided_by_h_x : 
  ∃ (q : ℝ → ℝ), h (x^10) = q x * h x + 7 := by sorry

end remainder_h_x_10_divided_by_h_x_l1879_187973


namespace village_population_l1879_187958

theorem village_population (population_percentage : Real) (partial_population : Nat) :
  population_percentage = 80 →
  partial_population = 23040 →
  (partial_population : Real) / (population_percentage / 100) = 28800 :=
by sorry

end village_population_l1879_187958


namespace pencil_sharpener_time_l1879_187924

/-- Represents the time in minutes for which we're solving -/
def t : ℝ := 6

/-- Time (in seconds) for hand-crank sharpener to sharpen one pencil -/
def hand_crank_time : ℝ := 45

/-- Time (in seconds) for electric sharpener to sharpen one pencil -/
def electric_time : ℝ := 20

/-- The difference in number of pencils sharpened -/
def pencil_difference : ℕ := 10

theorem pencil_sharpener_time :
  (60 * t / electric_time) = (60 * t / hand_crank_time) + pencil_difference :=
sorry

end pencil_sharpener_time_l1879_187924


namespace expedition_ratio_l1879_187928

/-- Proves the ratio of weeks spent on the last expedition to the second expedition -/
theorem expedition_ratio : 
  ∀ (first second last : ℕ) (total : ℕ),
  first = 3 →
  second = first + 2 →
  total = 7 * (first + second + last) →
  total = 126 →
  last = 2 * second :=
by
  sorry

end expedition_ratio_l1879_187928


namespace visibility_theorem_l1879_187915

/-- Represents the position of a person at a given time -/
structure Position where
  x : ℝ
  y : ℝ

/-- Represents the movement of a person -/
def move (initial : Position) (speed : ℝ) (time : ℝ) : Position :=
  { x := initial.x + speed * time, y := initial.y }

/-- Represents a circular building -/
structure Building where
  center : Position
  radius : ℝ

/-- The time when two people can see each other after being blocked by a building -/
def visibility_time (jenny_initial : Position) (kenny_initial : Position) 
                    (jenny_speed : ℝ) (kenny_speed : ℝ) (building : Building) : ℚ :=
  240 / 5

theorem visibility_theorem (jenny_initial : Position) (kenny_initial : Position) 
                           (jenny_speed : ℝ) (kenny_speed : ℝ) (building : Building) :
  jenny_initial.y - kenny_initial.y = 300 ∧
  jenny_speed = 1 ∧ 
  kenny_speed = 4 ∧
  building.center.y = (jenny_initial.y + kenny_initial.y) / 2 ∧
  building.radius = 100 →
  visibility_time jenny_initial kenny_initial jenny_speed kenny_speed building = 240 / 5 := by
  sorry

end visibility_theorem_l1879_187915


namespace abc_remainder_mod_7_l1879_187986

theorem abc_remainder_mod_7 (a b c : ℕ) 
  (h_a : a < 7) (h_b : b < 7) (h_c : c < 7)
  (h1 : (a + 3*b + 2*c) % 7 = 3)
  (h2 : (2*a + b + 3*c) % 7 = 2)
  (h3 : (3*a + 2*b + c) % 7 = 1) :
  (a * b * c) % 7 = 4 := by
sorry

end abc_remainder_mod_7_l1879_187986


namespace triangle_angle_proof_l1879_187994

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and the condition c * sin(A) = √3 * a * cos(C), prove that C = π/3. -/
theorem triangle_angle_proof (a b c A B C : ℝ) 
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
  (h_condition : c * Real.sin A = Real.sqrt 3 * a * Real.cos C) : 
  C = π / 3 := by
  sorry

end triangle_angle_proof_l1879_187994


namespace sqrt_fraction_simplification_l1879_187901

theorem sqrt_fraction_simplification : 
  Real.sqrt (16 / 25 + 9 / 4) = 17 / 10 := by
  sorry

end sqrt_fraction_simplification_l1879_187901


namespace rounding_approximation_less_than_exact_l1879_187941

theorem rounding_approximation_less_than_exact (x y z : ℕ+) :
  (↑(Int.floor (x : ℝ)) / ↑(Int.ceil (y : ℝ)) : ℝ) - ↑(Int.ceil (z : ℝ)) < (x : ℝ) / (y : ℝ) - (z : ℝ) := by
  sorry

end rounding_approximation_less_than_exact_l1879_187941


namespace arithmetic_calculations_l1879_187944

theorem arithmetic_calculations :
  (15 + (-6) + 3 - (-4) = 16) ∧
  (8 - 2^3 / (4/9) * (-2/3)^2 = 0) := by
sorry

end arithmetic_calculations_l1879_187944


namespace largest_after_removal_l1879_187964

/-- Represents the original sequence of digits --/
def original_sequence : List Nat := sorry

/-- Represents the sequence after removing 100 digits --/
def removed_sequence : List Nat := sorry

/-- The number of digits to be removed --/
def digits_to_remove : Nat := 100

/-- Function to convert a list of digits to a natural number --/
def list_to_number (l : List Nat) : Nat := sorry

/-- Function to check if a list of digits is a valid removal from the original sequence --/
def is_valid_removal (l : List Nat) : Prop := sorry

/-- Theorem stating that the removed_sequence is the largest possible after removing 100 digits --/
theorem largest_after_removal :
  (list_to_number removed_sequence = list_to_number original_sequence - digits_to_remove) ∧
  is_valid_removal removed_sequence ∧
  ∀ (other_sequence : List Nat),
    is_valid_removal other_sequence →
    list_to_number other_sequence ≤ list_to_number removed_sequence :=
sorry

end largest_after_removal_l1879_187964


namespace unique_g_50_l1879_187938

/-- A function from ℕ to ℕ satisfying the given property -/
def special_function (g : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, 2 * g (a^2 + 2*b^2) = (g a)^2 + 3*(g b)^2

theorem unique_g_50 (g : ℕ → ℕ) (h : special_function g) : g 50 = 0 := by
  sorry

#check unique_g_50

end unique_g_50_l1879_187938


namespace kareem_has_largest_result_l1879_187903

def jose_result (x : ℕ) : ℕ := ((x - 1) * 2) + 2

def thuy_result (x : ℕ) : ℕ := ((x * 2) - 1) + 2

def kareem_result (x : ℕ) : ℕ := ((x - 1) + 2) * 2

theorem kareem_has_largest_result :
  let start := 10
  kareem_result start > jose_result start ∧ kareem_result start > thuy_result start :=
by sorry

end kareem_has_largest_result_l1879_187903


namespace sum_of_three_odd_implies_one_odd_l1879_187987

theorem sum_of_three_odd_implies_one_odd (a b c : ℤ) : 
  Odd (a + b + c) → Odd a ∨ Odd b ∨ Odd c := by
  sorry

end sum_of_three_odd_implies_one_odd_l1879_187987


namespace weight_distribution_l1879_187946

theorem weight_distribution :
  ∀ x y z : ℕ,
  x + y + z = 100 →
  x + 10 * y + 50 * z = 500 →
  x = 60 ∧ y = 39 ∧ z = 1 :=
by sorry

end weight_distribution_l1879_187946


namespace solution_is_negative_eight_l1879_187976

/-- An arithmetic sequence is defined by its first three terms -/
structure ArithmeticSequence :=
  (a₁ : ℚ)
  (a₂ : ℚ)
  (a₃ : ℚ)

/-- The common difference of an arithmetic sequence -/
def ArithmeticSequence.commonDifference (seq : ArithmeticSequence) : ℚ :=
  seq.a₂ - seq.a₁

/-- A sequence is arithmetic if the difference between the second and third terms
    is equal to the difference between the first and second terms -/
def ArithmeticSequence.isArithmetic (seq : ArithmeticSequence) : Prop :=
  seq.a₃ - seq.a₂ = seq.a₂ - seq.a₁

/-- The given sequence -/
def givenSequence (x : ℚ) : ArithmeticSequence :=
  { a₁ := 2
    a₂ := (2*x + 1) / 3
    a₃ := 2*x + 4 }

theorem solution_is_negative_eight :
  ∃ x : ℚ, (givenSequence x).isArithmetic ∧ x = -8 := by sorry

end solution_is_negative_eight_l1879_187976


namespace marble_distribution_l1879_187962

theorem marble_distribution (n : ℕ) : n = 720 → 
  (Finset.filter (fun x => x > 1 ∧ x < n) (Finset.range (n + 1))).card = 28 := by
  sorry

end marble_distribution_l1879_187962


namespace inequality_proof_l1879_187993

theorem inequality_proof (x : ℝ) (n : ℕ) (h1 : x > 0) (h2 : n > 0) :
  x + n^n / x^n ≥ n + 1 := by
sorry

end inequality_proof_l1879_187993


namespace min_value_t_l1879_187942

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    satisfying certain conditions, the minimum value of t is 4√2/3. -/
theorem min_value_t (a b c : ℝ) (A B C : ℝ) (S : ℝ) (t : ℝ) :
  a - b = c / 3 →
  3 * Real.sin B = 2 * Real.sin A →
  2 ≤ a * c + c^2 →
  a * c + c^2 ≤ 32 →
  S = (1 / 2) * a * b * Real.sin C →
  t = (S + 2 * Real.sqrt 2) / a →
  t ≥ 4 * Real.sqrt 2 / 3 ∧ 
  ∃ (a₀ b₀ c₀ : ℝ) (A₀ B₀ C₀ : ℝ) (S₀ : ℝ),
    a₀ - b₀ = c₀ / 3 ∧
    3 * Real.sin B₀ = 2 * Real.sin A₀ ∧
    2 ≤ a₀ * c₀ + c₀^2 ∧
    a₀ * c₀ + c₀^2 ≤ 32 ∧
    S₀ = (1 / 2) * a₀ * b₀ * Real.sin C₀ ∧
    (S₀ + 2 * Real.sqrt 2) / a₀ = 4 * Real.sqrt 2 / 3 := by
  sorry

end min_value_t_l1879_187942


namespace car_distance_theorem_l1879_187983

/-- Given a car traveling at a specific speed for a certain time, 
    calculate the distance covered. -/
theorem car_distance_theorem (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 160 → time = 5 → distance = speed * time → distance = 800 :=
by sorry

end car_distance_theorem_l1879_187983


namespace clouddale_rainfall_2005_l1879_187914

/-- Represents the rainfall data for Clouddale -/
structure ClouddaleRainfall where
  initialYear : Nat
  initialAvgMonthlyRainfall : Real
  yearlyIncrease : Real

/-- Calculates the average monthly rainfall for a given year -/
def avgMonthlyRainfall (data : ClouddaleRainfall) (year : Nat) : Real :=
  data.initialAvgMonthlyRainfall + (year - data.initialYear : Real) * data.yearlyIncrease

/-- Calculates the total yearly rainfall for a given year -/
def totalYearlyRainfall (data : ClouddaleRainfall) (year : Nat) : Real :=
  (avgMonthlyRainfall data year) * 12

/-- Theorem: The total rainfall in Clouddale in 2005 was 522 mm -/
theorem clouddale_rainfall_2005 (data : ClouddaleRainfall) 
    (h1 : data.initialYear = 2003)
    (h2 : data.initialAvgMonthlyRainfall = 37.5)
    (h3 : data.yearlyIncrease = 3) : 
    totalYearlyRainfall data 2005 = 522 := by
  sorry


end clouddale_rainfall_2005_l1879_187914


namespace x_n_prime_iff_n_eq_two_l1879_187971

/-- Definition of x_n as a number of the form 10101...1 with n ones -/
def x_n (n : ℕ) : ℕ := (10^(2*n) - 1) / 99

/-- Theorem stating that x_n is prime only when n = 2 -/
theorem x_n_prime_iff_n_eq_two :
  ∀ n : ℕ, Nat.Prime (x_n n) ↔ n = 2 :=
sorry

end x_n_prime_iff_n_eq_two_l1879_187971


namespace y_elimination_condition_l1879_187968

/-- Given a system of linear equations 6x + my = 3 and 2x - ny = -6,
    y is directly eliminated by subtracting the second equation from the first
    if and only if m + n = 0 -/
theorem y_elimination_condition (m n : ℝ) : 
  (∀ x y : ℝ, 6 * x + m * y = 3 ∧ 2 * x - n * y = -6) →
  (∃! x : ℝ, ∀ y : ℝ, 6 * x + m * y = 3 ∧ 2 * x - n * y = -6) ↔
  m + n = 0 := by
sorry

end y_elimination_condition_l1879_187968


namespace percentage_of_percentage_l1879_187984

theorem percentage_of_percentage (y : ℝ) (h : y ≠ 0) :
  (0.3 * 0.6 * y) / y * 100 = 18 := by
  sorry

end percentage_of_percentage_l1879_187984


namespace arithmetic_sequence_max_a5_l1879_187977

theorem arithmetic_sequence_max_a5 (a : ℕ → ℝ) (s : ℕ → ℝ) :
  (∀ n, s n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2) →
  (∀ n, a n = a 1 + (n - 1) * (a 2 - a 1)) →
  s 2 ≥ 4 →
  s 4 ≤ 16 →
  a 5 ≤ 9 ∧ ∃ (a' : ℕ → ℝ), (∀ n, s n = (n * (2 * a' 1 + (n - 1) * (a' 2 - a' 1))) / 2) ∧
                             (∀ n, a' n = a' 1 + (n - 1) * (a' 2 - a' 1)) ∧
                             s 2 ≥ 4 ∧
                             s 4 ≤ 16 ∧
                             a' 5 = 9 :=
by sorry

end arithmetic_sequence_max_a5_l1879_187977


namespace triangle_properties_l1879_187965

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  (1/2 : ℝ) * Real.cos (2 * A) = Real.cos A ^ 2 - Real.cos A →
  a = 3 →
  Real.sin B = 2 * Real.sin C →
  A = π / 3 ∧ 
  (1/2 : ℝ) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 :=
by sorry

end triangle_properties_l1879_187965


namespace expand_expression_l1879_187902

theorem expand_expression (x : ℝ) : (11 * x + 5) * 3 * x^3 = 33 * x^4 + 15 * x^3 := by
  sorry

end expand_expression_l1879_187902


namespace not_square_sum_ceil_l1879_187904

theorem not_square_sum_ceil (a b : ℕ+) : ¬∃ k : ℤ, (a : ℤ)^2 + ⌈(4 * (a : ℤ)^2) / (b : ℤ)⌉ = k^2 := by
  sorry

end not_square_sum_ceil_l1879_187904


namespace stockholm_uppsala_distance_l1879_187978

/-- The distance between two cities on a map, in centimeters. -/
def map_distance : ℝ := 45

/-- The scale of the map, representing how many kilometers in reality correspond to one centimeter on the map. -/
def map_scale : ℝ := 20

/-- The actual distance between the two cities, in kilometers. -/
def actual_distance : ℝ := map_distance * map_scale

theorem stockholm_uppsala_distance : actual_distance = 900 := by
  sorry

end stockholm_uppsala_distance_l1879_187978


namespace ice_cream_scoops_l1879_187980

theorem ice_cream_scoops (oli_scoops : ℕ) (victoria_scoops : ℕ) : 
  oli_scoops = 4 → 
  victoria_scoops = 2 * oli_scoops → 
  victoria_scoops - oli_scoops = 8 :=
by
  sorry

end ice_cream_scoops_l1879_187980


namespace derivative_at_two_l1879_187956

/-- Given a function f(x) = ax³ + bx² + 3 where b = f'(2), prove that if f'(1) = -5, then f'(2) = -5 -/
theorem derivative_at_two (a b : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = a * x^3 + b * x^2 + 3)
  (h2 : b = (deriv f) 2) (h3 : (deriv f) 1 = -5) : (deriv f) 2 = -5 := by
  sorry

end derivative_at_two_l1879_187956


namespace bankers_interest_rate_l1879_187990

/-- Proves that given a time period of 3 years, a banker's gain of 270,
    and a banker's discount of 1020, the rate of interest per annum is 12%. -/
theorem bankers_interest_rate 
  (time : ℕ) (bankers_gain : ℚ) (bankers_discount : ℚ) :
  time = 3 → 
  bankers_gain = 270 → 
  bankers_discount = 1020 → 
  ∃ (rate : ℚ), rate = 12 ∧ 
    bankers_gain = bankers_discount - (bankers_discount / (1 + rate / 100 * time)) :=
by sorry

end bankers_interest_rate_l1879_187990


namespace roots_sum_zero_l1879_187909

theorem roots_sum_zero (a b c : ℂ) : 
  a^3 - 2*a^2 + 3*a - 4 = 0 →
  b^3 - 2*b^2 + 3*b - 4 = 0 →
  c^3 - 2*c^2 + 3*c - 4 = 0 →
  a ≠ b → b ≠ c → a ≠ c →
  1 / (a * (b^2 + c^2 - a^2)) + 1 / (b * (c^2 + a^2 - b^2)) + 1 / (c * (a^2 + b^2 - c^2)) = 0 :=
by sorry

end roots_sum_zero_l1879_187909


namespace expression_simplification_l1879_187948

theorem expression_simplification : 
  let x : ℝ := Real.sqrt 6 - Real.sqrt 2
  (x * (Real.sqrt 6 - x) + (x + Real.sqrt 5) * (x - Real.sqrt 5)) = 1 - 2 * Real.sqrt 3 := by
  sorry

end expression_simplification_l1879_187948


namespace least_number_of_cans_l1879_187950

def maaza_volume : ℕ := 60
def pepsi_volume : ℕ := 144
def sprite_volume : ℕ := 368

def can_volume : ℕ := Nat.gcd maaza_volume (Nat.gcd pepsi_volume sprite_volume)

def maaza_cans : ℕ := maaza_volume / can_volume
def pepsi_cans : ℕ := pepsi_volume / can_volume
def sprite_cans : ℕ := sprite_volume / can_volume

def total_cans : ℕ := maaza_cans + pepsi_cans + sprite_cans

theorem least_number_of_cans : total_cans = 143 := by
  sorry

end least_number_of_cans_l1879_187950


namespace min_colors_for_subdivided_rectangle_l1879_187970

/-- Represents an infinitely subdivided rectangle according to the given pattern. -/
structure InfinitelySubdividedRectangle where
  -- Add necessary fields here
  -- This is left abstract as the exact representation is not crucial for the theorem

/-- The minimum number of colors needed so that no two rectangles sharing an edge have the same color. -/
def minEdgeColors (r : InfinitelySubdividedRectangle) : ℕ := 3

/-- The minimum number of colors needed so that no two rectangles sharing a corner have the same color. -/
def minCornerColors (r : InfinitelySubdividedRectangle) : ℕ := 4

/-- Theorem stating the minimum number of colors needed for edge and corner coloring. -/
theorem min_colors_for_subdivided_rectangle (r : InfinitelySubdividedRectangle) :
  (minEdgeColors r, minCornerColors r) = (3, 4) := by sorry

end min_colors_for_subdivided_rectangle_l1879_187970


namespace slope_of_line_l1879_187945

theorem slope_of_line (x y : ℝ) :
  4 * x - 7 * y = 28 → (y - (-4)) / (x - 0) = 4 / 7 :=
by sorry

end slope_of_line_l1879_187945


namespace exponent_operations_l1879_187912

theorem exponent_operations (x : ℝ) (x_nonzero : x ≠ 0) :
  (x^2 * x^3 = x^5) ∧
  (x^2 + x^3 ≠ x^5) ∧
  ((x^3)^2 ≠ x^5) ∧
  (x^15 / x^3 ≠ x^5) :=
by sorry

end exponent_operations_l1879_187912


namespace zero_intersection_area_l1879_187921

-- Define the square pyramid
structure SquarePyramid where
  base_side : ℝ
  slant_edge : ℝ

-- Define the plane passing through midpoints
structure IntersectingPlane where
  pyramid : SquarePyramid

-- Define the intersection area
def intersection_area (plane : IntersectingPlane) : ℝ := sorry

-- Theorem statement
theorem zero_intersection_area 
  (pyramid : SquarePyramid) 
  (h1 : pyramid.base_side = 6) 
  (h2 : pyramid.slant_edge = 5) :
  intersection_area { pyramid := pyramid } = 0 := by sorry

end zero_intersection_area_l1879_187921


namespace product_of_fractions_l1879_187997

theorem product_of_fractions : (3 : ℚ) / 8 * 2 / 5 * 1 / 4 = 3 / 80 := by
  sorry

end product_of_fractions_l1879_187997


namespace square_area_equal_perimeter_triangle_l1879_187981

/-- The area of a square with perimeter equal to that of a triangle with sides 7.3 cm, 8.6 cm, and 10.1 cm is 42.25 square centimeters. -/
theorem square_area_equal_perimeter_triangle (a b c : ℝ) (s : ℝ) :
  a = 7.3 ∧ b = 8.6 ∧ c = 10.1 →
  4 * s = a + b + c →
  s^2 = 42.25 := by
  sorry

end square_area_equal_perimeter_triangle_l1879_187981


namespace eventually_divisible_by_large_power_of_two_l1879_187991

/-- Represents the state of the board at any given minute -/
structure BoardState where
  numbers : Finset ℕ
  odd_count : ℕ
  minute : ℕ

/-- The initial state of the board -/
def initial_board : BoardState :=
  { numbers := Finset.empty,  -- We don't know the specific numbers, so we use an empty set
    odd_count := 33,
    minute := 0 }

/-- The next state of the board after one minute -/
def next_board_state (state : BoardState) : BoardState :=
  { numbers := state.numbers,  -- We don't update the specific numbers
    odd_count := state.odd_count,
    minute := state.minute + 1 }

/-- Predicate to check if a number is divisible by 2^10000000 -/
def is_divisible_by_large_power_of_two (n : ℕ) : Prop :=
  ∃ k, n = k * (2^10000000)

/-- The main theorem to prove -/
theorem eventually_divisible_by_large_power_of_two :
  ∃ (n : ℕ) (state : BoardState), 
    state.minute = n ∧ 
    ∃ (m : ℕ), m ∈ state.numbers ∧ is_divisible_by_large_power_of_two m :=
  sorry

end eventually_divisible_by_large_power_of_two_l1879_187991


namespace number_of_trucks_l1879_187922

theorem number_of_trucks (total_packages : ℕ) (packages_per_truck : ℕ) (h1 : total_packages = 490) (h2 : packages_per_truck = 70) :
  total_packages / packages_per_truck = 7 := by
  sorry

end number_of_trucks_l1879_187922


namespace complex_fraction_power_four_l1879_187947

theorem complex_fraction_power_four (i : ℂ) (h : i * i = -1) : 
  ((1 + i) / (1 - i)) ^ 4 = 1 := by sorry

end complex_fraction_power_four_l1879_187947


namespace count_nonadjacent_permutations_l1879_187972

/-- The number of permutations of n distinct elements where two specific elements are not adjacent -/
def nonadjacent_permutations (n : ℕ) : ℕ :=
  (n - 2) * Nat.factorial (n - 1)

/-- Theorem stating that the number of permutations of n distinct elements 
    where two specific elements are not adjacent is (n-2)(n-1)! -/
theorem count_nonadjacent_permutations (n : ℕ) (h : n ≥ 2) :
  nonadjacent_permutations n = (n - 2) * Nat.factorial (n - 1) := by
  sorry

#check count_nonadjacent_permutations

end count_nonadjacent_permutations_l1879_187972


namespace power_division_rule_l1879_187996

theorem power_division_rule (m : ℝ) (h : m ≠ 0) : m^7 / m = m^6 := by
  sorry

end power_division_rule_l1879_187996


namespace pencil_price_l1879_187910

theorem pencil_price (x y : ℚ) 
  (eq1 : 3 * x + 5 * y = 345)
  (eq2 : 4 * x + 2 * y = 280) :
  y = 540 / 14 := by
  sorry

end pencil_price_l1879_187910


namespace min_filtration_cycles_l1879_187920

theorem min_filtration_cycles (initial_conc : ℝ) (reduction_rate : ℝ) (target_conc : ℝ) : 
  initial_conc = 225 →
  reduction_rate = 1/3 →
  target_conc = 7.5 →
  (∃ n : ℕ, (initial_conc * (1 - reduction_rate)^n ≤ target_conc ∧ 
             ∀ m : ℕ, m < n → initial_conc * (1 - reduction_rate)^m > target_conc)) →
  (∃ n : ℕ, n = 9 ∧ initial_conc * (1 - reduction_rate)^n ≤ target_conc ∧ 
             ∀ m : ℕ, m < n → initial_conc * (1 - reduction_rate)^m > target_conc) :=
by sorry

end min_filtration_cycles_l1879_187920


namespace reciprocal_minus_opposite_l1879_187988

theorem reciprocal_minus_opposite : 
  let x : ℚ := -4
  (-1 / x) - (-x) = -17 / 4 := by sorry

end reciprocal_minus_opposite_l1879_187988


namespace max_value_sum_fractions_l1879_187939

theorem max_value_sum_fractions (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x / (2 * x + y)) + (y / (x + 2 * y)) ≤ 2 / 3 ∧
  ((x / (2 * x + y)) + (y / (x + 2 * y)) = 2 / 3 ↔ x = y) :=
by sorry

end max_value_sum_fractions_l1879_187939


namespace triangle_side_length_l1879_187917

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π ∧
  a = 3 ∧ A = π/6 ∧ B = π/12 →
  c = 3 * Real.sqrt 2 :=
by sorry

end triangle_side_length_l1879_187917


namespace sqrt_sixteen_times_sqrt_sixteen_equals_eight_l1879_187925

theorem sqrt_sixteen_times_sqrt_sixteen_equals_eight : Real.sqrt (16 * Real.sqrt 16) = 2^3 := by
  sorry

end sqrt_sixteen_times_sqrt_sixteen_equals_eight_l1879_187925


namespace sqrt_x_minus_one_real_l1879_187934

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by
  sorry

end sqrt_x_minus_one_real_l1879_187934


namespace regular_polygon_interior_angle_l1879_187905

theorem regular_polygon_interior_angle (n : ℕ) (h : n - 3 = 5) :
  (180 * (n - 2) : ℝ) / n = 135 := by
  sorry

end regular_polygon_interior_angle_l1879_187905


namespace average_of_four_numbers_l1879_187923

theorem average_of_four_numbers (n : ℝ) :
  (3 + 16 + 33 + (n + 1)) / 4 = 20 → n = 27 := by
  sorry

end average_of_four_numbers_l1879_187923


namespace eight_stairs_climb_ways_l1879_187966

/-- The number of ways to climb n stairs, taking 1, 2, 3, or 4 steps at a time. -/
def climbWays (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | 3 => 4
  | m + 4 => climbWays m + climbWays (m + 1) + climbWays (m + 2) + climbWays (m + 3)

theorem eight_stairs_climb_ways :
  climbWays 8 = 108 := by
  sorry

#eval climbWays 8

end eight_stairs_climb_ways_l1879_187966


namespace car_distance_proof_l1879_187967

/-- Proves the initial distance between two cars driving towards each other --/
theorem car_distance_proof (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) 
  (h1 : speed1 = 100)
  (h2 : speed1 = 1.25 * speed2)
  (h3 : time = 4) :
  speed1 * time + speed2 * time = 720 := by
  sorry

end car_distance_proof_l1879_187967


namespace units_digit_of_7_to_50_l1879_187954

theorem units_digit_of_7_to_50 : 7^50 ≡ 9 [ZMOD 10] := by
  sorry

end units_digit_of_7_to_50_l1879_187954


namespace total_shaded_area_is_107_l1879_187960

/-- Represents a rectangle in a plane -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a triangle in a plane -/
structure Triangle where
  base : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ :=
  r.width * r.height

/-- Calculates the area of a triangle -/
def triangleArea (t : Triangle) : ℝ :=
  0.5 * t.base * t.height

/-- Represents the configuration of shapes in the plane -/
structure ShapeConfiguration where
  rect1 : Rectangle
  rect2 : Rectangle
  triangle : Triangle
  rect1TriangleOverlap : ℝ
  rect2TriangleOverlap : ℝ
  rectOverlap : ℝ

/-- Calculates the total shaded area given a ShapeConfiguration -/
def totalShadedArea (config : ShapeConfiguration) : ℝ :=
  rectangleArea config.rect1 + rectangleArea config.rect2 + triangleArea config.triangle -
  config.rectOverlap - config.rect1TriangleOverlap - config.rect2TriangleOverlap

/-- The theorem stating the total shaded area for the given configuration -/
theorem total_shaded_area_is_107 (config : ShapeConfiguration) :
  config.rect1 = ⟨5, 12⟩ →
  config.rect2 = ⟨4, 15⟩ →
  config.triangle = ⟨3, 4⟩ →
  config.rect1TriangleOverlap = 2 →
  config.rect2TriangleOverlap = 1 →
  config.rectOverlap = 16 →
  totalShadedArea config = 107 := by
  sorry

end total_shaded_area_is_107_l1879_187960


namespace celsius_to_fahrenheit_l1879_187936

/-- Given the relationship between Celsius (C) and Fahrenheit (F) temperatures,
    prove that when C is 25, F is 75. -/
theorem celsius_to_fahrenheit (C F : ℚ) : 
  C = 25 → C = (5 / 9) * (F - 30) → F = 75 := by sorry

end celsius_to_fahrenheit_l1879_187936


namespace percentage_of_x_l1879_187969

theorem percentage_of_x (x : ℝ) (p : ℝ) : 
  p * x = 0.3 * (0.7 * x) + 10 ↔ p = 0.21 + 10 / x :=
by sorry

end percentage_of_x_l1879_187969


namespace man_son_age_ratio_l1879_187982

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1,
    given the man is 24 years older than his son and the son's present age is 22 years. -/
theorem man_son_age_ratio :
  ∀ (son_age man_age : ℕ),
    son_age = 22 →
    man_age = son_age + 24 →
    (man_age + 2) / (son_age + 2) = 2 := by
  sorry

end man_son_age_ratio_l1879_187982


namespace fraction_of_odd_products_in_table_l1879_187913

-- Define the size of the multiplication table
def table_size : Nat := 16

-- Define a function to check if a number is odd
def is_odd (n : Nat) : Bool := n % 2 = 1

-- Define a function to count odd numbers in a range
def count_odd (n : Nat) : Nat :=
  (List.range n).filter is_odd |>.length

-- Statement of the theorem
theorem fraction_of_odd_products_in_table :
  (count_odd table_size ^ 2 : Rat) / (table_size ^ 2) = 1 / 4 := by
  sorry


end fraction_of_odd_products_in_table_l1879_187913


namespace first_part_speed_l1879_187957

/-- Proves that given a 50 km trip with two equal parts, where the second part is traveled at 33 km/h, 
    and the average speed of the entire trip is 44.00000000000001 km/h, 
    the speed of the first part of the trip is 66 km/h. -/
theorem first_part_speed (total_distance : ℝ) (first_part_distance : ℝ) (second_part_speed : ℝ) (average_speed : ℝ) :
  total_distance = 50 →
  first_part_distance = 25 →
  second_part_speed = 33 →
  average_speed = 44.00000000000001 →
  (total_distance / (first_part_distance / (total_distance - first_part_distance) * second_part_speed + first_part_distance / second_part_speed)) = average_speed →
  (total_distance - first_part_distance) / second_part_speed + first_part_distance / ((total_distance - first_part_distance) * second_part_speed / first_part_distance) = total_distance / average_speed →
  (total_distance - first_part_distance) * second_part_speed / first_part_distance = 66 :=
by sorry

end first_part_speed_l1879_187957


namespace circle_area_greater_than_rectangle_l1879_187935

theorem circle_area_greater_than_rectangle : ∀ (r : ℝ), r = 1 →
  π * r^2 ≥ 1 * 2.4 := by
  sorry

end circle_area_greater_than_rectangle_l1879_187935


namespace sally_total_spent_l1879_187919

/-- The total amount Sally spent on peaches and cherries -/
def total_spent (peach_price_after_coupon : ℚ) (cherry_price : ℚ) : ℚ :=
  peach_price_after_coupon + cherry_price

/-- Theorem stating that Sally spent $23.86 in total -/
theorem sally_total_spent : 
  total_spent 12.32 11.54 = 23.86 := by
  sorry

end sally_total_spent_l1879_187919


namespace quadratic_from_means_l1879_187930

theorem quadratic_from_means (a b : ℝ) (h_am : (a + b) / 2 = 7.5) (h_gm : Real.sqrt (a * b) = 12) :
  ∀ x, x ^ 2 - 15 * x + 144 = 0 ↔ (x = a ∨ x = b) :=
by sorry

end quadratic_from_means_l1879_187930


namespace place_values_of_fours_l1879_187949

def number : ℕ := 40649003

theorem place_values_of_fours (n : ℕ) (h : n = number) :
  (n / 10000000 % 10 = 4 ∧ n / 10000000 * 10000000 = 40000000) ∧
  (n / 10000 % 10 = 4 ∧ n / 10000 % 10000 * 10000 = 40000) :=
sorry

end place_values_of_fours_l1879_187949
