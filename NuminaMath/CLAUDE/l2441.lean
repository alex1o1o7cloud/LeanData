import Mathlib

namespace complex_fraction_simplification_l2441_244155

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (1 - 2*i) / (1 + 2*i) = -3/5 - 4/5*i :=
by
  sorry

end complex_fraction_simplification_l2441_244155


namespace sqrt_two_minus_two_cos_four_equals_two_sin_two_l2441_244132

theorem sqrt_two_minus_two_cos_four_equals_two_sin_two :
  Real.sqrt (2 - 2 * Real.cos 4) = 2 * Real.sin 2 := by
  sorry

end sqrt_two_minus_two_cos_four_equals_two_sin_two_l2441_244132


namespace imaginary_part_of_reciprocal_l2441_244135

/-- Given a complex number z = a^2 - 1 + (a+1)i where a ∈ ℝ and z is a pure imaginary number,
    the imaginary part of 1/(z+a) is -2/5 -/
theorem imaginary_part_of_reciprocal (a : ℝ) (z : ℂ) : 
  z = a^2 - 1 + (a + 1) * I → 
  z.re = 0 → 
  z.im ≠ 0 → 
  Complex.im (1 / (z + a)) = -2/5 := by sorry

end imaginary_part_of_reciprocal_l2441_244135


namespace completing_square_transform_l2441_244158

theorem completing_square_transform (x : ℝ) : 
  (x^2 - 2*x - 7 = 0) ↔ ((x - 1)^2 = 8) := by
  sorry

end completing_square_transform_l2441_244158


namespace game_points_calculation_l2441_244119

/-- Calculates the total points scored in a game given points per round and number of rounds played. -/
def totalPoints (pointsPerRound : ℕ) (numRounds : ℕ) : ℕ :=
  pointsPerRound * numRounds

/-- Theorem stating that for a game with 146 points per round and 157 rounds, the total points scored is 22822. -/
theorem game_points_calculation :
  totalPoints 146 157 = 22822 := by
  sorry

end game_points_calculation_l2441_244119


namespace remainder_76_pow_77_div_7_l2441_244150

theorem remainder_76_pow_77_div_7 : (76^77) % 7 = 6 := by
  sorry

end remainder_76_pow_77_div_7_l2441_244150


namespace book_club_monthly_books_l2441_244144

def prove_book_club_monthly_books (initial_books final_books bookstore_purchase yard_sale_purchase 
  daughter_gift mother_gift donated_books sold_books : ℕ) : Prop :=
  let total_acquired := bookstore_purchase + yard_sale_purchase + daughter_gift + mother_gift
  let total_removed := donated_books + sold_books
  let net_change := final_books - initial_books
  let book_club_total := net_change + total_removed - total_acquired
  (book_club_total % 12 = 0) ∧ (book_club_total / 12 = 1)

theorem book_club_monthly_books :
  prove_book_club_monthly_books 72 81 5 2 1 4 12 3 :=
sorry

end book_club_monthly_books_l2441_244144


namespace right_triangle_arctan_sum_l2441_244166

/-- 
In a right triangle ABC with right angle at B, 
the sum of arctan(b/(a+c)) and arctan(c/(a+b)) equals π/4, 
where a, b, and c are the lengths of the sides opposite to angles A, B, and C respectively.
-/
theorem right_triangle_arctan_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (right_angle : a^2 = b^2 + c^2) : 
  Real.arctan (b / (a + c)) + Real.arctan (c / (a + b)) = π / 4 := by
  sorry

end right_triangle_arctan_sum_l2441_244166


namespace intersection_point_is_solution_l2441_244185

theorem intersection_point_is_solution (k : ℝ) (hk : k ≠ 0) :
  (∃ (x y : ℝ), y = 2 * x - 1 ∧ y = k * x ∧ x = 1 ∧ y = 1) →
  (∃! (x y : ℝ), 2 * x - y = 1 ∧ k * x - y = 0 ∧ x = 1 ∧ y = 1) := by
  sorry

end intersection_point_is_solution_l2441_244185


namespace lcm_of_40_90_150_l2441_244153

theorem lcm_of_40_90_150 : Nat.lcm 40 (Nat.lcm 90 150) = 1800 := by sorry

end lcm_of_40_90_150_l2441_244153


namespace stratified_sample_correct_l2441_244178

/-- Represents the three age groups in the population -/
inductive AgeGroup
  | Elderly
  | MiddleAged
  | Young

/-- Represents the population sizes for each age group -/
def populationSize (group : AgeGroup) : Nat :=
  match group with
  | .Elderly => 25
  | .MiddleAged => 35
  | .Young => 40

/-- The total population size -/
def totalPopulation : Nat := populationSize .Elderly + populationSize .MiddleAged + populationSize .Young

/-- The desired sample size -/
def sampleSize : Nat := 40

/-- Calculates the stratified sample size for a given age group -/
def stratifiedSampleSize (group : AgeGroup) : Nat :=
  (populationSize group * sampleSize) / totalPopulation

/-- Theorem stating that the stratified sample sizes are correct -/
theorem stratified_sample_correct :
  stratifiedSampleSize .Elderly = 10 ∧
  stratifiedSampleSize .MiddleAged = 14 ∧
  stratifiedSampleSize .Young = 16 ∧
  stratifiedSampleSize .Elderly + stratifiedSampleSize .MiddleAged + stratifiedSampleSize .Young = sampleSize :=
by sorry

end stratified_sample_correct_l2441_244178


namespace twelve_gon_consecutive_sides_sum_l2441_244194

theorem twelve_gon_consecutive_sides_sum (sides : Fin 12 → ℕ) 
  (h1 : ∀ i : Fin 12, sides i = i.val + 1) : 
  ∃ i : Fin 12, sides i + sides (i + 1) + sides (i + 2) > 20 :=
by sorry

end twelve_gon_consecutive_sides_sum_l2441_244194


namespace family_weight_gain_l2441_244168

/-- The total weight gained by three family members at a reunion --/
theorem family_weight_gain (orlando_gain jose_gain fernando_gain : ℕ) : 
  orlando_gain = 5 →
  jose_gain = 2 * orlando_gain + 2 →
  fernando_gain = jose_gain / 2 - 3 →
  orlando_gain + jose_gain + fernando_gain = 20 := by
sorry

end family_weight_gain_l2441_244168


namespace max_piles_theorem_l2441_244101

/-- Represents the state of the stone piles -/
structure StonePiles where
  piles : List Nat
  sum_stones : Nat
  deriving Repr

/-- Check if the piles satisfy the size constraint -/
def valid_piles (sp : StonePiles) : Prop :=
  ∀ i j, i < sp.piles.length → j < sp.piles.length →
    2 * sp.piles[i]! > sp.piles[j]! ∧ 2 * sp.piles[j]! > sp.piles[i]!

/-- The initial state with 660 stones -/
def initial_state : StonePiles :=
  { piles := [660], sum_stones := 660 }

/-- A move splits one pile into two smaller piles -/
def move (sp : StonePiles) (index : Nat) (split : Nat) : Option StonePiles :=
  if index ≥ sp.piles.length ∨ split ≥ sp.piles[index]! then none
  else some {
    piles := sp.piles.set index (sp.piles[index]! - split) |>.insertNth index split,
    sum_stones := sp.sum_stones
  }

/-- The theorem to be proved -/
theorem max_piles_theorem (sp : StonePiles) :
  sp.sum_stones = 660 →
  valid_piles sp →
  sp.piles.length ≤ 30 :=
sorry

#eval initial_state

end max_piles_theorem_l2441_244101


namespace quadratic_equation_properties_l2441_244104

theorem quadratic_equation_properties (k : ℝ) :
  let f (x : ℝ) := x^2 + (2*k - 1)*x - k - 1
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  (x₁ + x₂ - 4*x₁*x₂ = 2 → k = -3/2) := by
  sorry

end quadratic_equation_properties_l2441_244104


namespace perpendicular_vectors_l2441_244124

def a : ℝ × ℝ := (1, 3)
def b (x : ℝ) : ℝ × ℝ := (x, -1)

theorem perpendicular_vectors (x : ℝ) : 
  (a.1 * (b x).1 + a.2 * (b x).2 = 0) → x = 3 := by sorry

end perpendicular_vectors_l2441_244124


namespace shuttle_speed_kph_l2441_244173

/-- Conversion factor from seconds to hours -/
def seconds_per_hour : ℕ := 3600

/-- Speed of the space shuttle in kilometers per second -/
def shuttle_speed_kps : ℝ := 6

/-- Theorem stating that the space shuttle's speed in kilometers per hour
    is equal to 21600 -/
theorem shuttle_speed_kph :
  shuttle_speed_kps * seconds_per_hour = 21600 := by
  sorry

end shuttle_speed_kph_l2441_244173


namespace unique_solution_l2441_244175

/-- Represents the number of students in a class -/
structure ClassSize where
  small : Nat
  large_min : Nat
  large_max : Nat

/-- Represents the number of classes for each school -/
structure SchoolClasses where
  shouchun_small : Nat
  binhu_small : Nat
  binhu_large : Nat

/-- Check if the given class distribution satisfies all conditions -/
def satisfies_conditions (cs : ClassSize) (sc : SchoolClasses) : Prop :=
  sc.shouchun_small + sc.binhu_small + sc.binhu_large = 45 ∧
  sc.binhu_small = 2 * sc.binhu_large ∧
  cs.small * (sc.shouchun_small + sc.binhu_small) + cs.large_min * sc.binhu_large ≤ 1800 ∧
  1800 ≤ cs.small * (sc.shouchun_small + sc.binhu_small) + cs.large_max * sc.binhu_large

theorem unique_solution (cs : ClassSize) (h_cs : cs.small = 36 ∧ cs.large_min = 70 ∧ cs.large_max = 75) :
  ∃! sc : SchoolClasses, satisfies_conditions cs sc ∧ 
    sc.shouchun_small = 30 ∧ sc.binhu_small = 10 ∧ sc.binhu_large = 5 :=
  sorry

end unique_solution_l2441_244175


namespace system_solution_l2441_244180

theorem system_solution (x y z w : ℤ) : 
  x + y + z + w = 20 ∧
  y + 2*z - 3*w = 28 ∧
  x - 2*y + z = 36 ∧
  -7*x - y + 5*z + 3*w = 84 →
  x = 4 ∧ y = -6 ∧ z = 20 ∧ w = 2 := by
sorry

end system_solution_l2441_244180


namespace polar_to_cartesian_line_l2441_244140

/-- The curve defined by the polar equation r = 1 / (sin θ + cos θ) is a straight line -/
theorem polar_to_cartesian_line : 
  ∀ (x y : ℝ), 
  (∃ (r θ : ℝ), r > 0 ∧ 
    x = r * Real.cos θ ∧ 
    y = r * Real.sin θ ∧ 
    r = 1 / (Real.sin θ + Real.cos θ)) 
  ↔ (x + y)^2 = 1 := by
  sorry

end polar_to_cartesian_line_l2441_244140


namespace square_of_number_ending_in_five_l2441_244182

theorem square_of_number_ending_in_five (n : ℕ) :
  ∃ k : ℕ, n = 10 * k + 5 →
    (n^2 % 100 = 25) ∧
    (n^2 = 100 * (k * (k + 1)) + 25) := by
  sorry

end square_of_number_ending_in_five_l2441_244182


namespace product_of_squares_l2441_244148

theorem product_of_squares (x y z : ℚ) 
  (hx : x = 1/4) 
  (hy : y = 1/2) 
  (hz : z = -8) : 
  x^2 * y^2 * z^2 = 1 := by sorry

end product_of_squares_l2441_244148


namespace line_segment_param_sum_l2441_244147

/-- Given a line segment connecting (1, -3) and (-4, 5), parameterized by x = pt + q and y = rt + s
    where 0 ≤ t ≤ 1 and t = 0 corresponds to (1, -3), prove that p^2 + q^2 + r^2 + s^2 = 99. -/
theorem line_segment_param_sum (p q r s : ℝ) : 
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → ∃ x y : ℝ, x = p * t + q ∧ y = r * t + s) →
  (q = 1 ∧ s = -3) →
  (p + q = -4 ∧ r + s = 5) →
  p^2 + q^2 + r^2 + s^2 = 99 := by
sorry

end line_segment_param_sum_l2441_244147


namespace hex_BF02_eq_48898_l2441_244160

/-- Converts a single hexadecimal digit to its decimal value -/
def hex_to_dec (c : Char) : ℕ :=
  match c with
  | 'B' => 11
  | 'F' => 15
  | '0' => 0
  | '2' => 2
  | _ => 0  -- Default case, should not be reached for this problem

/-- Converts a hexadecimal number represented as a string to its decimal value -/
def hex_to_decimal (s : String) : ℕ :=
  s.foldr (fun c acc => hex_to_dec c + 16 * acc) 0

/-- The hexadecimal number BF02 is equal to 48898 in decimal -/
theorem hex_BF02_eq_48898 : hex_to_decimal "BF02" = 48898 := by
  sorry

end hex_BF02_eq_48898_l2441_244160


namespace tangent_slope_at_one_l2441_244143

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 2

-- State the theorem
theorem tangent_slope_at_one :
  (deriv f) 1 = 2 :=
sorry

end tangent_slope_at_one_l2441_244143


namespace f_properties_l2441_244177

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - (1/2) * (x + a)^2

theorem f_properties (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f a x ≥ 0) →
  (HasDerivAt (f a) 1 0) →
  (∀ x : ℝ, StrictMono (f a)) ∧ 
  (a ≥ -Real.sqrt 2 ∧ a ≤ 2 - Real.log 2) := by
  sorry

end f_properties_l2441_244177


namespace fourth_task_completion_time_l2441_244120

-- Define a custom time type
structure Time where
  hours : Nat
  minutes : Nat

-- Define the problem parameters
def start_time : Time := { hours := 8, minutes := 45 }
def third_task_completion : Time := { hours := 11, minutes := 25 }
def num_tasks : Nat := 4

-- Calculate the time difference in minutes
def time_diff (t1 t2 : Time) : Nat :=
  (t2.hours * 60 + t2.minutes) - (t1.hours * 60 + t1.minutes)

-- Calculate the duration of a single task
def single_task_duration : Nat :=
  (time_diff start_time third_task_completion) / (num_tasks - 1)

-- Function to add minutes to a given time
def add_minutes (t : Time) (m : Nat) : Time :=
  let total_minutes := t.hours * 60 + t.minutes + m
  { hours := total_minutes / 60, minutes := total_minutes % 60 }

-- Theorem to prove
theorem fourth_task_completion_time :
  add_minutes third_task_completion single_task_duration = { hours := 12, minutes := 18 } := by
  sorry

end fourth_task_completion_time_l2441_244120


namespace centroid_of_concave_pentagon_l2441_244136

/-- A regular pentagon -/
structure RegularPentagon where
  vertices : Fin 5 → ℝ × ℝ
  is_regular : sorry

/-- A rhombus -/
structure Rhombus where
  vertices : Fin 4 → ℝ × ℝ
  is_rhombus : sorry

/-- The centroid of a planar figure -/
def centroid (figure : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- Theorem: Centroid of concave pentagonal plate -/
theorem centroid_of_concave_pentagon
  (ABCDE : RegularPentagon)
  (ABFE : Rhombus)
  (hF : F = sorry) -- F is the intersection of diagonals EC and BD
  (hABFE : sorry) -- ABFE is cut out from ABCDE
  : centroid (sorry) = F := by sorry

end centroid_of_concave_pentagon_l2441_244136


namespace total_people_in_program_l2441_244198

theorem total_people_in_program (parents pupils teachers : ℕ) 
  (h1 : parents = 73)
  (h2 : pupils = 724)
  (h3 : teachers = 744) :
  parents + pupils + teachers = 1541 := by
  sorry

end total_people_in_program_l2441_244198


namespace coin_problem_solution_l2441_244114

/-- Represents the number of coins of each denomination -/
structure CoinCount where
  one_jiao : ℕ
  five_jiao : ℕ

/-- Verifies if the given coin count satisfies the problem conditions -/
def is_valid_solution (coins : CoinCount) : Prop :=
  coins.one_jiao + coins.five_jiao = 30 ∧
  coins.one_jiao + 5 * coins.five_jiao = 86

/-- Theorem stating the unique solution to the coin problem -/
theorem coin_problem_solution : 
  ∃! (coins : CoinCount), is_valid_solution coins ∧ 
    coins.one_jiao = 16 ∧ coins.five_jiao = 14 := by sorry

end coin_problem_solution_l2441_244114


namespace batch_size_proof_l2441_244152

/-- The number of days it takes person A to complete the batch alone -/
def days_a : ℕ := 10

/-- The number of days it takes person B to complete the batch alone -/
def days_b : ℕ := 12

/-- The difference in parts processed by person A and B after working together for 1 day -/
def difference : ℕ := 40

/-- The total number of parts in the batch -/
def total_parts : ℕ := 2400

theorem batch_size_proof :
  (1 / days_a - 1 / days_b : ℚ) * total_parts = difference := by
  sorry

end batch_size_proof_l2441_244152


namespace percentage_of_270_l2441_244107

theorem percentage_of_270 : (33 + 1/3 : ℚ) / 100 * 270 = 90 := by sorry

end percentage_of_270_l2441_244107


namespace scarf_final_price_l2441_244128

def original_price : ℝ := 15
def first_discount : ℝ := 0.20
def second_discount : ℝ := 0.25

theorem scarf_final_price :
  original_price * (1 - first_discount) * (1 - second_discount) = 9 := by
  sorry

end scarf_final_price_l2441_244128


namespace f_10_equals_107_l2441_244110

/-- The function f defined as f(n) = n^2 - n + 17 for all n -/
def f (n : ℕ) : ℕ := n^2 - n + 17

/-- Theorem stating that f(10) = 107 -/
theorem f_10_equals_107 : f 10 = 107 := by
  sorry

end f_10_equals_107_l2441_244110


namespace multiplicative_inverse_modulo_l2441_244191

def A : Nat := 123456
def B : Nat := 142857
def M : Nat := 1000000

theorem multiplicative_inverse_modulo :
  (892857 * (A * B)) % M = 1 := by sorry

end multiplicative_inverse_modulo_l2441_244191


namespace swap_7_and_9_breaks_equality_l2441_244193

def original_number : ℕ := 271828
def swapped_number : ℕ := 291828
def target_sum : ℕ := 314159

def swap_digits (n : ℕ) (d1 d2 : ℕ) : ℕ := sorry

theorem swap_7_and_9_breaks_equality :
  swap_digits original_number 7 9 = swapped_number ∧
  swapped_number + original_number ≠ 2 * target_sum :=
sorry

end swap_7_and_9_breaks_equality_l2441_244193


namespace simplify_fraction_product_l2441_244186

theorem simplify_fraction_product : 5 * (14 / 9) * (27 / -63) = -30 := by
  sorry

end simplify_fraction_product_l2441_244186


namespace smallest_a_value_l2441_244162

/-- Given a polynomial x^3 - ax^2 + bx - 1890 with three positive integer roots,
    prove that the smallest possible value of a is 41 -/
theorem smallest_a_value (a b : ℤ) (x₁ x₂ x₃ : ℤ) : 
  (∀ x, x^3 - a*x^2 + b*x - 1890 = (x - x₁) * (x - x₂) * (x - x₃)) →
  x₁ > 0 → x₂ > 0 → x₃ > 0 →
  x₁ * x₂ * x₃ = 1890 →
  a = x₁ + x₂ + x₃ →
  ∀ a' : ℤ, (∃ b' x₁' x₂' x₃' : ℤ, 
    (∀ x, x^3 - a'*x^2 + b'*x - 1890 = (x - x₁') * (x - x₂') * (x - x₃')) ∧
    x₁' > 0 ∧ x₂' > 0 ∧ x₃' > 0 ∧
    x₁' * x₂' * x₃' = 1890) →
  a' ≥ 41 :=
by sorry

end smallest_a_value_l2441_244162


namespace squirrel_nuts_theorem_l2441_244145

/-- The number of nuts found by Pizizubka -/
def pizizubka_nuts : ℕ := 48

/-- The number of nuts found by Zrzečka -/
def zrzecka_nuts : ℕ := 96

/-- The number of nuts found by Ouška -/
def ouska_nuts : ℕ := 144

/-- The fraction of nuts Pizizubka ate -/
def pizizubka_ate : ℚ := 1/2

/-- The fraction of nuts Zrzečka ate -/
def zrzecka_ate : ℚ := 1/3

/-- The fraction of nuts Ouška ate -/
def ouska_ate : ℚ := 1/4

/-- The total number of nuts left -/
def total_nuts_left : ℕ := 196

theorem squirrel_nuts_theorem :
  zrzecka_nuts = 2 * pizizubka_nuts ∧
  ouska_nuts = 3 * pizizubka_nuts ∧
  (1 - pizizubka_ate) * pizizubka_nuts +
  (1 - zrzecka_ate) * zrzecka_nuts +
  (1 - ouska_ate) * ouska_nuts = total_nuts_left :=
by sorry

end squirrel_nuts_theorem_l2441_244145


namespace expression_simplification_l2441_244184

theorem expression_simplification (m : ℝ) (h1 : m^2 - 4 = 0) (h2 : m ≠ 2) :
  (m^2 + 6*m + 9) / (m - 2) / (m + 2 + (3*m + 4) / (m - 2)) = -1/2 := by
  sorry

end expression_simplification_l2441_244184


namespace points_on_line_l2441_244176

/-- Three points are collinear if they lie on the same straight line -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p3.2 - p1.2) * (p2.1 - p1.1) = (p2.2 - p1.2) * (p3.1 - p1.1)

/-- The problem statement -/
theorem points_on_line (k : ℝ) :
  collinear (1, 2) (3, -2) (4, k/3) → k = -12 := by
  sorry

end points_on_line_l2441_244176


namespace joe_needs_twelve_more_cars_l2441_244133

/-- Given that Joe has 50 toy cars initially and wants to have 62 cars in total,
    prove that the number of additional cars he needs is 12. -/
theorem joe_needs_twelve_more_cars (initial_cars : ℕ) (target_cars : ℕ) 
    (h1 : initial_cars = 50) (h2 : target_cars = 62) : 
    target_cars - initial_cars = 12 := by
  sorry

end joe_needs_twelve_more_cars_l2441_244133


namespace fib_150_mod_9_l2441_244125

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fib n + fib (n + 1)

-- Define the property of Fibonacci sequence modulo 9 repeating every 24 terms
axiom fib_mod_9_period_24 : ∀ n : ℕ, fib n % 9 = fib (n % 24) % 9

-- Theorem: The remainder when the 150th Fibonacci number is divided by 9 is 8
theorem fib_150_mod_9 : fib 150 % 9 = 8 := by
  sorry

end fib_150_mod_9_l2441_244125


namespace max_value_a_plus_2b_l2441_244113

theorem max_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a^2 + 2*a*b + 4*b^2 = 6) : 
  ∀ x y : ℝ, x > 0 → y > 0 → x^2 + 2*x*y + 4*y^2 = 6 → a + 2*b ≤ x + 2*y → a + 2*b ≤ 2 * Real.sqrt 2 :=
by
  sorry

end max_value_a_plus_2b_l2441_244113


namespace average_sale_is_7000_l2441_244134

/-- Calculates the average sale for six months given the sales of five months and a required sale for the sixth month. -/
def average_sale (sales : List ℕ) (required_sale : ℕ) : ℚ :=
  (sales.sum + required_sale) / 6

/-- Theorem stating that the average sale for the given problem is 7000. -/
theorem average_sale_is_7000 :
  let sales : List ℕ := [4000, 6524, 5689, 7230, 6000]
  let required_sale : ℕ := 12557
  average_sale sales required_sale = 7000 := by
  sorry

#eval average_sale [4000, 6524, 5689, 7230, 6000] 12557

end average_sale_is_7000_l2441_244134


namespace marcus_scored_half_l2441_244139

/-- Calculates the percentage of team points scored by Marcus -/
def marcus_percentage (three_point_goals : ℕ) (two_point_goals : ℕ) (team_total : ℕ) : ℚ :=
  let marcus_points := 3 * three_point_goals + 2 * two_point_goals
  (marcus_points : ℚ) / team_total * 100

/-- Proves that Marcus scored 50% of the team's total points -/
theorem marcus_scored_half (three_point_goals two_point_goals team_total : ℕ) 
  (h1 : three_point_goals = 5)
  (h2 : two_point_goals = 10)
  (h3 : team_total = 70) :
  marcus_percentage three_point_goals two_point_goals team_total = 50 := by
sorry

#eval marcus_percentage 5 10 70

end marcus_scored_half_l2441_244139


namespace min_value_theorem_l2441_244192

theorem min_value_theorem (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), m = 40/3 ∧ ∀ (a b c : ℝ), a^3 + b^3 + c^3 - 3*a*b*c = 8 → 
    a^2 + b^2 + c^2 + 2*a*c ≥ m ∧ 
    (∃ (p q r : ℝ), p^3 + q^3 + r^3 - 3*p*q*r = 8 ∧ p^2 + q^2 + r^2 + 2*p*r = m) :=
by sorry

end min_value_theorem_l2441_244192


namespace books_per_day_l2441_244164

def total_books : ℕ := 15
def total_days : ℕ := 3

theorem books_per_day : (total_books / total_days : ℚ) = 5 := by
  sorry

end books_per_day_l2441_244164


namespace equidistant_point_location_l2441_244151

-- Define a 2D point
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a quadrilateral
structure Quadrilateral where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

-- Define the property of being convex
def isConvex (q : Quadrilateral) : Prop := sorry

-- Define the distance between two points
def distance (p1 p2 : Point2D) : ℝ := sorry

-- Define the property of a point being equidistant from all vertices
def isEquidistant (p : Point2D) (q : Quadrilateral) : Prop :=
  distance p q.A = distance p q.B ∧
  distance p q.A = distance p q.C ∧
  distance p q.A = distance p q.D

-- Define the property of a point being inside a quadrilateral
def isInside (p : Point2D) (q : Quadrilateral) : Prop := sorry

-- Define the property of a point being outside a quadrilateral
def isOutside (p : Point2D) (q : Quadrilateral) : Prop := sorry

-- Define the property of a point being on the boundary of a quadrilateral
def isOnBoundary (p : Point2D) (q : Quadrilateral) : Prop := sorry

theorem equidistant_point_location (q : Quadrilateral) (h : isConvex q) :
  ∃ p : Point2D, isEquidistant p q ∧
    (isInside p q ∨ isOutside p q ∨ isOnBoundary p q) := by
  sorry

end equidistant_point_location_l2441_244151


namespace imaginary_part_of_z_l2441_244187

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * z = (1/2 : ℂ) * (1 + Complex.I)) :
  z.im = -1/2 := by sorry

end imaginary_part_of_z_l2441_244187


namespace power_two_100_mod_3_l2441_244165

theorem power_two_100_mod_3 : 2^100 ≡ 1 [ZMOD 3] := by sorry

end power_two_100_mod_3_l2441_244165


namespace max_profit_scheme_l2441_244106

-- Define the variables
def bean_sprout_price : ℚ := 60
def dried_tofu_price : ℚ := 40
def bean_sprout_sell : ℚ := 80
def dried_tofu_sell : ℚ := 55
def total_units : ℕ := 200
def max_cost : ℚ := 10440

-- Define the profit function
def profit (bean_sprouts : ℕ) : ℚ :=
  (bean_sprout_sell - bean_sprout_price) * bean_sprouts + 
  (dried_tofu_sell - dried_tofu_price) * (total_units - bean_sprouts)

-- Theorem statement
theorem max_profit_scheme :
  ∀ bean_sprouts : ℕ,
  (2 * bean_sprout_price + 3 * dried_tofu_price = 240) →
  (3 * bean_sprout_price + 4 * dried_tofu_price = 340) →
  (bean_sprouts + (total_units - bean_sprouts) = total_units) →
  (bean_sprout_price * bean_sprouts + dried_tofu_price * (total_units - bean_sprouts) ≤ max_cost) →
  (bean_sprouts ≥ (3/2) * (total_units - bean_sprouts)) →
  profit bean_sprouts ≤ profit 122 ∧ profit 122 = 3610 := by
sorry

end max_profit_scheme_l2441_244106


namespace trevor_placed_105_pieces_l2441_244181

/-- Represents the puzzle problem --/
def PuzzleProblem (total : ℕ) (border : ℕ) (missing : ℕ) (joeMultiplier : ℕ) :=
  {trevor : ℕ // 
    trevor + joeMultiplier * trevor + border + missing = total ∧
    trevor > 0 ∧
    joeMultiplier > 0}

theorem trevor_placed_105_pieces :
  ∃ (p : PuzzleProblem 500 75 5 3), p.val = 105 := by
  sorry

end trevor_placed_105_pieces_l2441_244181


namespace lcm_12_18_l2441_244196

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l2441_244196


namespace arrangements_count_l2441_244171

/-- The number of arrangements of 3 boys and 2 girls in a row with girls at both ends -/
def arrangements_with_girls_at_ends : ℕ :=
  let num_boys : ℕ := 3
  let num_girls : ℕ := 2
  let girl_arrangements : ℕ := 2  -- A_2^2
  let boy_arrangements : ℕ := 6  -- A_3^3
  girl_arrangements * boy_arrangements

/-- Theorem stating that the number of arrangements is 12 -/
theorem arrangements_count : arrangements_with_girls_at_ends = 12 := by
  sorry

end arrangements_count_l2441_244171


namespace zigzag_angle_theorem_l2441_244105

theorem zigzag_angle_theorem (ACB FEG DCE DEC : Real) : 
  ACB = 10 →
  FEG = 26 →
  DCE + 14 + 80 = 180 →
  DEC + 33 + 64 = 180 →
  ∃ θ : Real, θ = 11 ∧ θ + DCE + DEC = 180 :=
by sorry

end zigzag_angle_theorem_l2441_244105


namespace min_value_sum_reciprocals_l2441_244170

theorem min_value_sum_reciprocals (m n : ℝ) 
  (h1 : 2 * m + n = 2) 
  (h2 : m > 0) 
  (h3 : n > 0) : 
  (∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 2 → 1 / m + 2 / n ≤ 1 / x + 2 / y) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 2 ∧ 1 / x + 2 / y = 4) := by
  sorry

end min_value_sum_reciprocals_l2441_244170


namespace sin_from_tan_l2441_244112

theorem sin_from_tan (a b x : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 0 < x) (h4 : x < π / 2)
  (h5 : Real.tan x = 2 * a * b / (a^2 - b^2)) : 
  Real.sin x = 2 * a * b / (a^2 + b^2) := by
  sorry

end sin_from_tan_l2441_244112


namespace simplify_fraction_l2441_244146

theorem simplify_fraction (a : ℝ) (h1 : a ≠ 1) (h2 : a ≠ -1) :
  ((3 * a / (a^2 - 1) - 1 / (a - 1)) / ((2 * a - 1) / (a + 1))) = 1 / (a - 1) := by
  sorry

end simplify_fraction_l2441_244146


namespace coefficient_x10_is_179_l2441_244118

/-- The coefficient of x^10 in the expansion of (x+2)^10(x^2-1) -/
def coefficient_x10 : ℤ := 179

/-- The expansion of (x+2)^10(x^2-1) -/
def expansion (x : ℝ) : ℝ := (x + 2)^10 * (x^2 - 1)

/-- Theorem stating that the coefficient of x^10 in the expansion is equal to 179 -/
theorem coefficient_x10_is_179 : 
  (∃ f : ℝ → ℝ, ∀ x, expansion x = coefficient_x10 * x^10 + f x ∧ (∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → |f x| < ε * |x|^10)) :=
sorry

end coefficient_x10_is_179_l2441_244118


namespace bacon_tomatoes_difference_l2441_244188

/-- The number of students who suggested mashed potatoes -/
def mashed_potatoes : ℕ := 228

/-- The number of students who suggested bacon -/
def bacon : ℕ := 337

/-- The number of students who suggested tomatoes -/
def tomatoes : ℕ := 23

/-- The difference between the number of students who suggested bacon and tomatoes -/
theorem bacon_tomatoes_difference : bacon - tomatoes = 314 := by
  sorry

end bacon_tomatoes_difference_l2441_244188


namespace smallest_number_satisfying_conditions_l2441_244174

theorem smallest_number_satisfying_conditions : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 3 = 1 ∧ 
  n % 5 = 3 ∧ 
  n % 8 = 4 ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 3 = 1 ∧ m % 5 = 3 ∧ m % 8 = 4 → m ≥ n) ∧
  n = 28 :=
sorry

end smallest_number_satisfying_conditions_l2441_244174


namespace card_game_proof_l2441_244190

def deck_size : ℕ := 60
def hand_size : ℕ := 12

theorem card_game_proof :
  let combinations := Nat.choose deck_size hand_size
  ∃ (B : ℕ), 
    B = 7 ∧ 
    combinations = 17 * 10^10 + B * 10^9 + B * 10^7 + 5 * 10^6 + 2 * 10^5 + 9 * 10^4 + 8 * 10 + B ∧
    combinations % 6 = 0 := by
  sorry

end card_game_proof_l2441_244190


namespace find_unknown_number_l2441_244154

theorem find_unknown_number : ∃ x : ℝ, 
  (14 + x + 53) / 3 = (21 + 47 + 22) / 3 + 3 ∧ x = 32 := by
  sorry

end find_unknown_number_l2441_244154


namespace sine_of_angle_l2441_244122

/-- Given an angle α with vertex at the origin, initial side on the non-negative x-axis,
    and terminal side in the third quadrant intersecting the unit circle at (-√5/5, m),
    prove that sin α = -2√5/5 -/
theorem sine_of_angle (α : Real) (m : Real) : 
  ((-Real.sqrt 5 / 5) ^ 2 + m ^ 2 = 1) →  -- Point on unit circle
  (m < 0) →  -- In third quadrant
  (Real.sin α = m) →  -- Definition of sine
  (Real.sin α = -2 * Real.sqrt 5 / 5) := by
  sorry

end sine_of_angle_l2441_244122


namespace circle_radius_d_value_l2441_244111

theorem circle_radius_d_value (d : ℝ) :
  (∀ x y : ℝ, x^2 - 8*x + y^2 + 10*y + d = 0 → (x - 4)^2 + (y + 5)^2 = 36) →
  d = 5 := by
sorry

end circle_radius_d_value_l2441_244111


namespace trigonometric_expression_equality_l2441_244189

theorem trigonometric_expression_equality : 
  (Real.cos (190 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180))) / 
  (Real.sin (290 * π / 180) * Real.sqrt (1 - Real.cos (40 * π / 180))) = 2 * Real.sqrt 2 := by
  sorry

end trigonometric_expression_equality_l2441_244189


namespace probability_theorem_l2441_244109

def is_valid (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  1 ≤ a ∧ a ≤ 12 ∧
  1 ≤ b ∧ b ≤ 12 ∧
  1 ≤ c ∧ c ≤ 12 ∧
  a = 2 * b ∧ b = 2 * c

def total_assignments : ℕ := 12 * 11 * 10

def valid_assignments : ℕ := 3

theorem probability_theorem :
  (valid_assignments : ℚ) / total_assignments = 1 / 440 := by
  sorry

end probability_theorem_l2441_244109


namespace visitors_equal_cats_l2441_244195

/-- In a cat show scenario -/
structure CatShow where
  /-- The set of visitors -/
  visitors : Type
  /-- The set of cats -/
  cats : Type
  /-- The relation representing a visitor petting a cat -/
  pets : visitors → cats → Prop
  /-- Each visitor pets exactly three cats -/
  visitor_pets_three : ∀ v : visitors, ∃! (c₁ c₂ c₃ : cats), pets v c₁ ∧ pets v c₂ ∧ pets v c₃ ∧ c₁ ≠ c₂ ∧ c₁ ≠ c₃ ∧ c₂ ≠ c₃
  /-- Each cat is petted by exactly three visitors -/
  cat_petted_by_three : ∀ c : cats, ∃! (v₁ v₂ v₃ : visitors), pets v₁ c ∧ pets v₂ c ∧ pets v₃ c ∧ v₁ ≠ v₂ ∧ v₁ ≠ v₃ ∧ v₂ ≠ v₃

/-- The number of visitors is equal to the number of cats -/
theorem visitors_equal_cats (cs : CatShow) : Nonempty (Equiv cs.visitors cs.cats) := by
  sorry

end visitors_equal_cats_l2441_244195


namespace divisibility_condition_l2441_244138

/-- Sum of divisors of n -/
def A (n : ℕ+) : ℕ := sorry

/-- Sum of products of pairs of divisors of n -/
def B (n : ℕ+) : ℕ := sorry

/-- A positive integer n is a perfect square -/
def is_perfect_square (n : ℕ+) : Prop := ∃ m : ℕ+, n = m ^ 2

theorem divisibility_condition (n : ℕ+) : 
  (A n ∣ B n) ↔ is_perfect_square n := by sorry

end divisibility_condition_l2441_244138


namespace determinant_implies_cosine_l2441_244100

theorem determinant_implies_cosine (α : Real) : 
  (Real.cos (75 * π / 180) * Real.cos α + Real.sin (75 * π / 180) * Real.sin α = 1/3) →
  (Real.cos ((30 * π / 180) + 2 * α) = 7/9) := by
sorry

end determinant_implies_cosine_l2441_244100


namespace M_intersect_N_eq_N_l2441_244129

-- Define the sets M and N
def M : Set ℝ := {x | x > 0}
def N : Set ℝ := {x | Real.log x > 0}

-- State the theorem
theorem M_intersect_N_eq_N : M ∩ N = N := by sorry

end M_intersect_N_eq_N_l2441_244129


namespace sample_probability_l2441_244197

/-- Simple random sampling with given conditions -/
def SimpleRandomSampling (n : ℕ) : Prop :=
  n > 0 ∧ 
  (1 : ℚ) / n = 1 / 8 ∧
  ∀ i : ℕ, i ≤ n → (1 - (1 : ℚ) / n)^3 = (n - 1 : ℚ)^3 / n^3

theorem sample_probability (n : ℕ) (h : SimpleRandomSampling n) : 
  n = 8 ∧ (1 - (7 : ℚ) / 8^3) = 169 / 512 := by
  sorry

#check sample_probability

end sample_probability_l2441_244197


namespace functional_equation_identity_l2441_244161

open Function

theorem functional_equation_identity (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (y - f x) = f x - 2 * x + f (f y)) →
  (∀ x : ℝ, f x = x) :=
by sorry

end functional_equation_identity_l2441_244161


namespace problem_statement_l2441_244126

theorem problem_statement (n m : ℕ) : 
  2 * 8^n * 16^n = 2^15 →
  (∀ x y : ℝ, (m*x + y) * (2*x - y) = 2*m*x^2 - y^2) →
  n - m = 0 :=
by sorry

end problem_statement_l2441_244126


namespace object_ends_on_left_l2441_244121

/-- Represents the sides of a square --/
inductive SquareSide
  | Top
  | Right
  | Bottom
  | Left

/-- Represents the vertices of a regular octagon --/
inductive OctagonVertex
  | Bottom
  | BottomLeft
  | Left
  | TopLeft
  | Top
  | TopRight
  | Right
  | BottomRight

/-- The number of sides a square rolls to reach the leftmost position from the bottom --/
def numRolls : Nat := 4

/-- The angle of rotation for each roll of the square --/
def rotationPerRoll : Int := 135

/-- Function to calculate the final position of an object on a square
    after rolling around an octagon --/
def finalPosition (initialSide : SquareSide) (rolls : Nat) : SquareSide :=
  sorry

/-- Theorem stating that an object initially on the right side of the square
    will end up on the left side after rolling to the leftmost position --/
theorem object_ends_on_left :
  finalPosition SquareSide.Right numRolls = SquareSide.Left :=
  sorry

end object_ends_on_left_l2441_244121


namespace arithmetic_geometric_inequality_l2441_244159

theorem arithmetic_geometric_inequality (a b : ℝ) :
  (a + b) / 2 ≥ Real.sqrt (a * b) ∧
  ((a + b) / 2 = Real.sqrt (a * b) ↔ a = b) := by
  sorry

end arithmetic_geometric_inequality_l2441_244159


namespace number_of_combinations_prob_one_black_prob_at_least_one_blue_l2441_244103

/-- Represents the total number of pens -/
def total_pens : ℕ := 6

/-- Represents the number of black pens -/
def black_pens : ℕ := 3

/-- Represents the number of blue pens -/
def blue_pens : ℕ := 2

/-- Represents the number of red pens -/
def red_pens : ℕ := 1

/-- Represents the number of pens to be selected -/
def selected_pens : ℕ := 3

/-- Theorem stating the number of possible combinations when selecting 3 pens out of 6 -/
theorem number_of_combinations : Nat.choose total_pens selected_pens = 20 := by sorry

/-- Theorem stating the probability of selecting exactly one black pen -/
theorem prob_one_black : (Nat.choose black_pens 1 * Nat.choose (blue_pens + red_pens) 2) / Nat.choose total_pens selected_pens = 9 / 20 := by sorry

/-- Theorem stating the probability of selecting at least one blue pen -/
theorem prob_at_least_one_blue : 1 - (Nat.choose (black_pens + red_pens) selected_pens) / Nat.choose total_pens selected_pens = 4 / 5 := by sorry

end number_of_combinations_prob_one_black_prob_at_least_one_blue_l2441_244103


namespace tangent_x_axis_tangent_y_axis_unique_x_intercept_unique_y_intercept_is_parabola_l2441_244131

/-- A parabola represented by the equation (x + 1/2y - 1)² = 0 -/
def parabola (x y : ℝ) : Prop := (x + 1/2 * y - 1)^2 = 0

/-- The parabola is tangent to the x-axis at the point (1,0) -/
theorem tangent_x_axis : parabola 1 0 := by sorry

/-- The parabola is tangent to the y-axis at the point (0,2) -/
theorem tangent_y_axis : parabola 0 2 := by sorry

/-- The parabola touches the x-axis only at (1,0) -/
theorem unique_x_intercept (x : ℝ) : 
  parabola x 0 → x = 1 := by sorry

/-- The parabola touches the y-axis only at (0,2) -/
theorem unique_y_intercept (y : ℝ) : 
  parabola 0 y → y = 2 := by sorry

/-- The equation represents a parabola -/
theorem is_parabola : 
  ∃ (a b c : ℝ), ∀ (x y : ℝ), parabola x y ↔ y = a*x^2 + b*x + c := by sorry

end tangent_x_axis_tangent_y_axis_unique_x_intercept_unique_y_intercept_is_parabola_l2441_244131


namespace wednesday_speed_l2441_244163

/-- Jonathan's exercise routine -/
structure ExerciseRoutine where
  monday_speed : ℝ
  wednesday_speed : ℝ
  friday_speed : ℝ
  daily_distance : ℝ
  weekly_time : ℝ

/-- Theorem: Jonathan's walking speed on Wednesdays is 3 miles per hour -/
theorem wednesday_speed (routine : ExerciseRoutine)
  (h1 : routine.monday_speed = 2)
  (h2 : routine.friday_speed = 6)
  (h3 : routine.daily_distance = 6)
  (h4 : routine.weekly_time = 6) :
  routine.wednesday_speed = 3 := by
  sorry

end wednesday_speed_l2441_244163


namespace inverse_proportion_problem_l2441_244142

/-- Given that α is inversely proportional to β, prove that α = -8/3 when β = -3,
    given that α = 4 when β = 2. -/
theorem inverse_proportion_problem (α β : ℝ) (h1 : ∃ k, ∀ x y, x * y = k → (α = x ↔ β = y))
    (h2 : α = 4 ∧ β = 2) : β = -3 → α = -8/3 := by
  sorry

end inverse_proportion_problem_l2441_244142


namespace reinforcement_size_is_300_l2441_244179

/-- Calculates the reinforcement size given the initial garrison size, 
    initial provision duration, days passed, and remaining provision duration -/
def calculate_reinforcement (initial_garrison : ℕ) (initial_duration : ℕ) 
                             (days_passed : ℕ) (remaining_duration : ℕ) : ℕ :=
  let total_provisions := initial_garrison * initial_duration
  let provisions_left := initial_garrison * (initial_duration - days_passed)
  (provisions_left / remaining_duration) - initial_garrison

/-- Theorem stating that the reinforcement size is 300 given the problem conditions -/
theorem reinforcement_size_is_300 : 
  calculate_reinforcement 150 31 16 5 = 300 := by
  sorry

end reinforcement_size_is_300_l2441_244179


namespace linear_equation_solution_l2441_244117

theorem linear_equation_solution (x y a : ℝ) : 
  x = 1 → y = 2 → x - a * y = 3 → a = -1 := by sorry

end linear_equation_solution_l2441_244117


namespace min_value_of_transformed_sine_l2441_244172

theorem min_value_of_transformed_sine (φ : ℝ) (h : |φ| < π/2) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (2*x - π/3)
  ∃ (x : ℝ), x ∈ Set.Icc 0 (π/2) ∧ f x = -Real.sqrt 3 / 2 ∧
    ∀ y ∈ Set.Icc 0 (π/2), f y ≥ -Real.sqrt 3 / 2 := by
  sorry


end min_value_of_transformed_sine_l2441_244172


namespace total_cows_l2441_244199

theorem total_cows (cows_per_herd : ℕ) (num_herds : ℕ) 
  (h1 : cows_per_herd = 40) 
  (h2 : num_herds = 8) : 
  cows_per_herd * num_herds = 320 := by
  sorry

end total_cows_l2441_244199


namespace subtracted_amount_l2441_244116

theorem subtracted_amount (N : ℝ) (A : ℝ) : 
  N = 300 → 0.30 * N - A = 20 → A = 70 := by
  sorry

end subtracted_amount_l2441_244116


namespace inequality_and_equality_condition_l2441_244169

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (pos_d : d > 0)
  (h : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 ∧ 
  (a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d = 10 ↔ a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) :=
by sorry

end inequality_and_equality_condition_l2441_244169


namespace cross_product_result_l2441_244149

def u : ℝ × ℝ × ℝ := (3, 4, 2)
def v : ℝ × ℝ × ℝ := (1, -2, 5)

def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.1 * b.2.2 - a.2.2 * b.2.1,
   a.2.2 * b.1 - a.1 * b.2.2,
   a.1 * b.2.1 - a.2.1 * b.1)

theorem cross_product_result : cross_product u v = (24, -13, -10) := by
  sorry

end cross_product_result_l2441_244149


namespace r_plus_s_equals_six_l2441_244183

theorem r_plus_s_equals_six (r s : ℕ) (h1 : 2^r = 16) (h2 : 5^s = 25) : r + s = 6 := by
  sorry

end r_plus_s_equals_six_l2441_244183


namespace solution_set_when_a_is_one_range_of_a_when_f_leq_one_l2441_244157

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 5 - |x + a| - |x - 2|

theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≥ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 3} := by sorry

theorem range_of_a_when_f_leq_one :
  {a : ℝ | ∀ x, f a x ≤ 1} = {a : ℝ | a ≤ -6 ∨ a ≥ 2} := by sorry

end solution_set_when_a_is_one_range_of_a_when_f_leq_one_l2441_244157


namespace armband_cost_is_fifteen_l2441_244108

/-- The cost of an individual ride ticket in dollars -/
def ticket_cost : ℚ := 0.75

/-- The number of rides equivalent to the armband -/
def equivalent_rides : ℕ := 20

/-- The cost of the armband in dollars -/
def armband_cost : ℚ := ticket_cost * equivalent_rides

/-- Theorem stating that the armband costs $15.00 -/
theorem armband_cost_is_fifteen : armband_cost = 15 := by
  sorry

end armband_cost_is_fifteen_l2441_244108


namespace unique_campers_rowing_l2441_244167

theorem unique_campers_rowing (total_campers : ℕ) (morning : ℕ) (afternoon : ℕ) (evening : ℕ)
  (morning_and_afternoon : ℕ) (afternoon_and_evening : ℕ) (morning_and_evening : ℕ) (all_three : ℕ)
  (h1 : total_campers = 500)
  (h2 : morning = 235)
  (h3 : afternoon = 387)
  (h4 : evening = 142)
  (h5 : morning_and_afternoon = 58)
  (h6 : afternoon_and_evening = 23)
  (h7 : morning_and_evening = 15)
  (h8 : all_three = 8) :
  morning + afternoon + evening - (morning_and_afternoon + afternoon_and_evening + morning_and_evening) + all_three = 572 :=
by sorry

end unique_campers_rowing_l2441_244167


namespace points_six_units_from_negative_one_l2441_244141

theorem points_six_units_from_negative_one :
  let a : ℝ := -1
  let distance : ℝ := 6
  let point_left : ℝ := a - distance
  let point_right : ℝ := a + distance
  point_left = -7 ∧ point_right = 5 := by
sorry

end points_six_units_from_negative_one_l2441_244141


namespace max_value_theorem_l2441_244137

theorem max_value_theorem (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_eq : x^2 - x*y + 2*y^2 = 8) :
  x^2 + x*y + 2*y^2 ≤ (72 + 32*Real.sqrt 2) / 7 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀^2 - x₀*y₀ + 2*y₀^2 = 8 ∧
  x₀^2 + x₀*y₀ + 2*y₀^2 = (72 + 32*Real.sqrt 2) / 7 := by
  sorry

end max_value_theorem_l2441_244137


namespace equation_system_solution_l2441_244115

/-- Represents a solution to the equation system -/
structure Solution :=
  (x : ℚ)
  (y : ℚ)

/-- Represents the equation system 2ax + y = 5 and 2x - by = 13 -/
def EquationSystem (a b : ℚ) (sol : Solution) : Prop :=
  2 * a * sol.x + sol.y = 5 ∧ 2 * sol.x - b * sol.y = 13

/-- Theorem stating the conditions and the correct solution -/
theorem equation_system_solution :
  let personA : Solution := ⟨7/2, -2⟩
  let personB : Solution := ⟨3, -7⟩
  let correctSol : Solution := ⟨2, -3⟩
  ∀ a b : ℚ,
    (EquationSystem 1 b personA) →  -- Person A misread a as 1
    (EquationSystem a 1 personB) →  -- Person B misread b as 1
    (a = 2 ∧ b = 3) ∧               -- Correct values of a and b
    (EquationSystem a b correctSol) -- Correct solution
  := by sorry

end equation_system_solution_l2441_244115


namespace fraction_to_decimal_l2441_244127

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l2441_244127


namespace plant_structure_unique_solution_l2441_244123

/-- Represents a plant with branches and small branches -/
structure Plant where
  branches : ℕ
  smallBranchesPerBranch : ℕ

/-- The total number of parts (main stem, branches, and small branches) in a plant -/
def totalParts (p : Plant) : ℕ :=
  1 + p.branches + p.branches * p.smallBranchesPerBranch

/-- Theorem stating that a plant with 6 small branches per branch satisfies the given conditions -/
theorem plant_structure : ∃ (p : Plant), p.smallBranchesPerBranch = 6 ∧ totalParts p = 43 :=
  sorry

/-- Theorem proving that 6 is the unique solution for the number of small branches per branch -/
theorem unique_solution (p : Plant) (h : totalParts p = 43) : p.smallBranchesPerBranch = 6 :=
  sorry

end plant_structure_unique_solution_l2441_244123


namespace geometry_theorem_l2441_244102

-- Define the types for lines and planes
variable {Point : Type*}
variable {Line : Type*}
variable {Plane : Type*}

-- Define the necessary relations
variable (subset : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Define the theorem
theorem geometry_theorem 
  (m n l : Line) (α β : Plane) 
  (h_different_lines : m ≠ n ∧ m ≠ l ∧ n ≠ l) 
  (h_different_planes : α ≠ β) :
  (¬(subset m α) ∧ (subset n α) ∧ (parallel_lines m n) → parallel_line_plane m α) ∧
  ((subset m α) ∧ (perpendicular_line_plane m β) → perpendicular_planes α β) :=
sorry

end geometry_theorem_l2441_244102


namespace biff_hourly_rate_l2441_244156

/-- Biff's bus trip expenses and earnings -/
def biff_trip (hourly_rate : ℚ) : Prop :=
  let ticket : ℚ := 11
  let snacks : ℚ := 3
  let headphones : ℚ := 16
  let wifi_rate : ℚ := 2
  let trip_duration : ℚ := 3
  let total_expenses : ℚ := ticket + snacks + headphones + wifi_rate * trip_duration
  hourly_rate * trip_duration = total_expenses

/-- Theorem stating Biff's hourly rate for online work -/
theorem biff_hourly_rate : 
  ∃ (rate : ℚ), biff_trip rate ∧ rate = 12 := by
  sorry

end biff_hourly_rate_l2441_244156


namespace sally_savings_l2441_244130

/-- Represents the trip expenses and savings for Sally's Sea World trip --/
structure SeaWorldTrip where
  parking_cost : ℕ
  entrance_cost : ℕ
  meal_pass_cost : ℕ
  distance_to_sea_world : ℕ
  car_efficiency : ℕ
  gas_cost_per_gallon : ℕ
  additional_savings_needed : ℕ

/-- Calculates the total cost of the trip --/
def total_cost (trip : SeaWorldTrip) : ℕ :=
  trip.parking_cost + trip.entrance_cost + trip.meal_pass_cost +
  (2 * trip.distance_to_sea_world * trip.gas_cost_per_gallon + trip.car_efficiency - 1) / trip.car_efficiency

/-- Theorem stating that Sally has already saved $28 --/
theorem sally_savings (trip : SeaWorldTrip)
  (h1 : trip.parking_cost = 10)
  (h2 : trip.entrance_cost = 55)
  (h3 : trip.meal_pass_cost = 25)
  (h4 : trip.distance_to_sea_world = 165)
  (h5 : trip.car_efficiency = 30)
  (h6 : trip.gas_cost_per_gallon = 3)
  (h7 : trip.additional_savings_needed = 95) :
  total_cost trip - trip.additional_savings_needed = 28 := by
  sorry


end sally_savings_l2441_244130
