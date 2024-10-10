import Mathlib

namespace relay_race_arrangements_l1018_101890

theorem relay_race_arrangements (total_students : Nat) (boys : Nat) (girls : Nat) 
  (selected_students : Nat) (selected_boys : Nat) (selected_girls : Nat) : 
  total_students = 8 →
  boys = 6 →
  girls = 2 →
  selected_students = 4 →
  selected_boys = 3 →
  selected_girls = 1 →
  (Nat.choose girls selected_girls) * 
  (Nat.choose boys selected_boys) * 
  selected_boys * 
  (Nat.factorial (selected_students - 1)) = 720 := by
sorry

end relay_race_arrangements_l1018_101890


namespace minimize_y_l1018_101876

/-- The function y in terms of x, a, and b -/
def y (x a b : ℝ) : ℝ := (x - a)^3 + (x - b)^3

/-- The theorem stating that (a+b)/2 minimizes y -/
theorem minimize_y (a b : ℝ) :
  ∃ (x : ℝ), ∀ (z : ℝ), y z a b ≥ y x a b ∧ x = (a + b) / 2 :=
sorry

end minimize_y_l1018_101876


namespace tetrahedron_properties_l1018_101821

def A1 : ℝ × ℝ × ℝ := (3, 10, -1)
def A2 : ℝ × ℝ × ℝ := (-2, 3, -5)
def A3 : ℝ × ℝ × ℝ := (-6, 0, -3)
def A4 : ℝ × ℝ × ℝ := (1, -1, 2)

def tetrahedron_volume (A1 A2 A3 A4 : ℝ × ℝ × ℝ) : ℝ := sorry

def tetrahedron_height (A1 A2 A3 A4 : ℝ × ℝ × ℝ) : ℝ := sorry

theorem tetrahedron_properties :
  tetrahedron_volume A1 A2 A3 A4 = 45.5 ∧
  tetrahedron_height A1 A2 A3 A4 = 7 := by sorry

end tetrahedron_properties_l1018_101821


namespace factorial_ratio_squared_l1018_101820

theorem factorial_ratio_squared : (Nat.factorial 10 / Nat.factorial 9) ^ 2 = 100 := by
  sorry

end factorial_ratio_squared_l1018_101820


namespace painted_cubes_count_l1018_101888

/-- Represents a cube constructed from unit cubes -/
structure LargeCube where
  side_length : ℕ
  unpainted_cubes : ℕ

/-- Calculates the number of cubes with at least one face painted -/
def painted_cubes (c : LargeCube) : ℕ :=
  c.side_length ^ 3 - c.unpainted_cubes

/-- The theorem states that for a cube with 22 unpainted cubes,
    42 cubes have at least one face painted red -/
theorem painted_cubes_count (c : LargeCube) 
  (h1 : c.unpainted_cubes = 22) 
  (h2 : c.side_length = 4) : 
  painted_cubes c = 42 := by
  sorry

#check painted_cubes_count

end painted_cubes_count_l1018_101888


namespace race_distance_l1018_101858

theorem race_distance (race_length : ℝ) (gap : ℝ) : 
  race_length > 0 → 
  gap > 0 → 
  gap < race_length → 
  let v1 := race_length
  let v2 := race_length - gap
  let v3 := (race_length - gap) * ((race_length - gap) / race_length)
  (race_length - v3) = 19 := by
  sorry

end race_distance_l1018_101858


namespace abs_value_of_specific_complex_l1018_101837

/-- Given a complex number z = (1-i)/i, prove that its absolute value |z| is equal to √2 -/
theorem abs_value_of_specific_complex : let z : ℂ := (1 - Complex.I) / Complex.I
  Complex.abs z = Real.sqrt 2 := by
  sorry

end abs_value_of_specific_complex_l1018_101837


namespace inequality_proof_l1018_101864

theorem inequality_proof (x y z : ℝ) : 
  (x^2 + y^2 + z^2) * ((x^2 + y^2 + z^2)^2 - (x*y + y*z + z*x)^2) ≥ 
  (x + y + z)^2 * ((x^2 + y^2 + z^2) - (x*y + y*z + z*x))^2 := by
  sorry

end inequality_proof_l1018_101864


namespace boys_neither_happy_nor_sad_l1018_101811

/-- Given information about children's emotions and gender distribution -/
structure ChildrenData where
  total : Nat
  happy : Nat
  sad : Nat
  neither : Nat
  boys : Nat
  girls : Nat
  happy_boys : Nat
  sad_girls : Nat

/-- Theorem stating the number of boys who are neither happy nor sad -/
theorem boys_neither_happy_nor_sad (data : ChildrenData)
  (h1 : data.total = 60)
  (h2 : data.happy = 30)
  (h3 : data.sad = 10)
  (h4 : data.neither = 20)
  (h5 : data.boys = 19)
  (h6 : data.girls = 41)
  (h7 : data.happy_boys = 6)
  (h8 : data.sad_girls = 4)
  (h9 : data.total = data.happy + data.sad + data.neither)
  (h10 : data.total = data.boys + data.girls) :
  data.boys - data.happy_boys - (data.sad - data.sad_girls) = 7 := by
  sorry


end boys_neither_happy_nor_sad_l1018_101811


namespace selection_plans_l1018_101800

theorem selection_plans (n m : ℕ) (h1 : n = 6) (h2 : m = 3) : 
  (n.choose m) * m.factorial = 120 := by
  sorry

end selection_plans_l1018_101800


namespace function_domain_range_equality_l1018_101856

/-- The function f(x) = x^2 - 2x + 2 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

/-- The theorem stating that b = 2 for the given conditions -/
theorem function_domain_range_equality (b : ℝ) (h1 : b > 1) 
  (h2 : Set.Icc 1 b = Set.range f)
  (h3 : ∀ x, x ∈ Set.Icc 1 b → f x ∈ Set.Icc 1 b) : b = 2 := by
  sorry

#check function_domain_range_equality

end function_domain_range_equality_l1018_101856


namespace new_cards_count_l1018_101843

def cards_per_page : ℕ := 3
def pages_used : ℕ := 6
def old_cards : ℕ := 10

theorem new_cards_count :
  (cards_per_page * pages_used) - old_cards = 8 :=
by
  sorry

end new_cards_count_l1018_101843


namespace least_subtraction_for_divisibility_l1018_101880

theorem least_subtraction_for_divisibility : 
  ∃! x : ℕ, x ≤ 86 ∧ (13605 - x) % 87 = 0 ∧ ∀ y : ℕ, y < x → (13605 - y) % 87 ≠ 0 :=
by sorry

end least_subtraction_for_divisibility_l1018_101880


namespace smallest_five_times_decrease_five_times_decrease_form_no_twelve_times_decrease_divisible_by_k_condition_l1018_101877

def is_valid_number (N : ℕ) : Prop :=
  ∃ (x n : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ N = x * 10^n + (N % 10^n)

theorem smallest_five_times_decrease (N : ℕ) :
  is_valid_number N →
  (∃ (x n : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ N = x * 10^n + (N % 10^n) ∧ N = 5 * (N % 10^n)) →
  N ≥ 25 :=
sorry

theorem five_times_decrease_form (N : ℕ) :
  is_valid_number N →
  (∃ (x n : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ N = x * 10^n + (N % 10^n) ∧ N = 5 * (N % 10^n)) →
  ∃ (m : ℕ), N = 12 * 10^m ∨ N = 24 * 10^m ∨ N = 36 * 10^m ∨ N = 48 * 10^m :=
sorry

theorem no_twelve_times_decrease (N : ℕ) :
  is_valid_number N →
  ¬(∃ (x n : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ N = x * 10^n + (N % 10^n) ∧ N = 12 * (N % 10^n)) :=
sorry

theorem divisible_by_k_condition (k : ℕ) :
  (∃ (N : ℕ), is_valid_number N ∧ 
   ∃ (x n : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ N = x * 10^n + (N % 10^n) ∧ k ∣ (N % 10^n)) ↔
  ∃ (x a b : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ a + b > 0 ∧ k = x * 2^a * 5^b :=
sorry

end smallest_five_times_decrease_five_times_decrease_form_no_twelve_times_decrease_divisible_by_k_condition_l1018_101877


namespace cos_eq_neg_mul_sin_at_beta_l1018_101897

open Real

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := |cos x| - k * x

theorem cos_eq_neg_mul_sin_at_beta
  (k : ℝ) (hk : k > 0)
  (α β : ℝ) (hα : α > 0) (hβ : β > 0) (hαβ : α < β)
  (hzeros : ∀ x, x > 0 → f k x = 0 ↔ x = α ∨ x = β)
  : cos β = -β * sin β :=
sorry

end cos_eq_neg_mul_sin_at_beta_l1018_101897


namespace monic_quartic_value_at_zero_l1018_101882

def is_monic_quartic (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + d

theorem monic_quartic_value_at_zero 
  (f : ℝ → ℝ) 
  (h_monic : is_monic_quartic f)
  (h_m2 : f (-2) = -4)
  (h_1 : f 1 = -1)
  (h_3 : f 3 = -9)
  (h_5 : f 5 = -25) :
  f 0 = 30 := by
sorry

end monic_quartic_value_at_zero_l1018_101882


namespace evaluate_expression_l1018_101868

theorem evaluate_expression :
  (3 * Real.sqrt 8) / (Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7) =
  -(6 * (Real.sqrt 2 - Real.sqrt 6 - Real.sqrt 10 - Real.sqrt 14)) / 13 := by
  sorry

end evaluate_expression_l1018_101868


namespace total_pencils_is_fifty_l1018_101884

/-- The number of pencils Sabrina has -/
def sabrina_pencils : ℕ := 14

/-- The number of pencils Justin has -/
def justin_pencils : ℕ := 2 * sabrina_pencils + 8

/-- The total number of pencils Justin and Sabrina have combined -/
def total_pencils : ℕ := justin_pencils + sabrina_pencils

/-- Theorem stating that the total number of pencils is 50 -/
theorem total_pencils_is_fifty : total_pencils = 50 := by
  sorry

end total_pencils_is_fifty_l1018_101884


namespace apple_count_correct_l1018_101885

/-- The number of apples in a box containing apples and oranges -/
def num_apples : ℕ := 14

/-- The initial number of oranges in the box -/
def initial_oranges : ℕ := 20

/-- The number of oranges removed from the box -/
def removed_oranges : ℕ := 14

/-- The percentage of apples after removing oranges -/
def apple_percentage : ℝ := 0.7

theorem apple_count_correct :
  num_apples = 14 ∧
  initial_oranges = 20 ∧
  removed_oranges = 14 ∧
  apple_percentage = 0.7 ∧
  (num_apples : ℝ) / ((num_apples : ℝ) + (initial_oranges - removed_oranges : ℝ)) = apple_percentage :=
by sorry

end apple_count_correct_l1018_101885


namespace slope_of_line_l1018_101886

/-- The slope of a line represented by the equation 4x + 5y = 20 is -4/5 -/
theorem slope_of_line (x y : ℝ) : 4 * x + 5 * y = 20 → (y - 4) / x = -4 / 5 := by
  sorry

end slope_of_line_l1018_101886


namespace optimal_group_division_l1018_101822

theorem optimal_group_division (total_members : ℕ) (large_group_size : ℕ) (small_group_size : ℕ) 
  (h1 : total_members = 90)
  (h2 : large_group_size = 7)
  (h3 : small_group_size = 3) :
  ∃ (large_groups small_groups : ℕ),
    large_groups * large_group_size + small_groups * small_group_size = total_members ∧
    large_groups = 12 ∧
    ∀ (lg sg : ℕ), lg * large_group_size + sg * small_group_size = total_members → lg ≤ large_groups :=
by
  sorry

end optimal_group_division_l1018_101822


namespace track_circumference_l1018_101845

/-- Represents the circular track and the movement of A and B -/
structure CircularTrack where
  circumference : ℝ
  speed_A : ℝ
  speed_B : ℝ

/-- The conditions of the problem -/
def problem_conditions (track : CircularTrack) : Prop :=
  ∃ (first_meet second_meet : ℝ),
    -- B has traveled 150 yards at first meeting
    track.speed_B * first_meet = 150 ∧
    -- A is 90 yards away from completing one lap at second meeting
    track.speed_A * second_meet = track.circumference - 90 ∧
    -- B's total distance at second meeting
    track.speed_B * second_meet = track.circumference / 2 + 90 ∧
    -- A and B start from opposite points and move in opposite directions
    track.speed_A > 0 ∧ track.speed_B > 0

/-- The theorem to prove -/
theorem track_circumference :
  ∀ (track : CircularTrack),
    problem_conditions track →
    track.circumference = 720 := by
  sorry

end track_circumference_l1018_101845


namespace pie_shop_earnings_l1018_101875

def price_per_slice : ℕ := 3
def slices_per_pie : ℕ := 10
def number_of_pies : ℕ := 6

theorem pie_shop_earnings : 
  price_per_slice * slices_per_pie * number_of_pies = 180 := by
  sorry

end pie_shop_earnings_l1018_101875


namespace sum_equals_1332_l1018_101814

/-- Converts a base 4 number (represented as a list of digits) to its decimal equivalent -/
def base4ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 4 * acc + d) 0

/-- Converts a decimal number to its base 4 representation (as a list of digits) -/
def decimalToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- The sum of 232₄, 121₄, and 313₄ in base 4 -/
def sumInBase4 : List Nat :=
  decimalToBase4 (base4ToDecimal [2,3,2] + base4ToDecimal [1,2,1] + base4ToDecimal [3,1,3])

theorem sum_equals_1332 : sumInBase4 = [1,3,3,2] := by
  sorry

end sum_equals_1332_l1018_101814


namespace mailman_junk_mail_l1018_101833

/-- Given a total number of mail pieces and a number of magazines, 
    calculate the number of junk mail pieces. -/
def junk_mail (total : ℕ) (magazines : ℕ) : ℕ :=
  total - magazines

theorem mailman_junk_mail :
  junk_mail 11 5 = 6 := by
  sorry

end mailman_junk_mail_l1018_101833


namespace quadratic_root_complex_l1018_101865

theorem quadratic_root_complex (c d : ℝ) : 
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  (3 - 4 * Complex.I : ℂ) ^ 2 + c * (3 - 4 * Complex.I : ℂ) + d = 0 →
  c = -6 ∧ d = 25 := by sorry

end quadratic_root_complex_l1018_101865


namespace sqrt_six_over_sqrt_two_equals_sqrt_three_l1018_101838

theorem sqrt_six_over_sqrt_two_equals_sqrt_three : 
  Real.sqrt 6 / Real.sqrt 2 = Real.sqrt 3 := by
  sorry

end sqrt_six_over_sqrt_two_equals_sqrt_three_l1018_101838


namespace friendship_theorem_l1018_101839

/-- Represents a simple undirected graph with 6 vertices -/
def Graph := Fin 6 → Fin 6 → Bool

/-- The friendship relation is symmetric -/
def symmetric (g : Graph) : Prop :=
  ∀ i j : Fin 6, g i j = g j i

/-- A set of three vertices form a triangle in the graph -/
def isTriangle (g : Graph) (v1 v2 v3 : Fin 6) : Prop :=
  v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧
  ((g v1 v2 ∧ g v2 v3 ∧ g v1 v3) ∨ (¬g v1 v2 ∧ ¬g v2 v3 ∧ ¬g v1 v3))

/-- Main theorem: any graph with 6 vertices contains a monochromatic triangle -/
theorem friendship_theorem (g : Graph) (h : symmetric g) :
  ∃ v1 v2 v3 : Fin 6, isTriangle g v1 v2 v3 := by
  sorry

end friendship_theorem_l1018_101839


namespace cosine_in_special_triangle_l1018_101805

/-- Given a triangle ABC where the sides a, b, and c are in the ratio 2:3:4, 
    prove that cos C = -1/4 -/
theorem cosine_in_special_triangle (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
    (ratio : ∃ (x : ℝ), x > 0 ∧ a = 2*x ∧ b = 3*x ∧ c = 4*x) : 
    (a^2 + b^2 - c^2) / (2*a*b) = -1/4 := by
  sorry

end cosine_in_special_triangle_l1018_101805


namespace percy_swimming_weeks_l1018_101801

/-- Represents Percy's swimming schedule and calculates the number of weeks to swim a given total hours -/
def swimming_schedule (weekday_hours_per_day : ℕ) (weekday_days : ℕ) (weekend_hours : ℕ) (total_hours : ℕ) : ℕ :=
  let hours_per_week := weekday_hours_per_day * weekday_days + weekend_hours
  total_hours / hours_per_week

/-- Proves that Percy's swimming schedule over 52 hours covers 4 weeks -/
theorem percy_swimming_weeks : swimming_schedule 2 5 3 52 = 4 := by
  sorry

#eval swimming_schedule 2 5 3 52

end percy_swimming_weeks_l1018_101801


namespace remaining_kittens_l1018_101841

def initial_kittens : ℕ := 8
def given_away : ℕ := 2

theorem remaining_kittens : initial_kittens - given_away = 6 := by
  sorry

end remaining_kittens_l1018_101841


namespace apples_in_box_l1018_101806

/-- The number of apples in a box -/
def apples_per_box : ℕ := 14

/-- The number of people eating apples -/
def num_people : ℕ := 2

/-- The number of weeks spent eating apples -/
def num_weeks : ℕ := 3

/-- The number of boxes of apples -/
def num_boxes : ℕ := 3

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of apples eaten per person per day -/
def apples_per_person_per_day : ℕ := 1

theorem apples_in_box :
  apples_per_box * num_boxes = num_people * apples_per_person_per_day * num_weeks * days_per_week :=
by sorry

end apples_in_box_l1018_101806


namespace concertHallSeats_l1018_101883

/-- Represents a concert hall with a specific seating arrangement. -/
structure ConcertHall where
  rows : ℕ
  middleRowSeats : ℕ
  middleRowIndex : ℕ
  increaseFactor : ℕ

/-- Calculates the total number of seats in the concert hall. -/
def totalSeats (hall : ConcertHall) : ℕ :=
  let firstRowSeats := hall.middleRowSeats - 2 * (hall.middleRowIndex - 1)
  let lastRowSeats := hall.middleRowSeats + 2 * (hall.rows - hall.middleRowIndex)
  hall.rows * (firstRowSeats + lastRowSeats) / 2

/-- Theorem stating that a concert hall with the given properties has 1984 seats. -/
theorem concertHallSeats (hall : ConcertHall) 
    (h1 : hall.rows = 31)
    (h2 : hall.middleRowSeats = 64)
    (h3 : hall.middleRowIndex = 16)
    (h4 : hall.increaseFactor = 2) : 
  totalSeats hall = 1984 := by
  sorry


end concertHallSeats_l1018_101883


namespace min_value_theorem_l1018_101819

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 5) :
  9/x + 4/y + 25/z ≥ 20 ∧ ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧ x' + y' + z' = 5 ∧ 9/x' + 4/y' + 25/z' = 20 :=
by sorry

end min_value_theorem_l1018_101819


namespace correct_equation_l1018_101872

theorem correct_equation (a b : ℝ) : 3 * a^2 * b - 4 * b * a^2 = -a^2 * b := by
  sorry

end correct_equation_l1018_101872


namespace stream_speed_l1018_101830

/-- Proves that the speed of a stream is 19 kmph given the conditions of the rowing problem -/
theorem stream_speed (boat_speed : ℝ) (upstream_time downstream_time : ℝ) : 
  boat_speed = 57 →
  upstream_time = 2 * downstream_time →
  (boat_speed - 19) * (boat_speed + 19) = boat_speed^2 :=
by
  sorry

#eval (57 : ℝ) - 19 -- Expected output: 38
#eval (57 : ℝ) + 19 -- Expected output: 76
#eval (57 : ℝ)^2    -- Expected output: 3249
#eval 38 * 76       -- Expected output: 2888

end stream_speed_l1018_101830


namespace peter_and_laura_seating_probability_l1018_101844

-- Define the number of chairs
def num_chairs : ℕ := 10

-- Define the probability of not sitting next to each other
def prob_not_adjacent : ℚ := 4 / 5

-- Theorem statement
theorem peter_and_laura_seating_probability :
  let total_ways := num_chairs.choose 2
  let adjacent_ways := num_chairs - 1
  prob_not_adjacent = 1 - (adjacent_ways : ℚ) / (total_ways : ℚ) := by
  sorry

end peter_and_laura_seating_probability_l1018_101844


namespace arithmetic_mean_problem_l1018_101887

/-- Given that the arithmetic mean of six expressions is 30, prove that x = 18.5 and y = 10. -/
theorem arithmetic_mean_problem (x y : ℝ) :
  ((2*x - y) + 20 + (3*x + y) + 16 + (x + 5) + (y + 8)) / 6 = 30 →
  x = 18.5 ∧ y = 10 := by
sorry

end arithmetic_mean_problem_l1018_101887


namespace min_ticket_cost_is_800_l1018_101808

/-- Represents the ticket pricing structure and group composition --/
structure TicketPricing where
  adultPrice : ℕ
  childPrice : ℕ
  groupPrice : ℕ
  groupMinSize : ℕ
  numAdults : ℕ
  numChildren : ℕ

/-- Calculates the minimum cost for tickets given the pricing structure --/
def minTicketCost (pricing : TicketPricing) : ℕ :=
  sorry

/-- Theorem stating that the minimum cost for the given scenario is 800 yuan --/
theorem min_ticket_cost_is_800 :
  let pricing : TicketPricing := {
    adultPrice := 100,
    childPrice := 50,
    groupPrice := 70,
    groupMinSize := 10,
    numAdults := 8,
    numChildren := 4
  }
  minTicketCost pricing = 800 := by sorry

end min_ticket_cost_is_800_l1018_101808


namespace three_digit_prime_not_divisor_of_permutation_l1018_101898

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def is_permutation (a b : ℕ) : Prop :=
  ∃ (x y z : ℕ), a = 100*x + 10*y + z ∧ b = 100*y + 10*z + x ∨
                  a = 100*x + 10*y + z ∧ b = 100*z + 10*x + y ∨
                  a = 100*x + 10*y + z ∧ b = 100*y + 10*x + z ∨
                  a = 100*x + 10*y + z ∧ b = 100*z + 10*y + x ∨
                  a = 100*x + 10*y + z ∧ b = 100*x + 10*z + y

theorem three_digit_prime_not_divisor_of_permutation (p : ℕ) (h1 : is_three_digit p) (h2 : Nat.Prime p) :
  ∀ n : ℕ, is_permutation p n → ¬(n % p = 0) := by
  sorry

end three_digit_prime_not_divisor_of_permutation_l1018_101898


namespace constant_magnitude_l1018_101869

theorem constant_magnitude (z₁ z₂ : ℂ) (h₁ : Complex.abs z₁ = 5) 
  (h₂ : ∀ θ : ℝ, z₁^2 - z₁ * z₂ * Complex.sin θ + z₂^2 = 0) : 
  Complex.abs z₂ = 5 := by
  sorry

end constant_magnitude_l1018_101869


namespace point_in_first_quadrant_l1018_101810

-- Define the complex number
def z : ℂ := Complex.I * (2 - Complex.I)

-- Theorem statement
theorem point_in_first_quadrant :
  Complex.re z > 0 ∧ Complex.im z > 0 := by
  sorry

end point_in_first_quadrant_l1018_101810


namespace problem_solution_l1018_101829

theorem problem_solution (t : ℚ) (x y : ℚ) 
  (h1 : x = 3 - 2 * t)
  (h2 : y = 5 * t + 6)
  (h3 : x = -2) :
  y = 37 / 2 := by
  sorry

end problem_solution_l1018_101829


namespace makeup_palette_cost_l1018_101816

/-- The cost of a makeup palette given the following conditions:
  * There are 3 makeup palettes
  * 4 lipsticks cost $2.50 each
  * 3 boxes of hair color cost $4 each
  * The total cost is $67
-/
theorem makeup_palette_cost :
  let num_palettes : ℕ := 3
  let num_lipsticks : ℕ := 4
  let lipstick_cost : ℚ := 5/2
  let num_hair_color : ℕ := 3
  let hair_color_cost : ℚ := 4
  let total_cost : ℚ := 67
  (total_cost - (num_lipsticks * lipstick_cost + num_hair_color * hair_color_cost)) / num_palettes = 15 := by
  sorry

end makeup_palette_cost_l1018_101816


namespace circus_tickets_l1018_101818

theorem circus_tickets (ticket_cost : ℕ) (total_spent : ℕ) (h1 : ticket_cost = 44) (h2 : total_spent = 308) :
  total_spent / ticket_cost = 7 :=
sorry

end circus_tickets_l1018_101818


namespace nine_grams_combinations_l1018_101848

def weight_combinations (n : ℕ) : ℕ :=
  let ones := Finset.range 4
  let twos := Finset.range 4
  let fives := Finset.range 2
  (ones.product twos).product fives
    |>.filter (fun ((a, b), c) => a + 2*b + 5*c == n)
    |>.card

theorem nine_grams_combinations : weight_combinations 9 = 8 := by
  sorry

end nine_grams_combinations_l1018_101848


namespace decimal_place_150_is_3_l1018_101828

/-- The decimal representation of 7/11 repeats every 2 digits -/
def repeat_length : ℕ := 2

/-- The repeating decimal representation of 7/11 -/
def decimal_rep : List ℕ := [6, 3]

/-- The 150th decimal place of 7/11 -/
def decimal_place_150 : ℕ := 
  decimal_rep[(150 - 1) % repeat_length]

theorem decimal_place_150_is_3 : decimal_place_150 = 3 := by sorry

end decimal_place_150_is_3_l1018_101828


namespace hyperbola_asymptote_distance_l1018_101874

theorem hyperbola_asymptote_distance (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ d : ℝ, d = (2 * b) / Real.sqrt (4 + b^2) ∧ d = Real.sqrt 2) →
  b = 2 := by
  sorry

end hyperbola_asymptote_distance_l1018_101874


namespace no_linear_term_implies_m_equals_9_l1018_101866

theorem no_linear_term_implies_m_equals_9 (m : ℝ) : 
  (∀ x : ℝ, ∃ a b : ℝ, (x - 3) * (3 * x + m) = a * x^2 + b) → m = 9 := by
  sorry

end no_linear_term_implies_m_equals_9_l1018_101866


namespace intersection_of_M_and_N_l1018_101862

def M : Set ℝ := {x | |x - 1| > 1}
def N : Set ℝ := {x | x^2 - 3*x ≤ 0}

theorem intersection_of_M_and_N : M ∩ N = {x | 2 < x ∧ x ≤ 3} := by sorry

end intersection_of_M_and_N_l1018_101862


namespace equipment_productivity_increase_l1018_101896

/-- Represents the productivity increase factor of the equipment -/
def productivity_increase : ℝ := 4

/-- Represents the time taken by the first worker to complete the job -/
def first_worker_time : ℝ := 8

/-- Represents the time taken by the second worker to complete the job -/
def second_worker_time : ℝ := 5

/-- Represents the setup time for the second worker -/
def setup_time : ℝ := 2

/-- Represents the time after which the second worker processes as many parts as the first worker -/
def equal_parts_time : ℝ := 1

theorem equipment_productivity_increase :
  (∃ (r : ℝ),
    r > 0 ∧
    r * first_worker_time = productivity_increase * r * (second_worker_time - setup_time) ∧
    r * (setup_time + equal_parts_time) = productivity_increase * r * equal_parts_time) :=
by
  sorry

#check equipment_productivity_increase

end equipment_productivity_increase_l1018_101896


namespace circle_reflection_translation_l1018_101863

/-- Reflects a point about the line y = x -/
def reflect_about_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

/-- Translates a point vertically by a given amount -/
def translate_vertical (p : ℝ × ℝ) (dy : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + dy)

/-- The main theorem -/
theorem circle_reflection_translation (center : ℝ × ℝ) :
  center = (3, -7) →
  (translate_vertical (reflect_about_y_eq_x center) 4) = (-7, 7) := by
  sorry


end circle_reflection_translation_l1018_101863


namespace wall_area_l1018_101825

theorem wall_area (small_tile_area small_tile_proportion total_wall_area : ℝ) 
  (h1 : small_tile_proportion = 1 / 2)
  (h2 : small_tile_area = 80)
  (h3 : small_tile_area = small_tile_proportion * total_wall_area) :
  total_wall_area = 160 := by
  sorry

end wall_area_l1018_101825


namespace sara_added_hundred_pencils_l1018_101892

/-- The number of pencils Sara placed in the drawer -/
def pencils_added (initial final : ℕ) : ℕ := final - initial

/-- Proof that Sara added 100 pencils to the drawer -/
theorem sara_added_hundred_pencils :
  pencils_added 115 215 = 100 := by sorry

end sara_added_hundred_pencils_l1018_101892


namespace linear_term_coefficient_l1018_101804

/-- The coefficient of the linear term in the expansion of (x-1)(1/x + x)^6 is 20 -/
theorem linear_term_coefficient : ℕ :=
  20

#check linear_term_coefficient

end linear_term_coefficient_l1018_101804


namespace cubic_real_root_existence_l1018_101827

theorem cubic_real_root_existence (a₀ a₁ a₂ a₃ : ℝ) (ha₀ : a₀ ≠ 0) :
  ∃ x : ℝ, a₀ * x^3 + a₁ * x^2 + a₂ * x + a₃ = 0 := by
  sorry

end cubic_real_root_existence_l1018_101827


namespace intern_distribution_l1018_101852

/-- The number of ways to distribute n intern teachers to k freshman classes,
    with each class having at least 1 intern -/
def distribution_plans (n k : ℕ) : ℕ :=
  if n ≥ k then (n - k + 1) else 0

/-- Theorem: There are 4 ways to distribute 5 intern teachers to 4 freshman classes,
    with each class having at least 1 intern -/
theorem intern_distribution : distribution_plans 5 4 = 4 := by
  sorry

end intern_distribution_l1018_101852


namespace square_equals_double_product_l1018_101854

theorem square_equals_double_product (a : ℤ) (b : ℝ) : 
  0 ≤ b → b < 1 → a^2 = 2*b*(a + b) → b = 0 ∨ b = (-1 + Real.sqrt 3) / 2 := by
  sorry

end square_equals_double_product_l1018_101854


namespace book_length_calculation_l1018_101855

theorem book_length_calculation (B₁ B₂ : ℕ) : 
  (2 : ℚ) / 3 * B₁ - (1 : ℚ) / 3 * B₁ = 90 →
  (3 : ℚ) / 4 * B₂ - (1 : ℚ) / 4 * B₂ = 120 →
  B₁ + B₂ = 510 := by
sorry

end book_length_calculation_l1018_101855


namespace root_implies_a_values_l1018_101879

theorem root_implies_a_values (a : ℝ) : 
  (2 * (-1)^2 + a * (-1) - a^2 = 0) → (a = 1 ∨ a = -2) := by
  sorry

end root_implies_a_values_l1018_101879


namespace particle_movement_ways_l1018_101878

/-- The number of distinct ways a particle can move on a number line -/
def distinct_ways (total_steps : ℕ) (final_distance : ℕ) : ℕ :=
  Nat.choose total_steps ((total_steps + final_distance) / 2) +
  Nat.choose total_steps ((total_steps - final_distance) / 2)

/-- Theorem stating the number of distinct ways for the given conditions -/
theorem particle_movement_ways :
  distinct_ways 10 4 = 240 := by
  sorry

end particle_movement_ways_l1018_101878


namespace expression_value_l1018_101873

theorem expression_value : 
  (20 - 19 + 18 - 17 + 16 - 15 + 14 - 13 + 12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / 
  (1 - 2 + 3 - 4 + 5 - 6 + 7 - 8 + 9 - 10 + 11 - 12 + 13 - 14 + 15 - 16 + 17 - 18 + 19 - 20) = -1 := by
sorry

end expression_value_l1018_101873


namespace seeds_per_flowerbed_l1018_101835

theorem seeds_per_flowerbed (total_seeds : ℕ) (num_flowerbeds : ℕ) 
  (h1 : total_seeds = 45)
  (h2 : num_flowerbeds = 9)
  (h3 : total_seeds % num_flowerbeds = 0) :
  total_seeds / num_flowerbeds = 5 := by
sorry

end seeds_per_flowerbed_l1018_101835


namespace x_plus_y_value_l1018_101853

theorem x_plus_y_value (x y : ℝ) 
  (h1 : x + Real.cos y = 1005)
  (h2 : x + 1005 * Real.sin y = 1003)
  (h3 : π ≤ y ∧ y ≤ 3 * π / 2) :
  x + y = 1005 + 3 * π / 2 := by
  sorry

end x_plus_y_value_l1018_101853


namespace cone_base_radius_l1018_101847

/-- Given a cone with slant height 5 and lateral area 15π, its base radius is 3 -/
theorem cone_base_radius (s : ℝ) (L : ℝ) (r : ℝ) : 
  s = 5 → L = 15 * Real.pi → L = Real.pi * r * s → r = 3 := by sorry

end cone_base_radius_l1018_101847


namespace local_road_speed_l1018_101894

/-- Proves that the speed of a car on local roads is 20 mph given the specified conditions -/
theorem local_road_speed (local_distance : ℝ) (highway_distance : ℝ) (highway_speed : ℝ) (average_speed : ℝ) :
  local_distance = 60 →
  highway_distance = 120 →
  highway_speed = 60 →
  average_speed = 36 →
  (local_distance + highway_distance) / (local_distance / (local_distance / (local_distance / average_speed - highway_distance / highway_speed)) + highway_distance / highway_speed) = average_speed →
  local_distance / (local_distance / average_speed - highway_distance / highway_speed) = 20 :=
by sorry

end local_road_speed_l1018_101894


namespace jenny_chocolate_squares_count_l1018_101859

/-- The number of chocolate squares Jenny ate -/
def jenny_chocolate_squares (mike_chocolate_squares : ℕ) : ℕ :=
  3 * mike_chocolate_squares + 5

/-- The number of candies Mike's friend ate -/
def mikes_friend_candies (mike_candies : ℕ) : ℕ :=
  mike_candies - 10

/-- The number of candies Jenny ate -/
def jenny_candies (mikes_friend_candies : ℕ) : ℕ :=
  2 * mikes_friend_candies

theorem jenny_chocolate_squares_count 
  (mike_chocolate_squares : ℕ) 
  (mike_candies : ℕ) 
  (h1 : mike_chocolate_squares = 20) 
  (h2 : mike_candies = 20) :
  jenny_chocolate_squares mike_chocolate_squares = 65 := by
  sorry

#check jenny_chocolate_squares_count

end jenny_chocolate_squares_count_l1018_101859


namespace polynomial_simplification_l1018_101871

theorem polynomial_simplification (x : ℝ) :
  (3 * x - 2) * (5 * x^9 + 3 * x^8 + 2 * x^7 + x^6) =
  15 * x^10 - x^9 + 3 * x^7 - 2 * x^6 := by
  sorry

end polynomial_simplification_l1018_101871


namespace charity_event_probability_l1018_101849

theorem charity_event_probability :
  let n : ℕ := 5  -- number of students
  let d : ℕ := 2  -- number of days (Saturday and Sunday)
  let total_outcomes : ℕ := d^n
  let same_day_outcomes : ℕ := 2  -- all choose Saturday or all choose Sunday
  let both_days_outcomes : ℕ := total_outcomes - same_day_outcomes
  (both_days_outcomes : ℚ) / total_outcomes = 15 / 16 :=
by sorry

end charity_event_probability_l1018_101849


namespace part_one_part_two_l1018_101826

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Part 1
theorem part_one (t : Triangle) (h1 : 2 * t.a * Real.sin t.B = Real.sqrt 3 * t.b) 
    (h2 : 0 < t.A ∧ t.A < Real.pi / 2) : t.A = Real.pi / 3 := by
  sorry

-- Part 2
theorem part_two (t : Triangle) (h1 : t.b = 5) (h2 : t.c = Real.sqrt 5) 
    (h3 : Real.cos t.C = 9/10) : t.a = 4 ∨ t.a = 5 := by
  sorry

end part_one_part_two_l1018_101826


namespace adult_books_count_l1018_101860

theorem adult_books_count (total : ℕ) (children_percent : ℚ) (h1 : total = 160) (h2 : children_percent = 35 / 100) :
  (total : ℚ) * (1 - children_percent) = 104 := by
  sorry

end adult_books_count_l1018_101860


namespace backyard_area_l1018_101867

/-- Represents a rectangular backyard with specific walking conditions -/
structure Backyard where
  length : ℝ
  width : ℝ
  length_condition : 25 * length = 1000
  perimeter_condition : 10 * (2 * (length + width)) = 1000

/-- The area of a backyard with the given conditions is 400 square meters -/
theorem backyard_area (b : Backyard) : b.length * b.width = 400 := by
  sorry

end backyard_area_l1018_101867


namespace symmetric_difference_eq_zero_three_l1018_101857

-- Define the function f
def f (n : ℕ) : ℕ := 2 * n + 1

-- Define the sets P and Q
def P : Set ℕ := {1, 2, 3, 4, 5}
def Q : Set ℕ := {3, 4, 5, 6, 7}

-- Define sets A and B
def A : Set ℕ := {n : ℕ | f n ∈ P}
def B : Set ℕ := {n : ℕ | f n ∈ Q}

-- State the theorem
theorem symmetric_difference_eq_zero_three :
  (A ∩ (Set.univ \ B)) ∪ (B ∩ (Set.univ \ A)) = {0, 3} := by sorry

end symmetric_difference_eq_zero_three_l1018_101857


namespace delta_max_success_ratio_l1018_101893

theorem delta_max_success_ratio 
  (charlie_day1_score charlie_day1_total : ℕ)
  (charlie_day2_score charlie_day2_total : ℕ)
  (delta_day1_score delta_day1_total : ℕ)
  (delta_day2_score delta_day2_total : ℕ)
  (h1 : charlie_day1_score = 200)
  (h2 : charlie_day1_total = 360)
  (h3 : charlie_day2_score = 160)
  (h4 : charlie_day2_total = 240)
  (h5 : delta_day1_score > 0)
  (h6 : delta_day2_score > 0)
  (h7 : delta_day1_total + delta_day2_total = 600)
  (h8 : delta_day1_total ≠ 360)
  (h9 : (delta_day1_score : ℚ) / delta_day1_total < (charlie_day1_score : ℚ) / charlie_day1_total)
  (h10 : (delta_day2_score : ℚ) / delta_day2_total < (charlie_day2_score : ℚ) / charlie_day2_total)
  (h11 : (charlie_day1_score + charlie_day2_score : ℚ) / (charlie_day1_total + charlie_day2_total) = 3/5) :
  (delta_day1_score + delta_day2_score : ℚ) / (delta_day1_total + delta_day2_total) ≤ 166/600 :=
by sorry


end delta_max_success_ratio_l1018_101893


namespace samson_activity_solution_l1018_101832

/-- Represents the utility function for Samson's activities -/
def utility (math : ℝ) (frisbee : ℝ) : ℝ := math * frisbee

/-- Represents the total hours spent on activities -/
def totalHours (math : ℝ) (frisbee : ℝ) : ℝ := math + frisbee

theorem samson_activity_solution :
  ∃ (t : ℝ),
    (utility (10 - t) t = utility (t + 5) (4 - t)) ∧
    (totalHours (10 - t) t ≥ 8) ∧
    (totalHours (t + 5) (4 - t) ≥ 8) ∧
    (t ≥ 0) ∧
    (∀ (s : ℝ),
      (utility (10 - s) s = utility (s + 5) (4 - s)) ∧
      (totalHours (10 - s) s ≥ 8) ∧
      (totalHours (s + 5) (4 - s) ≥ 8) ∧
      (s ≥ 0) →
      s = t) ∧
    t = 0 :=
by sorry

end samson_activity_solution_l1018_101832


namespace f_3_range_l1018_101851

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + b * x

-- State the theorem
theorem f_3_range (a b : ℝ) :
  (-1 ≤ f a b 1 ∧ f a b 1 ≤ 2) →
  (1 ≤ f a b 2 ∧ f a b 2 ≤ 3) →
  -3 ≤ f a b 3 ∧ f a b 3 ≤ 12 :=
by sorry

end f_3_range_l1018_101851


namespace max_value_expression_l1018_101823

theorem max_value_expression (a b c d : ℝ) 
  (ha : -4.5 ≤ a ∧ a ≤ 4.5)
  (hb : -4.5 ≤ b ∧ b ≤ 4.5)
  (hc : -4.5 ≤ c ∧ c ≤ 4.5)
  (hd : -4.5 ≤ d ∧ d ≤ 4.5) :
  a + 2*b + c + 2*d - a*b - b*c - c*d - d*a ≤ 90 ∧ 
  ∃ (a' b' c' d' : ℝ), 
    (-4.5 ≤ a' ∧ a' ≤ 4.5) ∧
    (-4.5 ≤ b' ∧ b' ≤ 4.5) ∧
    (-4.5 ≤ c' ∧ c' ≤ 4.5) ∧
    (-4.5 ≤ d' ∧ d' ≤ 4.5) ∧
    a' + 2*b' + c' + 2*d' - a'*b' - b'*c' - c'*d' - d'*a' = 90 := by
  sorry

end max_value_expression_l1018_101823


namespace determinant_calculation_l1018_101802

def determinant (a b c d : Int) : Int :=
  a * d - b * c

theorem determinant_calculation : determinant 2 3 (-6) (-5) = 8 := by
  sorry

end determinant_calculation_l1018_101802


namespace high_school_sample_senior_count_is_160_l1018_101807

theorem high_school_sample (total : ℕ) (junior_percent : ℚ) (not_sophomore_percent : ℚ) 
  (freshman_sophomore_diff : ℕ) : ℕ :=
  let junior_count : ℕ := (junior_percent * total).num.toNat
  let sophomore_count : ℕ := ((1 - not_sophomore_percent) * total).num.toNat
  let freshman_count : ℕ := sophomore_count + freshman_sophomore_diff
  total - (junior_count + sophomore_count + freshman_count)

theorem senior_count_is_160 :
  high_school_sample 800 (27/100) (75/100) 24 = 160 := by
  sorry

end high_school_sample_senior_count_is_160_l1018_101807


namespace lilith_cap_collection_l1018_101813

/-- Calculates the total number of caps Lilith has collected over 5 years -/
def total_caps_collected : ℕ :=
  let caps_first_year := 3 * 12
  let caps_after_first_year := 5 * 12 * 4
  let caps_from_christmas := 40 * 5
  let caps_lost := 15 * 5
  caps_first_year + caps_after_first_year + caps_from_christmas - caps_lost

/-- Theorem stating that the total number of caps Lilith has collected is 401 -/
theorem lilith_cap_collection : total_caps_collected = 401 := by
  sorry

end lilith_cap_collection_l1018_101813


namespace odd_cube_minus_odd_divisible_by_24_l1018_101889

theorem odd_cube_minus_odd_divisible_by_24 (n : ℤ) : 
  ∃ k : ℤ, (2*n + 1)^3 - (2*n + 1) = 24 * k := by
sorry

end odd_cube_minus_odd_divisible_by_24_l1018_101889


namespace half_x_is_32_implies_2x_is_128_l1018_101850

theorem half_x_is_32_implies_2x_is_128 (x : ℝ) (h : x / 2 = 32) : 2 * x = 128 := by
  sorry

end half_x_is_32_implies_2x_is_128_l1018_101850


namespace complex_expression_simplification_l1018_101842

theorem complex_expression_simplification (x : ℝ) :
  x * (x * (x * (3 - x) - 3) + 5) + 1 = -x^4 + 3*x^3 - 3*x^2 + 5*x + 1 := by
  sorry

end complex_expression_simplification_l1018_101842


namespace cos_equality_problem_l1018_101834

theorem cos_equality_problem (m : ℤ) : 
  0 ≤ m ∧ m ≤ 180 → (Real.cos (m * π / 180) = Real.cos (1234 * π / 180) ↔ m = 154) := by
  sorry

end cos_equality_problem_l1018_101834


namespace election_vote_count_l1018_101899

theorem election_vote_count (votes : List Nat) : 
  votes = [195, 142, 116, 90] →
  votes.length = 4 →
  votes[0]! = 195 →
  votes[0]! - votes[1]! = 53 →
  votes[0]! - votes[2]! = 79 →
  votes[0]! - votes[3]! = 105 →
  votes.sum = 543 := by
sorry

end election_vote_count_l1018_101899


namespace min_marbles_needed_l1018_101815

/-- The minimum number of additional marbles needed --/
def min_additional_marbles (n : ℕ) (current : ℕ) : ℕ :=
  (n * (n + 1)) / 2 - current

/-- Theorem stating the minimum number of additional marbles needed --/
theorem min_marbles_needed (n : ℕ) (current : ℕ) 
  (h_n : n = 12) (h_current : current = 40) : 
  min_additional_marbles n current = 38 := by
  sorry

#eval min_additional_marbles 12 40

end min_marbles_needed_l1018_101815


namespace system_solution_l1018_101881

theorem system_solution : ∃ (x y : ℝ), (4 * x - y = 7) ∧ (3 * x + 4 * y = 10) ∧ (x = 2) ∧ (y = 1) := by
  sorry

end system_solution_l1018_101881


namespace least_perimeter_triangle_l1018_101809

/-- 
Given a triangle with two sides of 36 units and 45 units, and the third side being an integer,
the least possible perimeter is 91 units.
-/
theorem least_perimeter_triangle : 
  ∀ (x : ℕ), 
  x > 0 → 
  x + 36 > 45 → 
  x + 45 > 36 → 
  36 + 45 > x → 
  (∀ y : ℕ, y > 0 → y + 36 > 45 → y + 45 > 36 → 36 + 45 > y → x + 36 + 45 ≤ y + 36 + 45) →
  x + 36 + 45 = 91 := by
sorry

end least_perimeter_triangle_l1018_101809


namespace function_equivalence_l1018_101891

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem function_equivalence : 
  (∀ x : ℝ, f (2 * x) = 6 * x - 1) → 
  (∀ x : ℝ, f x = 3 * x - 1) := by
  sorry

end function_equivalence_l1018_101891


namespace wheel_distance_l1018_101812

/-- Proves that a wheel rotating 20 times per minute and moving 35 cm per rotation will travel 420 meters in one hour -/
theorem wheel_distance (rotations_per_minute : ℕ) (distance_per_rotation_cm : ℕ) :
  rotations_per_minute = 20 →
  distance_per_rotation_cm = 35 →
  (rotations_per_minute * 60 * distance_per_rotation_cm : ℚ) / 100 = 420 := by
  sorry

#check wheel_distance

end wheel_distance_l1018_101812


namespace quadratic_rewrite_l1018_101803

theorem quadratic_rewrite (b : ℝ) (h1 : b > 0) :
  (∃ m : ℝ, ∀ x : ℝ, x^2 + b*x + 108 = (x + m)^2 - 4) →
  b = 8 * Real.sqrt 7 := by
sorry

end quadratic_rewrite_l1018_101803


namespace range_of_a_l1018_101836

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x : ℝ | 2*a < x ∧ x < a + 1}

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (¬(B a ⊆ A)) ↔ (1/2 ≤ a) :=
sorry

end range_of_a_l1018_101836


namespace hyperbola_asymptotes_l1018_101824

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1 (a > 0, b > 0) and eccentricity 2,
    prove that the equation of its asymptotes is y = ± √3 x. -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let eccentricity := Real.sqrt ((a^2 + b^2) / a^2)
  eccentricity = 2 →
  ∃ k : ℝ, k = Real.sqrt 3 ∧
    (∀ (x y : ℝ), (x, y) ∈ C → y = k * x ∨ y = -k * x) :=
by sorry

end hyperbola_asymptotes_l1018_101824


namespace yellow_mms_added_l1018_101817

/-- Represents the number of M&Ms of each color in the jar -/
structure MandMs where
  green : ℕ
  red : ℕ
  yellow : ℕ

/-- The initial state of the jar -/
def initial_jar : MandMs :=
  { green := 20, red := 20, yellow := 0 }

/-- The state of the jar after Carter eats 12 green M&Ms -/
def after_carter_eats (jar : MandMs) : MandMs :=
  { jar with green := jar.green - 12 }

/-- The state of the jar after Carter's sister eats half the red M&Ms -/
def after_sister_eats (jar : MandMs) : MandMs :=
  { jar with red := jar.red / 2 }

/-- The final state of the jar after yellow M&Ms are added -/
def final_jar (jar : MandMs) (yellow_added : ℕ) : MandMs :=
  { jar with yellow := jar.yellow + yellow_added }

/-- The probability of picking a green M&M from the jar -/
def prob_green (jar : MandMs) : ℚ :=
  jar.green / (jar.green + jar.red + jar.yellow)

/-- The theorem stating the number of yellow M&Ms added -/
theorem yellow_mms_added : 
  ∃ yellow_added : ℕ,
    let jar1 := after_carter_eats initial_jar
    let jar2 := after_sister_eats jar1
    let jar3 := final_jar jar2 yellow_added
    prob_green jar3 = 1/4 ∧ yellow_added = 14 := by
  sorry

end yellow_mms_added_l1018_101817


namespace quadratic_vertex_x_coordinate_l1018_101846

/-- Given a quadratic function f(x) = ax^2 + bx + c that passes through 
    the points (2, 3), (8, -1), and (11, 8), prove that the x-coordinate 
    of its vertex is 142/23. -/
theorem quadratic_vertex_x_coordinate 
  (f : ℝ → ℝ) 
  (a b c : ℝ) 
  (h1 : ∀ x, f x = a * x^2 + b * x + c) 
  (h2 : f 2 = 3) 
  (h3 : f 8 = -1) 
  (h4 : f 11 = 8) : 
  -b / (2 * a) = 142 / 23 := by
sorry

end quadratic_vertex_x_coordinate_l1018_101846


namespace exam_average_proof_l1018_101895

theorem exam_average_proof :
  let group1_count : ℕ := 15
  let group1_average : ℚ := 75/100
  let group2_count : ℕ := 10
  let group2_average : ℚ := 95/100
  let total_count : ℕ := group1_count + group2_count
  
  (group1_count * group1_average + group2_count * group2_average) / total_count = 83/100 :=
by sorry

end exam_average_proof_l1018_101895


namespace freshmen_psych_liberal_arts_percentage_l1018_101861

/-- Represents the percentage of students that are freshmen -/
def freshman_percentage : ℝ := 80

/-- Represents the percentage of freshmen enrolled in liberal arts -/
def liberal_arts_percentage : ℝ := 60

/-- Represents the percentage of liberal arts freshmen who are psychology majors -/
def psychology_percentage : ℝ := 50

/-- Theorem stating the percentage of students who are freshmen psychology majors in liberal arts -/
theorem freshmen_psych_liberal_arts_percentage :
  (freshman_percentage / 100) * (liberal_arts_percentage / 100) * (psychology_percentage / 100) * 100 = 24 := by
  sorry


end freshmen_psych_liberal_arts_percentage_l1018_101861


namespace sqrt_equation_solution_l1018_101831

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (5 * x + 13) = 15 → x = 212 / 5 := by
  sorry

end sqrt_equation_solution_l1018_101831


namespace rectangular_field_laps_l1018_101870

theorem rectangular_field_laps (length width total_distance : ℝ) 
  (h_length : length = 75)
  (h_width : width = 15)
  (h_total_distance : total_distance = 540) :
  total_distance / (2 * (length + width)) = 3 := by
  sorry

end rectangular_field_laps_l1018_101870


namespace compared_same_type_as_reference_l1018_101840

/-- Two expressions are of the same type if they have the same variables with the same exponents -/
def same_type (e1 e2 : ℕ → ℕ → ℚ) : Prop :=
  ∀ a b, ∃ k : ℚ, e1 a b = k * e2 a b

/-- The reference expression a^2 * b -/
def reference (a b : ℕ) : ℚ := (a^2 : ℚ) * b

/-- The expression to be compared: -2/5 * b * a^2 -/
def compared (a b : ℕ) : ℚ := -(2/5 : ℚ) * b * (a^2 : ℚ)

/-- Theorem stating that the compared expression is of the same type as the reference -/
theorem compared_same_type_as_reference : same_type compared reference := by
  sorry

end compared_same_type_as_reference_l1018_101840
