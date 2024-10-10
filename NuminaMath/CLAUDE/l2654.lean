import Mathlib

namespace prob_exactly_one_red_l2654_265484

structure Box where
  red : ℕ
  black : ℕ

def Box.total (b : Box) : ℕ := b.red + b.black

def prob_red (b : Box) : ℚ := b.red / b.total

def prob_black (b : Box) : ℚ := b.black / b.total

def box_A : Box := ⟨1, 2⟩
def box_B : Box := ⟨2, 2⟩

theorem prob_exactly_one_red : 
  prob_red box_A * prob_black box_B + prob_black box_A * prob_red box_B = 1/2 := by
  sorry

end prob_exactly_one_red_l2654_265484


namespace simplify_polynomial_l2654_265436

theorem simplify_polynomial (x : ℝ) : 
  2*x*(4*x^3 - 3*x + 1) - 7*(x^3 - x^2 + 3*x - 4) = 8*x^4 - 7*x^3 + x^2 - 19*x + 28 := by
  sorry

end simplify_polynomial_l2654_265436


namespace minimum_value_of_f_l2654_265481

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 2)

theorem minimum_value_of_f :
  ∀ x : ℝ, x > 2 → f x ≥ f 3 :=
sorry

end minimum_value_of_f_l2654_265481


namespace race_length_l2654_265485

theorem race_length (L : ℝ) 
  (h1 : L - 70 = L * ((L - 100) / L))  -- A beats B by 70 m
  (h2 : L - 163 = (L - 100) * ((L - 163) / (L - 100)))  -- B beats C by 100 m
  (h3 : L - 163 = L * ((L - 163) / L))  -- A beats C by 163 m
  : L = 1000 := by
  sorry

end race_length_l2654_265485


namespace sum_of_binary_digits_315_l2654_265412

/-- The sum of the digits in the binary representation of 315 is 6. -/
theorem sum_of_binary_digits_315 : 
  (Nat.digits 2 315).sum = 6 := by sorry

end sum_of_binary_digits_315_l2654_265412


namespace area_trapezoid_EFBA_l2654_265495

/-- Rectangle ABCD with points E and F on side DC -/
structure RectangleWithPoints where
  /-- Length of side AB -/
  AB : ℝ
  /-- Length of side BC -/
  BC : ℝ
  /-- Length of segment DE -/
  DE : ℝ
  /-- Length of segment FC -/
  FC : ℝ
  /-- Area of rectangle ABCD -/
  area_ABCD : ℝ
  /-- AB is positive -/
  AB_pos : AB > 0
  /-- BC is positive -/
  BC_pos : BC > 0
  /-- DE is positive -/
  DE_pos : DE > 0
  /-- FC is positive -/
  FC_pos : FC > 0
  /-- Area of ABCD is product of AB and BC -/
  area_eq : area_ABCD = AB * BC
  /-- DE + EF + FC = DC = AB -/
  side_sum : DE + (AB - DE - FC) + FC = AB

/-- The area of trapezoid EFBA is 14 square units -/
theorem area_trapezoid_EFBA (r : RectangleWithPoints) (h1 : r.AB = 10) (h2 : r.BC = 2) 
    (h3 : r.DE = 2) (h4 : r.FC = 4) (h5 : r.area_ABCD = 20) : 
    r.AB * r.BC - r.DE * r.BC / 2 - r.FC * r.BC / 2 = 14 := by
  sorry

end area_trapezoid_EFBA_l2654_265495


namespace stu_has_four_books_l2654_265431

/-- Given the number of books for Elmo, Laura, and Stu, we define their relationships --/
def book_relation (elmo laura stu : ℕ) : Prop :=
  elmo = 3 * laura ∧ laura = 2 * stu ∧ elmo = 24

/-- Theorem stating that if the book relation holds, then Stu has 4 books --/
theorem stu_has_four_books (elmo laura stu : ℕ) :
  book_relation elmo laura stu → stu = 4 := by
  sorry

end stu_has_four_books_l2654_265431


namespace project_completion_time_l2654_265452

/-- The number of days person A takes to complete the project alone -/
def days_A : ℝ := 45

/-- The number of days person B takes to complete the project alone -/
def days_B : ℝ := 30

/-- The number of days person B works alone initially -/
def initial_days_B : ℝ := 22

/-- The total number of days to complete the project -/
def total_days : ℝ := 34

theorem project_completion_time :
  (total_days - initial_days_B) / days_A + initial_days_B / days_B = 1 := by sorry

end project_completion_time_l2654_265452


namespace transformed_area_theorem_l2654_265490

-- Define the transformation matrix
def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 8, -2]

-- Define the original region's area
def original_area : ℝ := 15

-- Theorem statement
theorem transformed_area_theorem :
  let transformed_area := original_area * abs (Matrix.det A)
  transformed_area = 570 := by sorry

end transformed_area_theorem_l2654_265490


namespace max_quarters_is_19_l2654_265419

/-- Represents the number of each coin type in the piggy bank -/
structure CoinCount where
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- Checks if the given coin count satisfies the problem conditions -/
def isValidCoinCount (c : CoinCount) : Prop :=
  c.nickels > 0 ∧ c.dimes > 0 ∧ c.quarters > 0 ∧
  c.nickels + c.dimes + c.quarters = 120 ∧
  5 * c.nickels + 10 * c.dimes + 25 * c.quarters = 1000

/-- Theorem stating that 19 is the maximum number of quarters possible -/
theorem max_quarters_is_19 :
  ∀ c : CoinCount, isValidCoinCount c → c.quarters ≤ 19 :=
by sorry

end max_quarters_is_19_l2654_265419


namespace monomial_same_type_l2654_265441

/-- A structure representing a monomial with coefficients in ℤ -/
structure Monomial :=
  (coeff : ℤ)
  (m_exp : ℕ)
  (n_exp : ℕ)

/-- Two monomials are of the same type if they have the same variables with the same exponents -/
def same_type (a b : Monomial) : Prop :=
  a.m_exp = b.m_exp ∧ a.n_exp = b.n_exp

/-- The monomial -2mn^2 -/
def monomial1 : Monomial :=
  { coeff := -2, m_exp := 1, n_exp := 2 }

/-- The monomial mn^2 -/
def monomial2 : Monomial :=
  { coeff := 1, m_exp := 1, n_exp := 2 }

theorem monomial_same_type : same_type monomial1 monomial2 := by
  sorry

end monomial_same_type_l2654_265441


namespace different_color_pairs_count_l2654_265459

/- Given a drawer with distinguishable socks: -/
def white_socks : ℕ := 6
def brown_socks : ℕ := 5
def blue_socks : ℕ := 4

/- Define the function to calculate the number of ways to choose two socks of different colors -/
def different_color_pairs : ℕ :=
  white_socks * brown_socks +
  brown_socks * blue_socks +
  white_socks * blue_socks

/- The theorem to prove -/
theorem different_color_pairs_count : different_color_pairs = 74 := by
  sorry

end different_color_pairs_count_l2654_265459


namespace male_attendees_on_time_l2654_265426

/-- Proves that the fraction of male attendees who arrived on time is 0.875 -/
theorem male_attendees_on_time (total_attendees : ℝ) : 
  let male_attendees := (3/5 : ℝ) * total_attendees
  let female_attendees := (2/5 : ℝ) * total_attendees
  let on_time_female := (9/10 : ℝ) * female_attendees
  let not_on_time := 0.115 * total_attendees
  let on_time := total_attendees - not_on_time
  ∃ (on_time_male : ℝ), 
    on_time_male + on_time_female = on_time ∧ 
    on_time_male / male_attendees = 0.875 :=
by sorry

end male_attendees_on_time_l2654_265426


namespace fish_tank_problem_l2654_265498

/-- Given 3 fish tanks with a total of 100 fish, where two tanks have twice as many fish as the first tank, prove that the first tank contains 20 fish. -/
theorem fish_tank_problem (first_tank : ℕ) : 
  first_tank + 2 * first_tank + 2 * first_tank = 100 → first_tank = 20 := by
  sorry

end fish_tank_problem_l2654_265498


namespace product_of_distinct_nonzero_reals_l2654_265418

theorem product_of_distinct_nonzero_reals (x y : ℝ) : 
  x ≠ 0 → y ≠ 0 → x ≠ y → x + 3 / x = y + 3 / y → x * y = 3 := by
  sorry

end product_of_distinct_nonzero_reals_l2654_265418


namespace equation_solution_l2654_265474

theorem equation_solution :
  ∃ x : ℝ, (x^2 + 3*x + 2) / (x^2 + 1) = x - 2 ∧ x = 4 := by
sorry

end equation_solution_l2654_265474


namespace square_sum_ge_product_sum_l2654_265425

theorem square_sum_ge_product_sum (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + b*c + a*c := by
  sorry

end square_sum_ge_product_sum_l2654_265425


namespace tv_purchase_hours_l2654_265444

/-- The number of additional hours needed to buy a TV given the TV cost, hourly wage, and weekly work hours. -/
def additional_hours_needed (tv_cost : ℕ) (hourly_wage : ℕ) (weekly_work_hours : ℕ) : ℕ :=
  let monthly_work_hours := weekly_work_hours * 4
  let monthly_earnings := monthly_work_hours * hourly_wage
  let additional_amount_needed := tv_cost - monthly_earnings
  additional_amount_needed / hourly_wage

/-- Theorem stating that given a TV cost of $1700, an hourly wage of $10, and a 30-hour workweek, 
    the additional hours needed to buy the TV is 50. -/
theorem tv_purchase_hours : additional_hours_needed 1700 10 30 = 50 := by
  sorry

end tv_purchase_hours_l2654_265444


namespace cone_base_circumference_l2654_265405

/-- The circumference of the base of a right circular cone formed by removing a 180° sector from a circle with radius 6 inches is equal to 6π. -/
theorem cone_base_circumference (r : ℝ) (h : r = 6) :
  let original_circumference := 2 * π * r
  let removed_sector_angle := π  -- 180° in radians
  let full_circle_angle := 2 * π  -- 360° in radians
  let base_circumference := (removed_sector_angle / full_circle_angle) * original_circumference
  base_circumference = 6 * π :=
by sorry

end cone_base_circumference_l2654_265405


namespace find_number_l2654_265445

theorem find_number (a b : ℕ+) (hcf : Nat.gcd a b = 12) (lcm : Nat.lcm a b = 396) (hb : b = 198) : a = 24 := by
  sorry

end find_number_l2654_265445


namespace quadratic_inequality_solution_l2654_265479

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 - b*x + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → 
  a + b = -10 := by
sorry

end quadratic_inequality_solution_l2654_265479


namespace allen_blocks_count_l2654_265451

/-- The number of blocks per color -/
def blocks_per_color : ℕ := 7

/-- The number of colors used -/
def colors_used : ℕ := 7

/-- The total number of blocks -/
def total_blocks : ℕ := blocks_per_color * colors_used

theorem allen_blocks_count : total_blocks = 49 := by
  sorry

end allen_blocks_count_l2654_265451


namespace sum_of_squared_distances_bounded_l2654_265440

/-- A point on the perimeter of a unit square -/
structure PerimeterPoint where
  x : Real
  y : Real
  on_perimeter : (x = 0 ∨ x = 1 ∨ y = 0 ∨ y = 1) ∧ 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1

/-- Four points on the perimeter of a unit square, in order -/
structure FourPoints where
  A : PerimeterPoint
  B : PerimeterPoint
  C : PerimeterPoint
  D : PerimeterPoint
  in_order : (A.x ≤ B.x ∧ A.y ≥ B.y) ∧ (B.x ≤ C.x ∧ B.y ≤ C.y) ∧ (C.x ≥ D.x ∧ C.y ≤ D.y) ∧ (D.x ≤ A.x ∧ D.y ≤ A.y)
  each_side_has_point : (A.x = 0 ∨ B.x = 0 ∨ C.x = 0 ∨ D.x = 0) ∧
                        (A.x = 1 ∨ B.x = 1 ∨ C.x = 1 ∨ D.x = 1) ∧
                        (A.y = 0 ∨ B.y = 0 ∨ C.y = 0 ∨ D.y = 0) ∧
                        (A.y = 1 ∨ B.y = 1 ∨ C.y = 1 ∨ D.y = 1)

/-- The squared distance between two points -/
def squared_distance (p1 p2 : PerimeterPoint) : Real :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- The theorem to be proved -/
theorem sum_of_squared_distances_bounded (points : FourPoints) :
  2 ≤ squared_distance points.A points.B +
      squared_distance points.B points.C +
      squared_distance points.C points.D +
      squared_distance points.D points.A
  ∧
  squared_distance points.A points.B +
  squared_distance points.B points.C +
  squared_distance points.C points.D +
  squared_distance points.D points.A ≤ 4 := by
  sorry

end sum_of_squared_distances_bounded_l2654_265440


namespace carly_job_applications_l2654_265491

/-- The number of job applications Carly sent to her home state -/
def home_state_apps : ℕ := 200

/-- The number of job applications Carly sent to the neighboring state -/
def neighboring_state_apps : ℕ := 2 * home_state_apps

/-- The number of job applications Carly sent to each of the other 3 states -/
def other_state_apps : ℕ := neighboring_state_apps - 50

/-- The number of other states Carly sent applications to -/
def num_other_states : ℕ := 3

/-- The total number of job applications Carly sent -/
def total_applications : ℕ := home_state_apps + neighboring_state_apps + (num_other_states * other_state_apps)

theorem carly_job_applications : total_applications = 1650 := by
  sorry

end carly_job_applications_l2654_265491


namespace abs_sum_inequality_range_l2654_265406

theorem abs_sum_inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x + 3| + |x - 1| ≥ a^2 - 3*a) ↔ a ∈ Set.Icc (-1) 4 :=
sorry

end abs_sum_inequality_range_l2654_265406


namespace intersection_line_hyperbola_l2654_265471

theorem intersection_line_hyperbola (a : ℝ) :
  (∃ A B : ℝ × ℝ, 
    (A.2 = a * A.1 + 1 ∧ 3 * A.1^2 - A.2^2 = 1) ∧
    (B.2 = a * B.1 + 1 ∧ 3 * B.1^2 - B.2^2 = 1) ∧
    A ≠ B) →
  (∃ A B : ℝ × ℝ, 
    (A.2 = a * A.1 + 1 ∧ 3 * A.1^2 - A.2^2 = 1) ∧
    (B.2 = a * B.1 + 1 ∧ 3 * B.1^2 - B.2^2 = 1) ∧
    A ≠ B ∧
    A.1 * B.1 + A.2 * B.2 = 0) →
  a = 1 ∨ a = -1 := by
sorry

end intersection_line_hyperbola_l2654_265471


namespace soldiers_on_great_wall_count_l2654_265494

/-- The number of soldiers in beacon towers along the Great Wall --/
def soldiers_on_great_wall (wall_length : ℕ) (tower_interval : ℕ) (soldiers_per_tower : ℕ) : ℕ :=
  (wall_length / tower_interval) * soldiers_per_tower

/-- Theorem stating the number of soldiers on the Great Wall --/
theorem soldiers_on_great_wall_count :
  soldiers_on_great_wall 7300 5 2 = 2920 := by
  sorry

end soldiers_on_great_wall_count_l2654_265494


namespace count_eight_digit_numbers_product_7000_l2654_265437

/-- The number of eight-digit numbers whose digits multiply to 7000 -/
def eight_digit_numbers_with_product_7000 : ℕ := 5600

/-- The prime factorization of 7000 -/
def prime_factorization_7000 : List ℕ := [7, 2, 2, 2, 5, 5, 5]

theorem count_eight_digit_numbers_product_7000 :
  eight_digit_numbers_with_product_7000 = 5600 := by
  sorry

end count_eight_digit_numbers_product_7000_l2654_265437


namespace adjacent_pairs_after_10_minutes_l2654_265477

/-- Represents the number of adjacent pairs of the same letter after n minutes -/
def a (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | n+1 => 2^n - 1 - a n

/-- The transformation rule applied for n minutes -/
def transform (n : ℕ) : String :=
  match n with
  | 0 => "A"
  | n+1 => String.replace (transform n) "A" "AB" |>.replace "B" "BA"

theorem adjacent_pairs_after_10_minutes :
  (transform 10).length = 1024 ∧ a 10 = 341 := by
  sorry

#eval a 10  -- Should output 341

end adjacent_pairs_after_10_minutes_l2654_265477


namespace bacteria_growth_proof_l2654_265482

/-- The number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- The doubling time of the bacteria population in seconds -/
def doubling_time : ℕ := 30

/-- The total time of the experiment in minutes -/
def total_time : ℕ := 4

/-- The final number of bacteria after the experiment -/
def final_bacteria_count : ℕ := 524288

/-- The initial number of bacteria -/
def initial_bacteria_count : ℕ := 2048

theorem bacteria_growth_proof :
  initial_bacteria_count * 2^(total_time * seconds_per_minute / doubling_time) = final_bacteria_count :=
sorry

end bacteria_growth_proof_l2654_265482


namespace triangle_angle_constraint_l2654_265403

/-- 
Given a triangle ABC with the conditions:
1) 5 * sin(A) + 2 * cos(B) = 5
2) 2 * sin(B) + 5 * cos(A) = 2

This theorem states that either:
a) The triangle is degenerate with angle C = 180°, or
b) There is no solution for a non-degenerate triangle.
-/
theorem triangle_angle_constraint (A B C : ℝ) : 
  (5 * Real.sin A + 2 * Real.cos B = 5) →
  (2 * Real.sin B + 5 * Real.cos A = 2) →
  (A + B + C = Real.pi) →
  ((C = Real.pi ∧ (A = 0 ∨ B = 0)) ∨ 
   ∀ A B C, ¬(A > 0 ∧ B > 0 ∧ C > 0)) := by
  sorry


end triangle_angle_constraint_l2654_265403


namespace equation_solutions_l2654_265443

theorem equation_solutions : 
  let equation := fun x : ℝ => x^2 * (x + 1)^2 + x^2 - 3 * (x + 1)^2
  ∀ x : ℝ, equation x = 0 ↔ x = (1 + Real.sqrt 5) / 2 ∨ x = (1 - Real.sqrt 5) / 2 := by
  sorry

end equation_solutions_l2654_265443


namespace paul_final_stock_l2654_265427

def pencils_per_day : ℕ := 100
def work_days_per_week : ℕ := 5
def initial_stock : ℕ := 80
def pencils_sold : ℕ := 350

def final_stock : ℕ := initial_stock + (pencils_per_day * work_days_per_week) - pencils_sold

theorem paul_final_stock : final_stock = 230 := by sorry

end paul_final_stock_l2654_265427


namespace min_steps_to_remove_zeros_l2654_265467

/-- Represents the state of the blackboard -/
structure BoardState where
  zeros : Nat
  ones : Nat

/-- Defines a step operation on the board state -/
def step (s : BoardState) : BoardState :=
  { zeros := s.zeros - 1, ones := s.ones + 1 }

/-- Theorem: Minimum steps to remove all zeroes -/
theorem min_steps_to_remove_zeros (initial : BoardState) 
  (h1 : initial.zeros = 150) 
  (h2 : initial.ones = 151) : 
  ∃ (n : Nat), n = 150 ∧ (step^[n] initial).zeros = 0 :=
sorry

end min_steps_to_remove_zeros_l2654_265467


namespace extremum_values_l2654_265413

def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

theorem extremum_values (a b : ℝ) :
  (∃ (ε : ℝ), ∀ (x : ℝ), |x - 1| < ε → f a b x ≤ f a b 1) ∧
  (∃ (δ : ℝ), ∀ (x : ℝ), |x - 1| < δ → f a b x ≥ f a b 1) ∧
  f a b 1 = 10 →
  a = 4 ∧ b = -11 := by sorry

end extremum_values_l2654_265413


namespace f_monotonicity_and_extreme_l2654_265409

open Real

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (-x)

theorem f_monotonicity_and_extreme :
  (∀ x y, x < y ∧ y < 1 → f x < f y) ∧
  (∀ x y, 1 < x ∧ x < y → f y < f x) ∧
  (∀ x, f x ≤ f 1) ∧
  (f 1 = 1 / Real.exp 1) := by
sorry

end f_monotonicity_and_extreme_l2654_265409


namespace sum_smallest_largest_primes_l2654_265468

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def primes_between (a b : ℕ) : Set ℕ :=
  {n : ℕ | a < n ∧ n < b ∧ is_prime n}

theorem sum_smallest_largest_primes :
  let P := primes_between 50 100
  ∃ (p q : ℕ), p ∈ P ∧ q ∈ P ∧
    (∀ x ∈ P, p ≤ x) ∧
    (∀ x ∈ P, x ≤ q) ∧
    p + q = 150 :=
sorry

end sum_smallest_largest_primes_l2654_265468


namespace total_counts_for_week_l2654_265442

/-- Represents the number of times Carla counts each item on a given day -/
structure DailyCounts where
  tiles : Nat
  books : Nat
  chairs : Nat

/-- The week's counting activities -/
def week : List DailyCounts := [
  ⟨1, 1, 0⟩,  -- Monday
  ⟨2, 3, 0⟩,  -- Tuesday
  ⟨0, 0, 4⟩,  -- Wednesday
  ⟨3, 0, 2⟩,  -- Thursday
  ⟨1, 2, 3⟩   -- Friday
]

/-- Calculates the total number of counts for a day -/
def totalCountsForDay (day : DailyCounts) : Nat :=
  day.tiles + day.books + day.chairs

/-- Theorem stating that the total number of counts for the week is 22 -/
theorem total_counts_for_week : (week.map totalCountsForDay).sum = 22 := by
  sorry

end total_counts_for_week_l2654_265442


namespace total_time_to_school_l2654_265420

def time_to_gate : ℕ := 15
def time_gate_to_building : ℕ := 6
def time_building_to_room : ℕ := 9

theorem total_time_to_school :
  time_to_gate + time_gate_to_building + time_building_to_room = 30 := by
  sorry

end total_time_to_school_l2654_265420


namespace no_solution_exists_l2654_265463

/-- P(n) denotes the greatest prime factor of n -/
def greatest_prime_factor (n : ℕ) : ℕ := sorry

/-- Theorem: There are no positive integers n > 1 such that both 
    P(n) = √n and P(n+36) = √(n+36) -/
theorem no_solution_exists : ¬ ∃ (n : ℕ), n > 1 ∧ 
  (greatest_prime_factor n = Nat.sqrt n) ∧ 
  (greatest_prime_factor (n + 36) = Nat.sqrt (n + 36)) := by
  sorry

end no_solution_exists_l2654_265463


namespace dividing_line_halves_area_l2654_265487

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The T-shaped region -/
def TRegion : Set Point := {p | 
  (0 ≤ p.x ∧ p.x ≤ 4 ∧ 0 ≤ p.y ∧ p.y ≤ 4) ∨
  (4 < p.x ∧ p.x ≤ 7 ∧ 0 ≤ p.y ∧ p.y ≤ 2)
}

/-- The line y = (1/2)x -/
def DividingLine (p : Point) : Prop :=
  p.y = (1/2) * p.x

/-- The area of a region -/
noncomputable def area (s : Set Point) : ℝ := sorry

/-- The part of the region above the line -/
def UpperRegion : Set Point :=
  {p ∈ TRegion | p.y > (1/2) * p.x}

/-- The part of the region below the line -/
def LowerRegion : Set Point :=
  {p ∈ TRegion | p.y < (1/2) * p.x}

/-- The theorem stating that the line y = (1/2)x divides the T-shaped region in half -/
theorem dividing_line_halves_area : 
  area UpperRegion = area LowerRegion := by sorry

end dividing_line_halves_area_l2654_265487


namespace solution_product_l2654_265466

theorem solution_product (p q : ℝ) : 
  (p - 7) * (2 * p + 10) = p^2 - 13 * p + 36 →
  (q - 7) * (2 * q + 10) = q^2 - 13 * q + 36 →
  p ≠ q →
  (p - 2) * (q - 2) = -84 := by
  sorry

end solution_product_l2654_265466


namespace circle_equation_l2654_265407

/-- Given a circle with center (0, -2) and a chord intercepted by the line 2x - y + 3 = 0
    with length 4√5, prove that the equation of the circle is x² + (y+2)² = 25. -/
theorem circle_equation (x y : ℝ) :
  let center : ℝ × ℝ := (0, -2)
  let chord_line (x y : ℝ) := 2 * x - y + 3 = 0
  let chord_length : ℝ := 4 * Real.sqrt 5
  ∃ (r : ℝ), r > 0 ∧
    (∀ (p : ℝ × ℝ), (p.1 - center.1)^2 + (p.2 - center.2)^2 = r^2 ↔
      x^2 + (y + 2)^2 = 25) :=
by
  sorry


end circle_equation_l2654_265407


namespace batsman_average_l2654_265433

theorem batsman_average (previous_total : ℕ) (previous_average : ℚ) : 
  previous_total = (16 : ℕ) * previous_average ∧ 
  (previous_total + 56) / 17 = previous_average + 3 →
  (previous_total + 56) / 17 = 8 := by
sorry

end batsman_average_l2654_265433


namespace ellipse_m_value_l2654_265472

/-- An ellipse with equation x²/(10-m) + y²/(m-2) = 1, major axis on x-axis, and focal distance 4 -/
structure Ellipse (m : ℝ) :=
  (eq : ∀ x y : ℝ, x^2 / (10 - m) + y^2 / (m - 2) = 1)
  (major_axis_x : ℝ → ℝ)
  (focal_distance : ℝ)
  (h_focal_distance : focal_distance = 4)

/-- The value of m for the given ellipse is 4 -/
theorem ellipse_m_value (m : ℝ) (e : Ellipse m) : m = 4 := by
  sorry

end ellipse_m_value_l2654_265472


namespace speed_in_still_water_l2654_265493

def upstream_speed : ℝ := 25
def downstream_speed : ℝ := 31

theorem speed_in_still_water :
  (upstream_speed + downstream_speed) / 2 = 28 := by
  sorry

end speed_in_still_water_l2654_265493


namespace blender_price_difference_l2654_265414

def in_store_price : ℚ := 75.99
def tv_payment : ℚ := 17.99
def shipping_fee : ℚ := 6.50
def handling_charge : ℚ := 2.50

theorem blender_price_difference :
  (4 * tv_payment + shipping_fee + handling_charge - in_store_price) * 100 = 497 := by
  sorry

end blender_price_difference_l2654_265414


namespace thirteen_y_minus_x_equals_one_l2654_265461

theorem thirteen_y_minus_x_equals_one (x y : ℤ) 
  (h1 : x > 0) 
  (h2 : x = 11 * y + 4) 
  (h3 : 2 * x = 8 * (3 * y) + 3) : 
  13 * y - x = 1 := by
  sorry

end thirteen_y_minus_x_equals_one_l2654_265461


namespace assignment_methods_count_l2654_265447

def number_of_departments : ℕ := 5
def number_of_graduates : ℕ := 4
def departments_to_fill : ℕ := 3

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))
def permute (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

def assignment_methods : ℕ := 
  (choose number_of_departments departments_to_fill) * 
  (choose number_of_graduates 2) * 
  (permute departments_to_fill departments_to_fill)

theorem assignment_methods_count : assignment_methods = 360 := by
  sorry

end assignment_methods_count_l2654_265447


namespace fourth_term_of_sequence_l2654_265486

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r ^ (n - 1)

theorem fourth_term_of_sequence (y : ℝ) :
  let a₁ := 8
  let a₂ := 32 * y^2
  let a₃ := 128 * y^4
  let r := a₂ / a₁
  geometric_sequence a₁ r 4 = 512 * y^6 := by
sorry

end fourth_term_of_sequence_l2654_265486


namespace committee_selection_l2654_265478

theorem committee_selection (n m : ℕ) (hn : n = 20) (hm : m = 3) :
  Nat.choose n m = 1140 := by
  sorry

end committee_selection_l2654_265478


namespace circle_properties_l2654_265480

-- Define the circle C in polar coordinates
def C (ρ θ : ℝ) : Prop := ρ^2 = 4*ρ*(Real.cos θ + Real.sin θ) - 6

-- Define the circle C in rectangular coordinates
def C_rect (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 2

-- Theorem statement
theorem circle_properties :
  -- 1. Equivalence of polar and rectangular equations
  (∀ x y : ℝ, C_rect x y ↔ ∃ ρ θ : ℝ, C ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ∧
  -- 2. Maximum value of x + y is 6
  (∀ x y : ℝ, C_rect x y → x + y ≤ 6) ∧
  -- 3. (3, 3) is on C and achieves the maximum
  C_rect 3 3 ∧ 3 + 3 = 6 :=
sorry

end circle_properties_l2654_265480


namespace polynomial_division_remainder_l2654_265458

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  X^3 + 3*X^2 = (X^2 + 4*X + 2) * q + (-X^2 - 2*X) := by
  sorry

end polynomial_division_remainder_l2654_265458


namespace geometric_sequence_solution_l2654_265476

theorem geometric_sequence_solution (x : ℝ) :
  (1 < x ∧ x < 9 ∧ x^2 = 9) ↔ (x = 3 ∨ x = -3) := by sorry

end geometric_sequence_solution_l2654_265476


namespace m_range_l2654_265432

def p (x : ℝ) : Prop := |1 - (x - 2)/3| ≤ 2

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

theorem m_range (m : ℝ) :
  (m > 0) →
  (∀ x, q x m → ¬(p x)) →
  (∃ x, q x m ∧ ¬(p x)) →
  m ≥ 10 :=
sorry

end m_range_l2654_265432


namespace croissants_leftover_l2654_265408

theorem croissants_leftover (total : Nat) (neighbors : Nat) (h1 : total = 59) (h2 : neighbors = 8) :
  total % neighbors = 3 := by
  sorry

end croissants_leftover_l2654_265408


namespace poncelet_theorem_l2654_265430

-- Define a circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a triangle type
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define an incircle
def incircle (t : Triangle) : Circle := sorry

-- Function to check if a point lies on a circle
def lies_on_circle (p : ℝ × ℝ) (c : Circle) : Prop := sorry

-- Theorem statement
theorem poncelet_theorem 
  (ABC DEF : Triangle) 
  (common_incircle : incircle ABC = incircle DEF)
  (c : Circle)
  (A_on_c : lies_on_circle ABC.A c)
  (B_on_c : lies_on_circle ABC.B c)
  (C_on_c : lies_on_circle ABC.C c)
  (D_on_c : lies_on_circle DEF.A c)
  (E_on_c : lies_on_circle DEF.B c) :
  lies_on_circle DEF.C c := by
  sorry


end poncelet_theorem_l2654_265430


namespace courses_last_year_is_six_l2654_265470

-- Define the number of courses taken last year
def courses_last_year : ℕ := 6

-- Define the average grade last year
def avg_grade_last_year : ℝ := 100

-- Define the number of courses taken the year before
def courses_year_before : ℕ := 5

-- Define the average grade for the year before
def avg_grade_year_before : ℝ := 50

-- Define the average grade for the entire two-year period
def avg_grade_two_years : ℝ := 77

-- Theorem statement
theorem courses_last_year_is_six :
  ((courses_year_before * avg_grade_year_before + courses_last_year * avg_grade_last_year) / 
   (courses_year_before + courses_last_year : ℝ) = avg_grade_two_years) ∧
  (courses_last_year = 6) :=
sorry

end courses_last_year_is_six_l2654_265470


namespace sixtieth_pair_l2654_265449

/-- Definition of the sequence of integer pairs -/
def sequence_pair : ℕ → ℕ × ℕ
| 0 => (1, 1)
| n + 1 => 
  let (a, b) := sequence_pair n
  if a = 1 then (b + 1, 1) else (a - 1, b + 1)

/-- The 60th pair in the sequence is (5, 7) -/
theorem sixtieth_pair : sequence_pair 59 = (5, 7) := by
  sorry

end sixtieth_pair_l2654_265449


namespace complex_number_in_second_quadrant_l2654_265453

def is_in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem complex_number_in_second_quadrant :
  let z : ℂ := Complex.mk (-2) 1
  is_in_second_quadrant z := by
  sorry

end complex_number_in_second_quadrant_l2654_265453


namespace half_angle_quadrant_l2654_265428

def is_second_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi + Real.pi / 2 < α ∧ α < 2 * k * Real.pi + Real.pi

def is_first_or_third_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, k * Real.pi < α ∧ α < k * Real.pi + Real.pi / 2

theorem half_angle_quadrant (α : Real) :
  is_second_quadrant α → is_first_or_third_quadrant (α / 2) :=
by sorry

end half_angle_quadrant_l2654_265428


namespace intersection_cardinality_l2654_265429

def M : Finset ℕ := {1, 2, 4, 6, 8}
def N : Finset ℕ := {1, 2, 3, 5, 6, 7}

theorem intersection_cardinality : Finset.card (M ∩ N) = 3 := by
  sorry

end intersection_cardinality_l2654_265429


namespace line_curve_properties_l2654_265424

/-- Line passing through a point with a given direction vector -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Curve defined by an equation -/
def Curve := (ℝ × ℝ) → Prop

def line_l : Line := { point := (1, 0), direction := (2, -1) }

def curve_C : Curve := fun (x, y) ↦ x^2 + y^2 - 2*x - 4*y - 4 = 0

/-- Distance from a point to a line -/
def distance_point_to_line (p : ℝ × ℝ) (l : Line) : ℝ := sorry

/-- Check if a line intersects a curve -/
def intersects (l : Line) (c : Curve) : Prop := sorry

/-- Length of the chord formed by the intersection of a line and a curve -/
def chord_length (l : Line) (c : Curve) : ℝ := sorry

theorem line_curve_properties :
  let origin := (0, 0)
  distance_point_to_line origin line_l = 1 / Real.sqrt 5 ∧
  intersects line_l curve_C ∧
  chord_length line_l curve_C = 2 * Real.sqrt 145 / 5 := by sorry

end line_curve_properties_l2654_265424


namespace expression_evaluation_l2654_265460

theorem expression_evaluation : 12 - 5 * 3^2 + 8 / 2 - 7 + 4^2 = -20 := by
  sorry

end expression_evaluation_l2654_265460


namespace quadratic_complete_square_l2654_265462

theorem quadratic_complete_square (x : ℝ) : ∃ (p q : ℝ), 
  (4 * x^2 + 8 * x - 448 = 0) ↔ ((x + p)^2 = q) ∧ q = 113 :=
by sorry

end quadratic_complete_square_l2654_265462


namespace rectangle_length_calculation_l2654_265450

theorem rectangle_length_calculation (rectangle_width square_width area_difference : ℝ) : 
  rectangle_width = 6 →
  square_width = 5 →
  rectangle_width * (32 / rectangle_width) - square_width * square_width = area_difference →
  area_difference = 7 →
  32 / rectangle_width = 32 / 6 :=
by
  sorry

end rectangle_length_calculation_l2654_265450


namespace max_a_cubic_function_l2654_265416

/-- Given a cubic function f(x) = ax^3 + bx^2 + cx + d with a ≠ 0,
    and |f'(x)| ≤ 1 for 0 ≤ x ≤ 1, the maximum value of a is 8/3. -/
theorem max_a_cubic_function (a b c d : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, 0 ≤ x → x ≤ 1 → |3 * a * x^2 + 2 * b * x + c| ≤ 1) →
  a ≤ 8/3 :=
by sorry

end max_a_cubic_function_l2654_265416


namespace range_of_a_l2654_265489

def A (a : ℝ) : Set ℝ := {x | -2 - a < x ∧ x < a}

theorem range_of_a (a : ℝ) :
  (a > 0) →
  ((1 ∈ A a) ∨ (2 ∈ A a)) ∧
  ¬((1 ∈ A a) ∧ (2 ∈ A a)) →
  1 < a ∧ a ≤ 2 :=
by
  sorry

#check range_of_a

end range_of_a_l2654_265489


namespace solve_system_l2654_265448

theorem solve_system (s t : ℤ) (eq1 : 11 * s + 7 * t = 160) (eq2 : s = 2 * t + 4) : t = 4 := by
  sorry

end solve_system_l2654_265448


namespace brothers_age_ratio_l2654_265454

theorem brothers_age_ratio :
  ∀ (rick_age oldest_age middle_age smallest_age youngest_age : ℕ),
    rick_age = 15 →
    oldest_age = 2 * rick_age →
    middle_age = oldest_age / 3 →
    youngest_age = 3 →
    smallest_age = youngest_age + 2 →
    (smallest_age : ℚ) / (middle_age : ℚ) = 1 / 2 := by
  sorry

end brothers_age_ratio_l2654_265454


namespace similar_right_triangle_longest_side_l2654_265410

theorem similar_right_triangle_longest_side
  (a b c : ℝ)
  (h_right : a^2 + b^2 = c^2)
  (h_sides : a = 8 ∧ b = 15 ∧ c = 17)
  (k : ℝ)
  (h_perimeter : k * (a + b + c) = 160)
  : k * c = 68 :=
by sorry

end similar_right_triangle_longest_side_l2654_265410


namespace symmetric_graphs_intersection_l2654_265446

noncomputable def f (a b x : ℝ) : ℝ := 2*a + 1/(x-b)

theorem symmetric_graphs_intersection (a b c d : ℝ) :
  (∃! x, f a b x = f c d x) ↔ (a - c) * (b - d) = 2 := by sorry

end symmetric_graphs_intersection_l2654_265446


namespace salary_change_l2654_265492

theorem salary_change (x : ℝ) :
  (1 - x / 100) * (1 + x / 100) = 0.75 → x = 50 := by
  sorry

end salary_change_l2654_265492


namespace pentagon_count_l2654_265499

/-- The number of points on the circumference of the circle -/
def n : ℕ := 15

/-- The number of vertices in each pentagon -/
def k : ℕ := 5

/-- The number of different convex pentagons that can be formed -/
def num_pentagons : ℕ := n.choose k

theorem pentagon_count : num_pentagons = 3003 := by
  sorry

end pentagon_count_l2654_265499


namespace systematic_sampling_result_l2654_265469

-- Define the total number of employees
def total_employees : ℕ := 200

-- Define the number of groups
def num_groups : ℕ := 40

-- Define the size of each group
def group_size : ℕ := 5

-- Define the group from which the known number is drawn
def known_group : ℕ := 5

-- Define the known number drawn
def known_number : ℕ := 23

-- Define the target group
def target_group : ℕ := 10

-- Theorem statement
theorem systematic_sampling_result :
  -- Ensure the total number of employees is divisible by the number of groups
  total_employees = num_groups * group_size →
  -- Ensure the known number is within the range of the known group
  known_number > (known_group - 1) * group_size ∧ known_number ≤ known_group * group_size →
  -- Prove that the number drawn from the target group is 48
  ∃ (n : ℕ), n = (target_group - 1) * group_size + (known_number - (known_group - 1) * group_size) ∧ n = 48 :=
by sorry

end systematic_sampling_result_l2654_265469


namespace democrat_ratio_l2654_265438

theorem democrat_ratio (total_participants : ℕ) 
  (female_participants male_participants : ℕ)
  (female_democrats male_democrats : ℕ) :
  total_participants = 720 →
  female_participants + male_participants = total_participants →
  female_democrats = female_participants / 2 →
  male_democrats = male_participants / 4 →
  female_democrats = 120 →
  (female_democrats + male_democrats) * 3 = total_participants :=
by sorry

end democrat_ratio_l2654_265438


namespace factor_sum_problem_l2654_265455

theorem factor_sum_problem (N : ℕ) 
  (h1 : N > 0)
  (h2 : ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a ∣ N ∧ b ∣ N ∧ a + b = 4 ∧ ∀ (x : ℕ), x > 0 → x ∣ N → x ≥ a ∧ x ≥ b)
  (h3 : ∃ (c d : ℕ), c > 0 ∧ d > 0 ∧ c ∣ N ∧ d ∣ N ∧ c + d = 204 ∧ ∀ (x : ℕ), x > 0 → x ∣ N → x ≤ c ∧ x ≤ d) :
  N = 153 := by
sorry

end factor_sum_problem_l2654_265455


namespace stock_price_calculation_l2654_265456

/-- Calculates the final stock price after two years of changes -/
def final_stock_price (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) : ℝ :=
  let price_after_first_year := initial_price * (1 + first_year_increase)
  price_after_first_year * (1 - second_year_decrease)

/-- Theorem stating that given the specific conditions, the final stock price is $262.5 -/
theorem stock_price_calculation :
  final_stock_price 150 1.5 0.3 = 262.5 := by
  sorry

end stock_price_calculation_l2654_265456


namespace specific_polyhedron_volume_l2654_265464

/-- Represents a polyhedron formed by folding a specific figure -/
structure Polyhedron where
  /-- Number of isosceles right triangles in the figure -/
  num_triangles : Nat
  /-- Number of squares in the figure -/
  num_squares : Nat
  /-- Number of regular hexagons in the figure -/
  num_hexagons : Nat
  /-- Side length of the isosceles right triangles -/
  triangle_side : ℝ
  /-- Side length of the squares -/
  square_side : ℝ
  /-- Side length of the regular hexagon -/
  hexagon_side : ℝ

/-- Calculates the volume of the polyhedron -/
def volume (p : Polyhedron) : ℝ :=
  sorry

/-- Theorem stating the volume of the specific polyhedron -/
theorem specific_polyhedron_volume :
  let p : Polyhedron := {
    num_triangles := 3,
    num_squares := 3,
    num_hexagons := 1,
    triangle_side := 2,
    square_side := 2,
    hexagon_side := Real.sqrt 8
  }
  volume p = 47 / 6 := by
  sorry

end specific_polyhedron_volume_l2654_265464


namespace nelly_babysitting_nights_l2654_265434

/-- The number of nights Nelly needs to babysit to afford pizza for herself and her friends -/
def nights_to_babysit (friends : ℕ) (pizza_cost : ℕ) (people_per_pizza : ℕ) (earnings_per_night : ℕ) : ℕ :=
  let total_people := friends + 1
  let pizzas_needed := (total_people + people_per_pizza - 1) / people_per_pizza
  let total_cost := pizzas_needed * pizza_cost
  (total_cost + earnings_per_night - 1) / earnings_per_night

/-- Theorem stating that Nelly needs to babysit for 15 nights to afford pizza for herself and 14 friends -/
theorem nelly_babysitting_nights :
  nights_to_babysit 14 12 3 4 = 15 := by
  sorry


end nelly_babysitting_nights_l2654_265434


namespace lemon_ratio_l2654_265475

-- Define the number of lemons for each person
def levi_lemons : ℕ := 5
def jayden_lemons : ℕ := levi_lemons + 6
def ian_lemons : ℕ := 66  -- This is derived from the total, not given directly
def eli_lemons : ℕ := ian_lemons / 2
def total_lemons : ℕ := 115

-- Theorem statement
theorem lemon_ratio : 
  levi_lemons = 5 ∧
  jayden_lemons = levi_lemons + 6 ∧
  eli_lemons = ian_lemons / 2 ∧
  levi_lemons + jayden_lemons + eli_lemons + ian_lemons = total_lemons ∧
  total_lemons = 115 →
  jayden_lemons * 3 = eli_lemons := by
  sorry

end lemon_ratio_l2654_265475


namespace total_water_consumption_is_417_total_water_consumption_proof_l2654_265439

/-- Represents a washing machine with water consumption rates for different wash types -/
structure WashingMachine where
  heavy_wash : ℕ
  regular_wash : ℕ
  light_wash : ℕ

/-- Calculates the total water consumption for a washing machine -/
def water_consumption (m : WashingMachine) (heavy regular light bleach : ℕ) : ℕ :=
  m.heavy_wash * heavy + m.regular_wash * regular + m.light_wash * (light + bleach)

/-- Theorem: The total water consumption for all machines is 417 gallons -/
theorem total_water_consumption_is_417 : ℕ :=
  let machine_a : WashingMachine := ⟨25, 15, 3⟩
  let machine_b : WashingMachine := ⟨20, 12, 2⟩
  let machine_c : WashingMachine := ⟨30, 18, 4⟩
  
  let total_consumption :=
    water_consumption machine_a 3 4 2 4 +
    water_consumption machine_b 2 3 1 3 +
    water_consumption machine_c 4 2 1 5

  417

theorem total_water_consumption_proof :
  (let machine_a : WashingMachine := ⟨25, 15, 3⟩
   let machine_b : WashingMachine := ⟨20, 12, 2⟩
   let machine_c : WashingMachine := ⟨30, 18, 4⟩
   
   let total_consumption :=
     water_consumption machine_a 3 4 2 4 +
     water_consumption machine_b 2 3 1 3 +
     water_consumption machine_c 4 2 1 5

   total_consumption) = 417 := by
  sorry

end total_water_consumption_is_417_total_water_consumption_proof_l2654_265439


namespace unique_satisfying_function_l2654_265496

/-- A function satisfying the given functional equations -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (Real.sqrt 3 / 3 * x) = Real.sqrt 3 * f x - 2 * Real.sqrt 3 / 3 * x) ∧
  (∀ x y, y ≠ 0 → f x * f y = f (x * y) + f (x / y))

/-- The theorem stating that x + 1/x is the only function satisfying the given equations -/
theorem unique_satisfying_function :
  ∀ f : ℝ → ℝ, SatisfyingFunction f ↔ ∀ x, f x = x + 1/x :=
sorry

end unique_satisfying_function_l2654_265496


namespace three_numbers_sum_l2654_265417

theorem three_numbers_sum (a b c : ℝ) :
  a ≤ b → b ≤ c →
  b = 10 →
  (a + b + c) / 3 = a + 20 →
  (a + b + c) / 3 = c - 10 →
  a + b + c = 0 := by
  sorry

end three_numbers_sum_l2654_265417


namespace balls_in_boxes_theorem_l2654_265415

def number_of_ways (n m k : ℕ) : ℕ :=
  Nat.choose n m * Nat.choose m k * Nat.factorial k

theorem balls_in_boxes_theorem : number_of_ways 5 4 2 = 180 := by
  sorry

end balls_in_boxes_theorem_l2654_265415


namespace gcd_459_357_l2654_265422

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end gcd_459_357_l2654_265422


namespace rotate_point_A_l2654_265465

/-- Rotate a point 180 degrees about the origin -/
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

theorem rotate_point_A : 
  let A : ℝ × ℝ := (-4, 1)
  rotate180 A = (4, -1) := by
sorry

end rotate_point_A_l2654_265465


namespace min_value_of_function_l2654_265401

theorem min_value_of_function (x : ℝ) (h : x > 2) :
  (x^2 - 4*x + 8) / (x - 2) ≥ 4 ∧ ∃ y > 2, (y^2 - 4*y + 8) / (y - 2) = 4 := by
sorry

end min_value_of_function_l2654_265401


namespace power_difference_inequality_l2654_265411

theorem power_difference_inequality (n : ℕ) (a b : ℝ) 
  (hn : n > 1) (hab : a > b) (hb : b > 0) :
  (a^n - b^n) * (1/b^(n-1) - 1/a^(n-1)) > 4*n*(n-1)*(Real.sqrt a - Real.sqrt b)^2 := by
  sorry

end power_difference_inequality_l2654_265411


namespace job_completion_time_l2654_265435

/-- Proves that given the conditions of the problem, A takes 30 days to complete the job alone. -/
theorem job_completion_time (x : ℝ) (h1 : x > 0) (h2 : 10 * (1 / x + 1 / 40) = 0.5833333333333334) : x = 30 := by
  sorry

end job_completion_time_l2654_265435


namespace probability_two_slate_rocks_l2654_265400

/-- The probability of selecting two slate rocks without replacement from a collection of rocks -/
theorem probability_two_slate_rocks (slate pumice granite : ℕ) 
  (h_slate : slate = 14)
  (h_pumice : pumice = 20)
  (h_granite : granite = 10) :
  let total := slate + pumice + granite
  (slate : ℚ) / total * ((slate - 1) : ℚ) / (total - 1) = 13 / 1892 := by
  sorry

end probability_two_slate_rocks_l2654_265400


namespace fraction_inequality_solution_set_l2654_265423

theorem fraction_inequality_solution_set :
  {x : ℝ | (x - 2) / (x + 1) < 0} = {x : ℝ | -1 < x ∧ x < 2} := by
  sorry

end fraction_inequality_solution_set_l2654_265423


namespace professor_percentage_l2654_265457

theorem professor_percentage (total : ℝ) (women_percent : ℝ) (tenured_percent : ℝ) (men_tenured_percent : ℝ) :
  women_percent = 70 →
  tenured_percent = 70 →
  men_tenured_percent = 50 →
  let women := total * (women_percent / 100)
  let tenured := total * (tenured_percent / 100)
  let men := total - women
  let men_tenured := men * (men_tenured_percent / 100)
  let women_tenured := tenured - men_tenured
  let women_or_tenured := women + tenured - women_tenured
  (women_or_tenured / total) * 100 = 85 := by
  sorry

end professor_percentage_l2654_265457


namespace anthonys_remaining_pencils_l2654_265404

/-- Represents the number of pencils Anthony has initially -/
def initial_pencils : ℝ := 56.0

/-- Represents the number of pencils Anthony gives to Kathryn -/
def pencils_given : ℝ := 9.5

/-- Theorem stating that Anthony's remaining pencils equal the initial amount minus the amount given away -/
theorem anthonys_remaining_pencils : 
  initial_pencils - pencils_given = 46.5 := by sorry

end anthonys_remaining_pencils_l2654_265404


namespace initial_books_count_l2654_265488

theorem initial_books_count (action_figures : ℕ) (added_books : ℕ) (difference : ℕ) : 
  action_figures = 7 →
  added_books = 4 →
  difference = 1 →
  ∃ (initial_books : ℕ), 
    initial_books + added_books + difference = action_figures ∧
    initial_books = 2 := by
  sorry

end initial_books_count_l2654_265488


namespace quadratic_root_implies_b_value_l2654_265497

theorem quadratic_root_implies_b_value (b : ℝ) : 
  (Complex.I ^ 2 = -1) →
  ((3 : ℂ) + Complex.I) ^ 2 - 6 * ((3 : ℂ) + Complex.I) + b = 0 →
  b = 10 := by
  sorry

end quadratic_root_implies_b_value_l2654_265497


namespace sum_of_x_solutions_is_zero_l2654_265402

theorem sum_of_x_solutions_is_zero (y : ℝ) (h1 : y = 10) (h2 : ∃ x : ℝ, x^2 + y^2 = 169) : 
  ∃ x1 x2 : ℝ, x1^2 + y^2 = 169 ∧ x2^2 + y^2 = 169 ∧ x1 + x2 = 0 :=
sorry

end sum_of_x_solutions_is_zero_l2654_265402


namespace triple_composition_even_l2654_265473

def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem triple_composition_even (g : ℝ → ℝ) (h : IsEven g) : IsEven (fun x ↦ g (g (g x))) := by
  sorry

end triple_composition_even_l2654_265473


namespace c_minus_a_equals_40_l2654_265421

theorem c_minus_a_equals_40
  (a b c d e : ℝ)
  (h1 : (a + b) / 2 = 40)
  (h2 : (b + c) / 2 = 60)
  (h3 : (d + e) / 2 = 80)
  (h4 : (a * b * d) = (b * c * e)) :
  c - a = 40 := by
  sorry

end c_minus_a_equals_40_l2654_265421


namespace system_solution_l2654_265483

theorem system_solution :
  let f (x y z : ℚ) := (x * y = x + 2 * y) ∧ (y * z = y + 3 * z) ∧ (z * x = z + 4 * x)
  ∀ x y z : ℚ, f x y z ↔ (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 25/9 ∧ y = 25/7 ∧ z = 25/4) :=
by sorry

end system_solution_l2654_265483
