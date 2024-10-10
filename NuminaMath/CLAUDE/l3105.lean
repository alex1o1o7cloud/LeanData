import Mathlib

namespace campaign_fund_family_contribution_percentage_l3105_310567

/-- Calculates the percentage of family contribution in a campaign fund scenario -/
theorem campaign_fund_family_contribution_percentage 
  (total_funds : ℝ) 
  (friends_percentage : ℝ) 
  (president_savings : ℝ) : 
  total_funds = 10000 →
  friends_percentage = 40 →
  president_savings = 4200 →
  let friends_contribution := (friends_percentage / 100) * total_funds
  let remaining_after_friends := total_funds - friends_contribution
  let family_contribution := remaining_after_friends - president_savings
  (family_contribution / remaining_after_friends) * 100 = 30 := by
sorry

end campaign_fund_family_contribution_percentage_l3105_310567


namespace job_completion_time_l3105_310531

/-- The time taken for three workers to complete a job together, given their individual efficiencies -/
theorem job_completion_time 
  (sakshi_time : ℝ) 
  (tanya_efficiency : ℝ) 
  (rahul_efficiency : ℝ) 
  (h1 : sakshi_time = 20) 
  (h2 : tanya_efficiency = 1.25) 
  (h3 : rahul_efficiency = 1.5) : 
  (1 / (1 / sakshi_time + tanya_efficiency * (1 / sakshi_time) + rahul_efficiency * tanya_efficiency * (1 / sakshi_time))) = 160 / 33 :=
by sorry

end job_completion_time_l3105_310531


namespace simplify_expression_l3105_310589

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 - b^3 = a - b) :
  a/b + b/a - 2/(a*b) = 1 - 1/(a*b) := by
  sorry

end simplify_expression_l3105_310589


namespace sports_meeting_participation_l3105_310595

theorem sports_meeting_participation (field_events track_events both : ℕ) 
  (h1 : field_events = 15)
  (h2 : track_events = 13)
  (h3 : both = 5) :
  field_events + track_events - both = 23 :=
by sorry

end sports_meeting_participation_l3105_310595


namespace equilateral_triangle_l3105_310549

theorem equilateral_triangle (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ a + c > b)
  (side_relation : a^2 + 2*b^2 + c^2 - 2*b*(a + c) = 0) : 
  a = b ∧ b = c := by
sorry

end equilateral_triangle_l3105_310549


namespace dimes_in_shorts_l3105_310558

/-- Given a total amount of money and the number of dimes in a jacket, 
    calculate the number of dimes in the shorts. -/
theorem dimes_in_shorts 
  (total : ℚ) 
  (jacket_dimes : ℕ) 
  (dime_value : ℚ) 
  (h1 : total = 19/10) 
  (h2 : jacket_dimes = 15) 
  (h3 : dime_value = 1/10) : 
  ↑jacket_dimes * dime_value + 4 * dime_value = total :=
sorry

end dimes_in_shorts_l3105_310558


namespace tan_100_degrees_l3105_310592

theorem tan_100_degrees (k : ℝ) (h : Real.sin (-(80 * π / 180)) = k) :
  Real.tan ((100 * π) / 180) = k / Real.sqrt (1 - k^2) := by sorry

end tan_100_degrees_l3105_310592


namespace remainder_problem_l3105_310509

theorem remainder_problem (N : ℤ) (h : N % 242 = 100) : N % 18 = 6 := by
  sorry

end remainder_problem_l3105_310509


namespace count_distinct_keys_l3105_310507

/-- Represents a rotational stencil cipher key of size n × n -/
structure StencilKey (n : ℕ) where
  size : n % 2 = 0  -- n is even

/-- The number of distinct rotational stencil cipher keys for a given even size n -/
def num_distinct_keys (n : ℕ) : ℕ := 4^(n^2/4)

/-- Theorem stating the number of distinct rotational stencil cipher keys -/
theorem count_distinct_keys (n : ℕ) (key : StencilKey n) :
  num_distinct_keys n = 4^(n^2/4) := by
  sorry

#check count_distinct_keys

end count_distinct_keys_l3105_310507


namespace solve_quadratic_equation_falling_object_time_l3105_310598

-- Part 1: Solving (x-1)^2 = 49
theorem solve_quadratic_equation :
  ∀ x : ℝ, (x - 1)^2 = 49 ↔ x = 8 ∨ x = -6 :=
by sorry

-- Part 2: Finding the time for an object to reach the ground
theorem falling_object_time (h t : ℝ) :
  h = 4.9 * t^2 →
  h = 10 →
  t = 10 / 7 :=
by sorry

end solve_quadratic_equation_falling_object_time_l3105_310598


namespace solve_equation_l3105_310591

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (n : ℝ) : Prop := (2 : ℂ) / (1 - i) = 1 + n * i

-- State the theorem
theorem solve_equation : ∃ (n : ℝ), equation n ∧ n = 1 := by
  sorry

end solve_equation_l3105_310591


namespace length_MN_is_six_l3105_310579

-- Define the points
variable (A B C D M N : ℝ)

-- Define the conditions
axiom on_segment : A < C ∧ C < D ∧ D < B
axiom midpoint_M : M = (A + C) / 2
axiom midpoint_N : N = (D + B) / 2
axiom length_AB : B - A = 10
axiom length_CD : D - C = 2

-- Theorem statement
theorem length_MN_is_six : N - M = 6 := by sorry

end length_MN_is_six_l3105_310579


namespace min_deliveries_to_breakeven_l3105_310539

def van_cost : ℕ := 8000
def earning_per_delivery : ℕ := 15
def gas_cost_per_delivery : ℕ := 5

theorem min_deliveries_to_breakeven :
  ∃ (d : ℕ), d * (earning_per_delivery - gas_cost_per_delivery) ≥ van_cost ∧
  ∀ (k : ℕ), k * (earning_per_delivery - gas_cost_per_delivery) ≥ van_cost → k ≥ d :=
by sorry

end min_deliveries_to_breakeven_l3105_310539


namespace david_rosy_age_difference_l3105_310546

/-- David and Rosy's ages problem -/
theorem david_rosy_age_difference :
  ∀ (david_age rosy_age : ℕ),
    rosy_age = 12 →
    david_age + 6 = 2 * (rosy_age + 6) →
    david_age - rosy_age = 18 :=
by sorry

end david_rosy_age_difference_l3105_310546


namespace percentage_deposited_approx_28_percent_l3105_310511

def deposit : ℝ := 4500
def monthly_income : ℝ := 16071.42857142857

theorem percentage_deposited_approx_28_percent :
  ∃ ε > 0, ε < 0.01 ∧ |deposit / monthly_income * 100 - 28| < ε := by
  sorry

end percentage_deposited_approx_28_percent_l3105_310511


namespace fraction_sum_equals_decimal_l3105_310563

theorem fraction_sum_equals_decimal : (1 : ℚ) / 10 + 2 / 20 - 3 / 60 = (15 : ℚ) / 100 := by
  sorry

end fraction_sum_equals_decimal_l3105_310563


namespace safari_count_difference_l3105_310590

/-- The number of animals Josie counted on safari --/
structure SafariCount where
  antelopes : ℕ
  rabbits : ℕ
  hyenas : ℕ
  wild_dogs : ℕ
  leopards : ℕ

/-- The conditions of Josie's safari count --/
def safari_conditions (count : SafariCount) : Prop :=
  count.antelopes = 80 ∧
  count.rabbits = count.antelopes + 34 ∧
  count.hyenas < count.antelopes + count.rabbits ∧
  count.wild_dogs = count.hyenas + 50 ∧
  count.leopards * 2 = count.rabbits ∧
  count.antelopes + count.rabbits + count.hyenas + count.wild_dogs + count.leopards = 605

/-- The theorem stating the difference between hyenas and the sum of antelopes and rabbits --/
theorem safari_count_difference (count : SafariCount) 
  (h : safari_conditions count) : 
  count.antelopes + count.rabbits - count.hyenas = 42 := by
  sorry

end safari_count_difference_l3105_310590


namespace shaded_square_area_ratio_l3105_310552

theorem shaded_square_area_ratio :
  ∀ (n : ℕ) (large_square_side : ℝ) (small_square_side : ℝ),
    n = 4 →
    large_square_side = n * small_square_side →
    small_square_side > 0 →
    (2 * small_square_side^2) / (large_square_side^2) = 1/8 :=
by sorry

end shaded_square_area_ratio_l3105_310552


namespace min_value_expression_l3105_310505

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 4) :
  (a + 3 * b) * (2 * b + 3 * c) * (a * c + 2) ≥ 192 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 4 ∧
    (a₀ + 3 * b₀) * (2 * b₀ + 3 * c₀) * (a₀ * c₀ + 2) = 192 :=
by sorry

end min_value_expression_l3105_310505


namespace power_of_seven_mod_four_l3105_310522

theorem power_of_seven_mod_four : 7^150 % 4 = 1 := by
  sorry

end power_of_seven_mod_four_l3105_310522


namespace a_minus_b_equals_two_l3105_310560

theorem a_minus_b_equals_two (a b : ℝ) (h : (a - 5)^2 + |b^3 - 27| = 0) : a - b = 2 := by
  sorry

end a_minus_b_equals_two_l3105_310560


namespace sphere_radius_is_six_l3105_310504

/-- The shadow length of the sphere -/
def sphere_shadow : ℝ := 12

/-- The height of the meter stick -/
def stick_height : ℝ := 1.5

/-- The shadow length of the meter stick -/
def stick_shadow : ℝ := 3

/-- The radius of the sphere -/
def sphere_radius : ℝ := 6

/-- Theorem stating that the radius of the sphere is 6 meters given the conditions -/
theorem sphere_radius_is_six :
  stick_height / stick_shadow = sphere_radius / sphere_shadow :=
by sorry

end sphere_radius_is_six_l3105_310504


namespace payment_is_two_l3105_310528

def payment_per_window (stories : ℕ) (windows_per_floor : ℕ) (subtraction_rate : ℚ)
  (days_taken : ℕ) (final_payment : ℚ) : ℚ :=
  let total_windows := stories * windows_per_floor
  let subtraction := (days_taken / 3 : ℚ) * subtraction_rate
  let original_payment := final_payment + subtraction
  original_payment / total_windows

theorem payment_is_two :
  payment_per_window 3 3 1 6 16 = 2 := by sorry

end payment_is_two_l3105_310528


namespace jimmy_garden_servings_l3105_310513

/-- Represents the number of plants in each plot -/
def plants_per_plot : ℕ := 9

/-- Represents the number of servings produced by each carrot plant -/
def carrot_servings : ℕ := 4

/-- Represents the number of servings produced by each corn plant -/
def corn_servings : ℕ := 5 * carrot_servings

/-- Represents the number of servings produced by each green bean plant -/
def green_bean_servings : ℕ := corn_servings / 2

/-- Calculates the total number of servings from all three plots -/
def total_servings : ℕ := 
  plants_per_plot * carrot_servings +
  plants_per_plot * corn_servings +
  plants_per_plot * green_bean_servings

/-- Theorem stating that the total number of servings is 306 -/
theorem jimmy_garden_servings : total_servings = 306 := by
  sorry

end jimmy_garden_servings_l3105_310513


namespace min_operations_rectangle_l3105_310576

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Measures the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

/-- Checks if two numbers are equal -/
def compare (a b : ℝ) : Bool :=
  sorry

/-- Checks if a quadrilateral is a rectangle -/
def isRectangle (q : Quadrilateral) : Bool :=
  sorry

/-- Counts the number of operations needed to determine if a quadrilateral is a rectangle -/
def countOperations (q : Quadrilateral) : ℕ :=
  sorry

/-- Theorem stating that the minimum number of operations to determine if a quadrilateral is a rectangle is 9 -/
theorem min_operations_rectangle (q : Quadrilateral) : 
  (countOperations q = 9) ∧ (∀ n : ℕ, n < 9 → ¬(∀ q' : Quadrilateral, isRectangle q' ↔ countOperations q' ≤ n)) :=
  sorry

end min_operations_rectangle_l3105_310576


namespace f_f_eq_f_solutions_l3105_310533

def f (x : ℝ) := x^2 - 2*x

theorem f_f_eq_f_solutions :
  {x : ℝ | f (f x) = f x} = {0, 2, -1, 3} := by sorry

end f_f_eq_f_solutions_l3105_310533


namespace pet_store_parakeets_l3105_310548

/-- Calculates the number of parakeets in a pet store given the number of cages, parrots, and average birds per cage. -/
theorem pet_store_parakeets 
  (num_cages : ℝ) 
  (num_parrots : ℝ) 
  (avg_birds_per_cage : ℝ) 
  (h1 : num_cages = 6)
  (h2 : num_parrots = 6)
  (h3 : avg_birds_per_cage = 1.333333333) :
  num_cages * avg_birds_per_cage - num_parrots = 2 := by
  sorry

end pet_store_parakeets_l3105_310548


namespace interest_rate_proof_l3105_310532

/-- Given simple interest and compound interest for 2 years, prove the interest rate -/
theorem interest_rate_proof (P : ℝ) (R : ℝ) : 
  (2 * P * R / 100 = 600) →  -- Simple interest condition
  (P * ((1 + R / 100)^2 - 1) = 630) →  -- Compound interest condition
  R = 10 := by
  sorry

end interest_rate_proof_l3105_310532


namespace baker_cakes_remaining_l3105_310599

/-- Given the initial number of cakes, additional cakes made, and cakes sold,
    prove that the number of cakes remaining is equal to 67. -/
theorem baker_cakes_remaining 
  (initial_cakes : ℕ) 
  (additional_cakes : ℕ) 
  (sold_cakes : ℕ) 
  (h1 : initial_cakes = 62)
  (h2 : additional_cakes = 149)
  (h3 : sold_cakes = 144) :
  initial_cakes + additional_cakes - sold_cakes = 67 := by
  sorry

end baker_cakes_remaining_l3105_310599


namespace playset_cost_indeterminate_l3105_310585

theorem playset_cost_indeterminate 
  (lumber_inflation : ℝ) 
  (nails_inflation : ℝ) 
  (fabric_inflation : ℝ) 
  (total_increase : ℝ) 
  (h1 : lumber_inflation = 0.20)
  (h2 : nails_inflation = 0.10)
  (h3 : fabric_inflation = 0.05)
  (h4 : total_increase = 97) :
  ∃ (L N F : ℝ), 
    L * lumber_inflation + N * nails_inflation + F * fabric_inflation = total_increase ∧
    ∃ (L' N' F' : ℝ), 
      L' ≠ L ∧
      L' * lumber_inflation + N' * nails_inflation + F' * fabric_inflation = total_increase :=
by sorry

end playset_cost_indeterminate_l3105_310585


namespace perpendicular_lines_m_value_l3105_310586

/-- Given two lines that are perpendicular, prove that the value of m is 1/2 -/
theorem perpendicular_lines_m_value (m : ℝ) : 
  (∀ x y : ℝ, x - m * y + 2 * m = 0 ∨ x + 2 * y - m = 0) → 
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    x₁ - m * y₁ + 2 * m = 0 → 
    x₂ + 2 * y₂ - m = 0 → 
    (x₁ - x₂) * (y₁ - y₂) = 0) →
  m = 1 / 2 := by
sorry

end perpendicular_lines_m_value_l3105_310586


namespace circle_and_line_problem_l3105_310501

/-- Given a circle A with center at (-1, 2) tangent to line m: x + 2y + 7 = 0,
    and a moving line l passing through B(-2, 0) intersecting circle A at M and N,
    prove the equation of circle A and find the equations of line l when |MN| = 2√19. -/
theorem circle_and_line_problem :
  ∀ (A : ℝ × ℝ) (m : ℝ → ℝ → Prop) (l : ℝ → ℝ → Prop) (M N : ℝ × ℝ),
  A = (-1, 2) →
  (∀ x y, m x y ↔ x + 2*y + 7 = 0) →
  (∃ r : ℝ, ∀ x y, (x + 1)^2 + (y - 2)^2 = r^2 ↔ m x y) →
  (∀ x, l x 0 ↔ x = -2) →
  (∃ x y, l x y ∧ (x + 1)^2 + (y - 2)^2 = 20) →
  (M.1 - N.1)^2 + (M.2 - N.2)^2 = 4*19 →
  ((∀ x y, (x + 1)^2 + (y - 2)^2 = 20 ↔ (x - A.1)^2 + (y - A.2)^2 = 20) ∧
   ((∀ x y, l x y ↔ 3*x - 4*y + 6 = 0) ∨ (∀ x y, l x y ↔ x = -2))) :=
by sorry

end circle_and_line_problem_l3105_310501


namespace cube_surface_area_increase_l3105_310570

theorem cube_surface_area_increase (s : ℝ) (h : s > 0) :
  let original_area := 6 * s^2
  let new_edge := 1.25 * s
  let new_area := 6 * new_edge^2
  (new_area - original_area) / original_area = 0.5625 := by
sorry

end cube_surface_area_increase_l3105_310570


namespace orchids_count_l3105_310597

/-- Represents the number of flowers in a vase -/
structure FlowerVase where
  initial_roses : Nat
  initial_orchids : Nat
  current_roses : Nat
  orchid_rose_difference : Nat

/-- Calculates the current number of orchids in the vase -/
def current_orchids (vase : FlowerVase) : Nat :=
  vase.current_roses + vase.orchid_rose_difference

theorem orchids_count (vase : FlowerVase) 
  (h1 : vase.initial_roses = 7)
  (h2 : vase.initial_orchids = 12)
  (h3 : vase.current_roses = 11)
  (h4 : vase.orchid_rose_difference = 9) :
  current_orchids vase = 20 := by
  sorry

end orchids_count_l3105_310597


namespace inequality_proof_l3105_310580

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  b^2 / a + a^2 / b ≥ a + b := by
  sorry

end inequality_proof_l3105_310580


namespace nonzero_terms_count_l3105_310596

def expand_polynomial (x : ℝ) : ℝ := (2*x+5)*(3*x^2+x+6) - 4*(x^3-3*x^2+5*x-1)

theorem nonzero_terms_count :
  ∃ (a b c d : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
  ∀ x, expand_polynomial x = a*x^3 + b*x^2 + c*x + d :=
sorry

end nonzero_terms_count_l3105_310596


namespace change_received_l3105_310574

def skirt_price : ℝ := 13
def blouse_price : ℝ := 6
def shoes_price : ℝ := 25
def handbag_price : ℝ := 35
def handbag_discount_rate : ℝ := 0.1
def coupon_discount : ℝ := 5
def amount_paid : ℝ := 150

def total_cost : ℝ := 2 * skirt_price + 3 * blouse_price + shoes_price + handbag_price

def discounted_handbag_price : ℝ := handbag_price * (1 - handbag_discount_rate)

def total_cost_after_discounts : ℝ := 
  2 * skirt_price + 3 * blouse_price + shoes_price + discounted_handbag_price - coupon_discount

theorem change_received : 
  amount_paid - total_cost_after_discounts = 54.5 := by sorry

end change_received_l3105_310574


namespace least_integer_greater_than_sqrt_450_l3105_310516

theorem least_integer_greater_than_sqrt_450 : 
  (∀ n : ℤ, n ≤ ⌊Real.sqrt 450⌋ → n < 22) ∧ 22 > Real.sqrt 450 := by
  sorry

end least_integer_greater_than_sqrt_450_l3105_310516


namespace binomial_12_choose_3_l3105_310510

theorem binomial_12_choose_3 : Nat.choose 12 3 = 220 := by
  sorry

end binomial_12_choose_3_l3105_310510


namespace hyperbola_equation_l3105_310512

/-- A hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_equation : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1
  h_focus : ∃ c : ℝ, c = 5 -- Right focus coincides with focus of y^2 = 20x
  h_asymptote : ∀ x y : ℝ, y = 4/3 * x ∨ y = -4/3 * x

/-- The theorem stating that the hyperbola with given properties has the equation x^2/9 - y^2/16 = 1 -/
theorem hyperbola_equation (C : Hyperbola) : C.a^2 = 9 ∧ C.b^2 = 16 := by
  sorry

end hyperbola_equation_l3105_310512


namespace function_nonnegative_implies_a_range_l3105_310540

theorem function_nonnegative_implies_a_range 
  (f : ℝ → ℝ) 
  (h : ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x ≥ 0) 
  (h_def : ∀ x, f x = x^2 + a*x + 3 - a) : 
  a ∈ Set.Icc (-7 : ℝ) 2 := by
sorry

end function_nonnegative_implies_a_range_l3105_310540


namespace square_sum_reciprocal_l3105_310537

theorem square_sum_reciprocal (x : ℝ) (h : x + 1/x = 8) : x^2 + 1/x^2 = 62 := by
  sorry

end square_sum_reciprocal_l3105_310537


namespace marble_redistribution_l3105_310534

/-- Given Tyrone's initial marbles -/
def tyrone_initial : ℕ := 150

/-- Given Eric's initial marbles -/
def eric_initial : ℕ := 30

/-- The number of marbles Tyrone gives to Eric -/
def marbles_given : ℕ := 15

theorem marble_redistribution :
  (tyrone_initial - marbles_given = 3 * (eric_initial + marbles_given)) ∧
  (0 < marbles_given) ∧ (marbles_given < tyrone_initial) := by
  sorry

end marble_redistribution_l3105_310534


namespace middle_card_is_six_l3105_310527

/-- Represents a set of three cards with positive integers -/
structure CardSet where
  left : Nat
  middle : Nat
  right : Nat
  sum_is_17 : left + middle + right = 17
  increasing : left < middle ∧ middle < right

/-- Predicate to check if a number allows for multiple possibilities when seen on the left -/
def leftIndeterminate (n : Nat) : Prop :=
  ∃ (cs1 cs2 : CardSet), cs1.left = n ∧ cs2.left = n ∧ cs1 ≠ cs2

/-- Predicate to check if a number allows for multiple possibilities when seen on the right -/
def rightIndeterminate (n : Nat) : Prop :=
  ∃ (cs1 cs2 : CardSet), cs1.right = n ∧ cs2.right = n ∧ cs1 ≠ cs2

/-- Predicate to check if a number allows for multiple possibilities when seen in the middle -/
def middleIndeterminate (n : Nat) : Prop :=
  ∃ (cs1 cs2 : CardSet), cs1.middle = n ∧ cs2.middle = n ∧ cs1 ≠ cs2

/-- The main theorem stating that the middle card must be 6 -/
theorem middle_card_is_six :
  ∀ (cs : CardSet),
    leftIndeterminate cs.left →
    rightIndeterminate cs.right →
    middleIndeterminate cs.middle →
    cs.middle = 6 := by
  sorry

end middle_card_is_six_l3105_310527


namespace unique_triple_solution_l3105_310550

theorem unique_triple_solution :
  ∃! (x y z : ℕ), (x + 1)^(y + 1) + 1 = (x + 2)^(z + 1) :=
by
  -- The proof goes here
  sorry

end unique_triple_solution_l3105_310550


namespace range_of_fraction_l3105_310581

-- Define the quadratic equation
def quadratic (a b x : ℝ) : Prop := x^2 + a*x + 2*b - 2 = 0

-- Define the theorem
theorem range_of_fraction (a b : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    quadratic a b x₁ ∧ quadratic a b x₂ ∧
    0 < x₁ ∧ x₁ < 1 ∧ 
    1 < x₂ ∧ x₂ < 2) →
  1/2 < (b - 4) / (a - 1) ∧ (b - 4) / (a - 1) < 3/2 :=
by sorry

end range_of_fraction_l3105_310581


namespace set_intersection_example_l3105_310593

theorem set_intersection_example : 
  let A : Set ℕ := {0, 1, 2}
  let B : Set ℕ := {0, 2, 4}
  A ∩ B = {0, 2} := by
sorry

end set_intersection_example_l3105_310593


namespace tangent_line_proof_l3105_310553

-- Define the given curve
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 5

-- Define the given line
def line1 (x y : ℝ) : Prop := 2*x - 6*y + 1 = 0

-- Define the tangent line we want to prove
def line2 (x y : ℝ) : Prop := 3*x + y + 6 = 0

-- Theorem statement
theorem tangent_line_proof :
  ∃ (x₀ y₀ : ℝ),
    -- The point (x₀, y₀) is on the curve
    f x₀ = y₀ ∧
    -- The point (x₀, y₀) is on line2
    line2 x₀ y₀ ∧
    -- line2 is tangent to the curve at (x₀, y₀)
    (deriv f x₀ = -3) ∧
    -- line1 and line2 are perpendicular
    (∀ (x₁ y₁ x₂ y₂ : ℝ),
      line1 x₁ y₁ → line1 x₂ y₂ → x₁ ≠ x₂ →
      line2 x₁ y₁ → line2 x₂ y₂ → x₁ ≠ x₂ →
      (y₂ - y₁) / (x₂ - x₁) * (y₂ - y₁) / (x₂ - x₁) = -1) :=
by
  sorry

end tangent_line_proof_l3105_310553


namespace fuchsia_survey_l3105_310521

theorem fuchsia_survey (total : ℕ) (kinda_pink : ℕ) (both : ℕ) (neither : ℕ)
  (h_total : total = 100)
  (h_kinda_pink : kinda_pink = 60)
  (h_both : both = 27)
  (h_neither : neither = 17) :
  ∃ (purply : ℕ), purply = 50 ∧ purply = total - (kinda_pink - both + neither) :=
by sorry

end fuchsia_survey_l3105_310521


namespace simplify_fraction_product_l3105_310508

theorem simplify_fraction_product : (240 / 24) * (7 / 140) * (6 / 4) = 3 / 4 := by
  sorry

end simplify_fraction_product_l3105_310508


namespace hyperbola_eccentricity_l3105_310594

/-- The eccentricity of a hyperbola with equation x^2/m - y^2/5 = 1 is 3/2,
    given that m > 0 and its right focus coincides with the focus of y^2 = 12x -/
theorem hyperbola_eccentricity (m : ℝ) (h1 : m > 0) : ∃ (a b c : ℝ),
  m = a^2 ∧
  b^2 = 5 ∧
  c^2 = a^2 + b^2 ∧
  c = 3 ∧
  c / a = 3 / 2 := by
sorry

end hyperbola_eccentricity_l3105_310594


namespace total_marbles_l3105_310518

theorem total_marbles (jungkook_marbles : ℕ) (jimin_extra_marbles : ℕ) : 
  jungkook_marbles = 3 → 
  jimin_extra_marbles = 4 → 
  jungkook_marbles + (jungkook_marbles + jimin_extra_marbles) = 10 := by
sorry

end total_marbles_l3105_310518


namespace parabola_shift_l3105_310584

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the horizontal shift
def shift : ℝ := 2

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ := (x - shift)^2

-- Theorem statement
theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x - shift) :=
by sorry

end parabola_shift_l3105_310584


namespace estimation_correct_l3105_310535

/-- Represents a school population --/
structure School where
  total_students : ℕ
  sample_size : ℕ
  sample_enthusiasts : ℕ

/-- Calculates the estimated number of enthusiasts in the entire school population --/
def estimate_enthusiasts (s : School) : ℕ :=
  (s.total_students * s.sample_enthusiasts) / s.sample_size

/-- Theorem stating that the estimation method in statement D is correct --/
theorem estimation_correct (s : School) 
  (h1 : s.total_students = 3200)
  (h2 : s.sample_size = 200)
  (h3 : s.sample_enthusiasts = 85) :
  estimate_enthusiasts s = 1360 := by
  sorry

#eval estimate_enthusiasts { total_students := 3200, sample_size := 200, sample_enthusiasts := 85 }

end estimation_correct_l3105_310535


namespace count_seven_to_800_l3105_310566

def count_seven (n : ℕ) : ℕ := 
  let units := n / 10
  let tens := n / 100
  let hundreds := if n ≥ 700 then 100 else 0
  units + tens * 10 + hundreds

theorem count_seven_to_800 : count_seven 800 = 260 := by sorry

end count_seven_to_800_l3105_310566


namespace solution_xy_l3105_310525

theorem solution_xy (x y : ℝ) 
  (h1 : x ≠ 0) 
  (h2 : x + y ≠ 0) 
  (h3 : (x + y) / x = y / (x + y)) 
  (h4 : x = 2 * y) : 
  x = 0 ∧ y = 0 := by
sorry

end solution_xy_l3105_310525


namespace ellipse_equation_l3105_310520

/-- The equation of an ellipse passing through points (1, √3/2) and (2, 0) -/
theorem ellipse_equation : ∃ (a b : ℝ), 
  (a > 0 ∧ b > 0) ∧ 
  (1 / a^2 + (Real.sqrt 3 / 2)^2 / b^2 = 1) ∧ 
  (4 / a^2 = 1) ∧
  ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 4 + y^2 = 1 := by
  sorry

end ellipse_equation_l3105_310520


namespace rectangular_solid_surface_area_l3105_310587

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem rectangular_solid_surface_area 
  (l w h : ℕ) 
  (prime_l : is_prime l) 
  (prime_w : is_prime w) 
  (prime_h : is_prime h) 
  (volume_eq : l * w * h = 1001) : 
  2 * (l * w + w * h + h * l) = 622 := by
sorry

end rectangular_solid_surface_area_l3105_310587


namespace illumination_theorem_l3105_310502

/-- Represents a rectangular room with a point light source and a mirror --/
structure IlluminatedRoom where
  length : ℝ
  width : ℝ
  height : ℝ
  mirror_width : ℝ
  light_source : ℝ × ℝ × ℝ

/-- Calculates the fraction of walls not illuminated in the room --/
def fraction_not_illuminated (room : IlluminatedRoom) : ℚ :=
  17 / 32

/-- Theorem stating that the fraction of walls not illuminated is 17/32 --/
theorem illumination_theorem (room : IlluminatedRoom) :
  fraction_not_illuminated room = 17 / 32 := by
  sorry

end illumination_theorem_l3105_310502


namespace percentage_of_360_equals_129_6_l3105_310529

theorem percentage_of_360_equals_129_6 : 
  (129.6 / 360) * 100 = 36 := by sorry

end percentage_of_360_equals_129_6_l3105_310529


namespace square_circle_radius_l3105_310545

theorem square_circle_radius (square_perimeter : ℝ) (circle_radius : ℝ) : 
  square_perimeter = 28 →
  circle_radius = square_perimeter / 4 →
  circle_radius = 7 := by
  sorry

end square_circle_radius_l3105_310545


namespace new_average_weight_l3105_310561

theorem new_average_weight 
  (initial_students : ℕ) 
  (initial_avg_weight : ℝ) 
  (new_student_weight : ℝ) : 
  initial_students = 29 → 
  initial_avg_weight = 28 → 
  new_student_weight = 10 → 
  let total_weight := initial_students * initial_avg_weight + new_student_weight
  let new_total_students := initial_students + 1
  (total_weight / new_total_students : ℝ) = 27.4 := by
sorry

end new_average_weight_l3105_310561


namespace equation_solution_l3105_310500

theorem equation_solution (x : ℝ) : 
  Real.sqrt ((3 + Real.sqrt 5) ^ x) + Real.sqrt ((3 - Real.sqrt 5) ^ x) = 2 ↔ x = 0 :=
by sorry

end equation_solution_l3105_310500


namespace evaluate_fraction_l3105_310519

theorem evaluate_fraction (a b : ℤ) (h1 : a = 7) (h2 : b = -3) :
  3 / (a - b) = 3 / 10 := by
  sorry

end evaluate_fraction_l3105_310519


namespace monomial_sum_implies_mn_value_l3105_310514

theorem monomial_sum_implies_mn_value 
  (m n : ℤ) 
  (h : ∃ (a : ℚ), 3 * X^(m+6) * Y^(2*n+1) + X * Y^7 = a * X^(m+6) * Y^(2*n+1)) : 
  m * n = -15 := by
  sorry

end monomial_sum_implies_mn_value_l3105_310514


namespace matrix_det_minus_two_l3105_310573

theorem matrix_det_minus_two (A : Matrix (Fin 2) (Fin 2) ℤ) :
  A = ![![9, 5], ![-3, 4]] →
  Matrix.det A - 2 = 49 := by
  sorry

end matrix_det_minus_two_l3105_310573


namespace average_reading_days_is_64_l3105_310583

/-- Represents the reading speed ratio between Emery and Serena for books -/
def book_speed_ratio : ℚ := 5

/-- Represents the reading speed ratio between Emery and Serena for articles -/
def article_speed_ratio : ℚ := 3

/-- Represents the number of days it takes Emery to read the book -/
def emery_book_days : ℕ := 20

/-- Represents the number of days it takes Emery to read the article -/
def emery_article_days : ℕ := 2

/-- Calculates the average number of days for Emery and Serena to read both the book and the article -/
def average_reading_days : ℚ := 
  let serena_book_days := emery_book_days * book_speed_ratio
  let serena_article_days := emery_article_days * article_speed_ratio
  let emery_total_days := emery_book_days + emery_article_days
  let serena_total_days := serena_book_days + serena_article_days
  (emery_total_days + serena_total_days) / 2

theorem average_reading_days_is_64 : average_reading_days = 64 := by
  sorry

end average_reading_days_is_64_l3105_310583


namespace rhombus_sides_equal_is_universal_and_true_l3105_310555

/-- A rhombus is a quadrilateral with four equal sides --/
structure Rhombus where
  sides : Fin 4 → ℝ
  all_sides_equal : ∀ i j : Fin 4, sides i = sides j

/-- The proposition "All sides of a rhombus are equal" is universal and true --/
theorem rhombus_sides_equal_is_universal_and_true :
  (∀ r : Rhombus, ∀ i j : Fin 4, r.sides i = r.sides j) ∧
  (∃ r : Rhombus, True) :=
sorry

end rhombus_sides_equal_is_universal_and_true_l3105_310555


namespace system_solution_l3105_310556

theorem system_solution (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x + y + x*y = 8)
  (eq2 : y + z + y*z = 15)
  (eq3 : z + x + z*x = 35) :
  x + y + z + x*y = 15 := by
sorry

end system_solution_l3105_310556


namespace n_fourth_plus_four_composite_l3105_310523

theorem n_fourth_plus_four_composite (n : ℕ) (h : n > 1) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^4 + 4 = a * b :=
by sorry

end n_fourth_plus_four_composite_l3105_310523


namespace math_problems_sum_l3105_310547

/-- The sum of four math problem answers given specific conditions -/
theorem math_problems_sum : 
  let answer1 : ℝ := 600
  let answer2 : ℝ := 2 * answer1
  let answer3 : ℝ := answer1 + answer2 - 400
  let answer4 : ℝ := (answer1 + answer2 + answer3) / 3
  (answer1 + answer2 + answer3 + answer4) = 4266.67 := by
  sorry

end math_problems_sum_l3105_310547


namespace max_popsicles_is_18_l3105_310568

/-- Represents the number of popsicles in a package -/
inductive Package
| Single : Package
| FourPack : Package
| SevenPack : Package
| NinePack : Package

/-- Returns the cost of a package in dollars -/
def cost (p : Package) : ℕ :=
  match p with
  | Package.Single => 2
  | Package.FourPack => 5
  | Package.SevenPack => 8
  | Package.NinePack => 10

/-- Returns the number of popsicles in a package -/
def popsicles (p : Package) : ℕ :=
  match p with
  | Package.Single => 1
  | Package.FourPack => 4
  | Package.SevenPack => 7
  | Package.NinePack => 9

/-- Represents a combination of packages -/
def Combination := List Package

/-- Calculates the total cost of a combination -/
def totalCost (c : Combination) : ℕ :=
  c.map cost |>.sum

/-- Calculates the total number of popsicles in a combination -/
def totalPopsicles (c : Combination) : ℕ :=
  c.map popsicles |>.sum

/-- Checks if a combination is within budget -/
def withinBudget (c : Combination) : Prop :=
  totalCost c ≤ 20

/-- Theorem: The maximum number of popsicles Pablo can buy with $20 is 18 -/
theorem max_popsicles_is_18 :
  ∀ c : Combination, withinBudget c → totalPopsicles c ≤ 18 :=
by sorry

end max_popsicles_is_18_l3105_310568


namespace gcd_of_powers_of_two_l3105_310554

theorem gcd_of_powers_of_two : 
  Nat.gcd (2^2050 - 1) (2^2040 - 1) = 2^10 - 1 := by sorry

end gcd_of_powers_of_two_l3105_310554


namespace drawer_probability_l3105_310575

theorem drawer_probability (shirts : ℕ) (shorts : ℕ) (socks : ℕ) :
  shirts = 6 →
  shorts = 7 →
  socks = 8 →
  let total := shirts + shorts + socks
  let favorable := Nat.choose shirts 2 * Nat.choose shorts 1 * Nat.choose socks 1
  let total_outcomes := Nat.choose total 4
  (favorable : ℚ) / total_outcomes = 56 / 399 := by
  sorry

end drawer_probability_l3105_310575


namespace non_shaded_perimeter_l3105_310536

/-- Given a large rectangle containing a smaller shaded rectangle, 
    where the total area is 180 square inches and the shaded area is 120 square inches,
    prove that the perimeter of the non-shaded region is 32 inches. -/
theorem non_shaded_perimeter (total_area shaded_area : ℝ) 
  (h1 : total_area = 180)
  (h2 : shaded_area = 120)
  (h3 : ∃ (a b : ℝ), a * b = total_area - shaded_area ∧ a + b = 16) :
  2 * 16 = 32 := by
  sorry

end non_shaded_perimeter_l3105_310536


namespace smallest_seven_digit_binary_l3105_310541

theorem smallest_seven_digit_binary : ∃ n : ℕ, n > 0 ∧ 
  (∀ m : ℕ, m > 0 → m.digits 2 = [1, 0, 0, 0, 0, 0, 0] → m ≥ n) ∧
  n.digits 2 = [1, 0, 0, 0, 0, 0, 0] ∧
  n = 64 := by
  sorry

end smallest_seven_digit_binary_l3105_310541


namespace john_payment_john_payment_is_8400_l3105_310559

/-- Calculates John's payment for lawyer fees --/
theorem john_payment (upfront_fee : ℕ) (hourly_rate : ℕ) (court_hours : ℕ) 
  (prep_time_multiplier : ℕ) (paperwork_fee : ℕ) (transport_costs : ℕ) : ℕ :=
  let total_hours := court_hours + prep_time_multiplier * court_hours
  let total_fee := upfront_fee + hourly_rate * total_hours + paperwork_fee + transport_costs
  total_fee / 2

/-- Proves that John's payment is $8400 given the specified conditions --/
theorem john_payment_is_8400 : 
  john_payment 1000 100 50 2 500 300 = 8400 := by
  sorry

end john_payment_john_payment_is_8400_l3105_310559


namespace binary_representation_of_106_l3105_310564

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec go (m : ℕ) (acc : List Bool) : List Bool :=
      if m = 0 then acc
      else go (m / 2) ((m % 2 = 1) :: acc)
    go n []

/-- Converts a list of bits to a string representation of a binary number -/
def binaryToString (bits : List Bool) : String :=
  bits.map (fun b => if b then '1' else '0') |> String.mk

theorem binary_representation_of_106 :
  binaryToString (toBinary 106) = "1101010" := by
  sorry

#eval binaryToString (toBinary 106)

end binary_representation_of_106_l3105_310564


namespace smallest_positive_multiple_of_seven_l3105_310506

theorem smallest_positive_multiple_of_seven (x : ℕ) : 
  (∃ k : ℕ, x = 7 * k) → -- x is a positive multiple of 7
  x^2 > 144 →            -- x^2 > 144
  x < 25 →               -- x < 25
  x = 14 :=              -- x = 14 is the smallest value satisfying all conditions
by
  sorry

end smallest_positive_multiple_of_seven_l3105_310506


namespace marble_box_capacity_l3105_310517

theorem marble_box_capacity (jack_capacity : ℕ) (lucy_scale : ℕ) : 
  jack_capacity = 50 → lucy_scale = 3 → 
  (lucy_scale ^ 3) * jack_capacity = 1350 := by
  sorry

end marble_box_capacity_l3105_310517


namespace sqrt_difference_simplification_l3105_310557

theorem sqrt_difference_simplification :
  3 * Real.sqrt 2 - |Real.sqrt 2 - Real.sqrt 3| = 4 * Real.sqrt 2 - Real.sqrt 3 := by
  sorry

end sqrt_difference_simplification_l3105_310557


namespace remainder_theorem_l3105_310572

theorem remainder_theorem (x y u v : ℕ) (h1 : y > 0) (h2 : x = u * y + v) (h3 : v < y) :
  (x + 3 * u * y) % y = v :=
by sorry

end remainder_theorem_l3105_310572


namespace power_product_equals_2025_l3105_310503

theorem power_product_equals_2025 (a b : ℕ) (h1 : 5^a = 3125) (h2 : 3^b = 81) :
  5^(a - 3) * 3^(2*b - 4) = 2025 := by
  sorry

end power_product_equals_2025_l3105_310503


namespace cone_base_radius_l3105_310571

/-- Represents a cone with given properties -/
structure Cone where
  surface_area : ℝ
  lateral_surface_semicircle : Prop

/-- Theorem: Given a cone with surface area 12π and lateral surface unfolding into a semicircle, 
    the radius of its base circle is 2 -/
theorem cone_base_radius 
  (cone : Cone) 
  (h1 : cone.surface_area = 12 * Real.pi) 
  (h2 : cone.lateral_surface_semicircle) : 
  ∃ (r : ℝ), r = 2 ∧ r > 0 ∧ 
  cone.surface_area = Real.pi * r^2 + Real.pi * r * (2 * r) := by
  sorry

end cone_base_radius_l3105_310571


namespace exists_irrational_less_than_neg_two_l3105_310542

theorem exists_irrational_less_than_neg_two : ∃ x : ℝ, Irrational x ∧ x < -2 := by
  sorry

end exists_irrational_less_than_neg_two_l3105_310542


namespace total_toys_count_l3105_310515

def bill_toys : ℕ := 60

def hana_toys : ℕ := (5 * bill_toys) / 6

def hash_toys : ℕ := hana_toys / 2 + 9

def total_toys : ℕ := bill_toys + hana_toys + hash_toys

theorem total_toys_count : total_toys = 144 := by
  sorry

end total_toys_count_l3105_310515


namespace sequence_negative_start_l3105_310578

def sequence_term (n : ℤ) : ℤ := 21 + 4*n - n^2

theorem sequence_negative_start :
  ∀ n : ℕ, n ≥ 8 → sequence_term n < 0 ∧ 
  ∀ k : ℕ, k < 8 → sequence_term k ≥ 0 :=
sorry

end sequence_negative_start_l3105_310578


namespace shortest_ribbon_length_l3105_310565

theorem shortest_ribbon_length (ribbon_length : ℕ) : 
  (ribbon_length % 2 = 0) ∧ 
  (ribbon_length % 5 = 0) ∧ 
  (ribbon_length % 7 = 0) ∧ 
  (∀ x : ℕ, x < ribbon_length → (x % 2 = 0 ∧ x % 5 = 0 ∧ x % 7 = 0) → False) → 
  ribbon_length = 70 := by
sorry

end shortest_ribbon_length_l3105_310565


namespace wipes_used_correct_l3105_310524

/-- Calculates the number of wipes used before refilling -/
def wipes_used (initial : ℕ) (refill : ℕ) (final : ℕ) : ℕ :=
  initial + refill - final

theorem wipes_used_correct (initial refill final : ℕ) 
  (h_initial : initial = 70)
  (h_refill : refill = 10)
  (h_final : final = 60) :
  wipes_used initial refill final = 20 := by
  sorry

#eval wipes_used 70 10 60

end wipes_used_correct_l3105_310524


namespace special_curve_hyperbola_range_l3105_310543

/-- A curve defined by the equation x^2 / (m + 2) + y^2 / (m + 1) = 1 --/
def is_special_curve (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m + 2) + y^2 / (m + 1) = 1

/-- The condition for the curve to be a hyperbola with foci on the x-axis --/
def is_hyperbola_x_foci (m : ℝ) : Prop :=
  (m + 2 > 0) ∧ (m + 1 < 0)

/-- The main theorem stating the range of m for which the curve is a hyperbola with foci on the x-axis --/
theorem special_curve_hyperbola_range (m : ℝ) :
  is_special_curve m ∧ is_hyperbola_x_foci m ↔ -2 < m ∧ m < -1 :=
sorry

end special_curve_hyperbola_range_l3105_310543


namespace dividend_calculation_l3105_310588

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17) 
  (h2 : quotient = 9) 
  (h3 : remainder = 6) : 
  divisor * quotient + remainder = 159 := by
  sorry

end dividend_calculation_l3105_310588


namespace jennifer_fruits_left_l3105_310562

/-- Calculates the number of fruits Jennifer has left after giving some to her sister. -/
def fruits_left (initial_pears initial_oranges : ℕ) (apples_multiplier : ℕ) (given_away : ℕ) : ℕ :=
  let initial_apples := initial_pears * apples_multiplier
  let remaining_pears := initial_pears - given_away
  let remaining_oranges := initial_oranges - given_away
  let remaining_apples := initial_apples - given_away
  remaining_pears + remaining_oranges + remaining_apples

/-- Theorem stating that Jennifer has 44 fruits left after giving some to her sister. -/
theorem jennifer_fruits_left : 
  fruits_left 10 20 2 2 = 44 := by sorry

end jennifer_fruits_left_l3105_310562


namespace expansion_and_a4_imply_a_and_sum_l3105_310569

/-- The expansion of (2x - a)^7 in terms of (x+1) -/
def expansion (a : ℝ) (x : ℝ) : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ :=
  λ a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ =>
    a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7

theorem expansion_and_a4_imply_a_and_sum :
  ∀ a : ℝ, ∀ a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ,
    (∀ x : ℝ, (2*x - a)^7 = expansion a x a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇) →
    a₄ = -560 →
    a = -1 ∧ |a₁| + |a₂| + |a₃| + |a₅| + |a₆| + |a₇| = 2186 :=
by sorry

end expansion_and_a4_imply_a_and_sum_l3105_310569


namespace sin_theta_value_l3105_310582

theorem sin_theta_value (θ : ℝ) (h : Real.cos (π / 4 - θ / 2) = 2 / 3) : 
  Real.sin θ = -1 / 9 := by
sorry

end sin_theta_value_l3105_310582


namespace probability_second_genuine_given_first_genuine_l3105_310577

theorem probability_second_genuine_given_first_genuine 
  (total_products : ℕ) 
  (genuine_products : ℕ) 
  (defective_products : ℕ) 
  (h1 : total_products = genuine_products + defective_products)
  (h2 : genuine_products = 6)
  (h3 : defective_products = 4) :
  (genuine_products - 1) / (total_products - 1) = 5 / 9 := by
  sorry

end probability_second_genuine_given_first_genuine_l3105_310577


namespace similar_triangles_perimeter_l3105_310551

-- Define the two similar triangles
def Triangle1 : Type := Unit
def Triangle2 : Type := Unit

-- Define the height ratio
def height_ratio : ℚ := 2 / 3

-- Define the sum of perimeters
def total_perimeter : ℝ := 50

-- Define the perimeters of the two triangles
def perimeter1 : ℝ := 20
def perimeter2 : ℝ := 30

-- Theorem statement
theorem similar_triangles_perimeter :
  (perimeter1 / perimeter2 = height_ratio) ∧
  (perimeter1 + perimeter2 = total_perimeter) :=
sorry

end similar_triangles_perimeter_l3105_310551


namespace age_ratio_l3105_310526

theorem age_ratio (sachin_age rahul_age : ℕ) : 
  sachin_age = 49 → 
  rahul_age = sachin_age + 14 → 
  (sachin_age : ℚ) / rahul_age = 7 / 9 := by
sorry

end age_ratio_l3105_310526


namespace min_values_xy_and_x_plus_y_l3105_310538

theorem min_values_xy_and_x_plus_y (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (hxy : x * y - x - y = 3) :
  (∃ (m : ℝ), m = 9 ∧ ∀ z w, z > 0 → w > 0 → z * w - z - w = 3 → x * y ≤ z * w) ∧
  (∃ (n : ℝ), n = 6 ∧ ∀ z w, z > 0 → w > 0 → z * w - z - w = 3 → x + y ≤ z + w) :=
by sorry

end min_values_xy_and_x_plus_y_l3105_310538


namespace map_to_actual_distance_l3105_310530

/-- Given a map distance between two towns and a scale factor, calculate the actual distance -/
theorem map_to_actual_distance 
  (map_distance : ℝ) 
  (scale_factor : ℝ) 
  (h1 : map_distance = 45) 
  (h2 : scale_factor = 10) : 
  map_distance * scale_factor = 450 := by
  sorry

end map_to_actual_distance_l3105_310530


namespace remainder_problem_l3105_310544

theorem remainder_problem (N : ℤ) : 
  N % 899 = 63 → N % 29 = 10 := by sorry

end remainder_problem_l3105_310544
