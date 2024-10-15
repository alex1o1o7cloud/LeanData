import Mathlib

namespace NUMINAMATH_CALUDE_second_year_compound_interest_l946_94675

/-- Represents the compound interest for a given year -/
def CompoundInterest (principal : ℝ) (rate : ℝ) (year : ℕ) : ℝ :=
  principal * (1 + rate) ^ year - principal

/-- Theorem stating that given a 5% interest rate and a third-year compound interest of $1260,
    the second-year compound interest is $1200 -/
theorem second_year_compound_interest
  (principal : ℝ)
  (h1 : CompoundInterest principal 0.05 3 = 1260)
  (h2 : principal > 0) :
  CompoundInterest principal 0.05 2 = 1200 := by
  sorry


end NUMINAMATH_CALUDE_second_year_compound_interest_l946_94675


namespace NUMINAMATH_CALUDE_rectangular_plot_length_difference_l946_94637

theorem rectangular_plot_length_difference (length breadth perimeter : ℝ) : 
  length = 63 ∧ 
  perimeter = 200 ∧ 
  perimeter = 2 * (length + breadth) → 
  length - breadth = 26 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_difference_l946_94637


namespace NUMINAMATH_CALUDE_friend_distribution_l946_94654

/-- The number of ways to distribute n distinguishable items among k categories -/
def distribute (n k : ℕ) : ℕ := k ^ n

/-- The number of friends to be distributed -/
def num_friends : ℕ := 8

/-- The number of clubs available -/
def num_clubs : ℕ := 4

theorem friend_distribution :
  distribute num_friends num_clubs = 65536 := by
  sorry

end NUMINAMATH_CALUDE_friend_distribution_l946_94654


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l946_94666

theorem sum_of_roots_quadratic (a b : ℝ) : 
  (a^2 - a - 2 = 0) → (b^2 - b - 2 = 0) → (a + b = 1) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l946_94666


namespace NUMINAMATH_CALUDE_blueberries_for_pint_of_jam_l946_94632

/-- The number of blueberries needed to make a pint of blueberry jam -/
def blueberries_per_pint (blueberries_for_pies : ℕ) (num_pies : ℕ) : ℕ :=
  blueberries_for_pies / (num_pies * 2)

/-- Theorem stating the number of blueberries needed for a pint of jam -/
theorem blueberries_for_pint_of_jam :
  blueberries_per_pint 2400 6 = 200 := by
  sorry

end NUMINAMATH_CALUDE_blueberries_for_pint_of_jam_l946_94632


namespace NUMINAMATH_CALUDE_unique_positive_integer_solution_l946_94640

theorem unique_positive_integer_solution : ∃! (x : ℕ), x > 0 ∧ (4 * x)^2 + 2 * x = 3528 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_integer_solution_l946_94640


namespace NUMINAMATH_CALUDE_initial_games_eq_sum_l946_94638

/-- Represents the number of video games Cody had initially -/
def initial_games : ℕ := 9

/-- Represents the number of video games Cody gave away -/
def games_given_away : ℕ := 4

/-- Represents the number of video games Cody still has -/
def games_remaining : ℕ := 5

/-- Theorem stating that the initial number of games equals the sum of games given away and games remaining -/
theorem initial_games_eq_sum : initial_games = games_given_away + games_remaining := by
  sorry

end NUMINAMATH_CALUDE_initial_games_eq_sum_l946_94638


namespace NUMINAMATH_CALUDE_ShortestDistance_l946_94647

/-- Line1 represents the first line (1, 2, 3) + u(1, 1, 2) -/
def Line1 (u : ℝ) : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => u + 1
  | 1 => u + 2
  | 2 => 2*u + 3

/-- Line2 represents the second line (2, 4, 0) + v(2, -1, 1) -/
def Line2 (v : ℝ) : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 2*v + 2
  | 1 => -v + 4
  | 2 => v

/-- DistanceSquared calculates the squared distance between two points on the lines -/
def DistanceSquared (u v : ℝ) : ℝ :=
  (Line1 u 0 - Line2 v 0)^2 + (Line1 u 1 - Line2 v 1)^2 + (Line1 u 2 - Line2 v 2)^2

/-- ShortestDistance states that the minimum value of the square root of DistanceSquared is √5 -/
theorem ShortestDistance : 
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 5 ∧ 
  ∀ (u v : ℝ), Real.sqrt (DistanceSquared u v) ≥ min_dist := by
  sorry

end NUMINAMATH_CALUDE_ShortestDistance_l946_94647


namespace NUMINAMATH_CALUDE_max_product_value_l946_94677

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- State the theorem
theorem max_product_value :
  (∀ x, -7 ≤ f x ∧ f x ≤ 4) →
  (∀ x, -3 ≤ g x ∧ g x ≤ 2) →
  (∃ x, f x * g x = 21) ∧
  (∀ x, f x * g x ≤ 21) :=
sorry

end NUMINAMATH_CALUDE_max_product_value_l946_94677


namespace NUMINAMATH_CALUDE_angle_with_complement_40percent_of_supplement_l946_94619

theorem angle_with_complement_40percent_of_supplement (x : ℝ) : 
  (90 - x = 0.4 * (180 - x)) → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_angle_with_complement_40percent_of_supplement_l946_94619


namespace NUMINAMATH_CALUDE_second_markdown_percentage_l946_94649

theorem second_markdown_percentage 
  (original_price : ℝ) 
  (first_markdown_percentage : ℝ) 
  (second_markdown_percentage : ℝ) 
  (h1 : first_markdown_percentage = 10)
  (h2 : (1 - first_markdown_percentage / 100) * (1 - second_markdown_percentage / 100) * original_price = 0.81 * original_price) :
  second_markdown_percentage = 10 := by
sorry

end NUMINAMATH_CALUDE_second_markdown_percentage_l946_94649


namespace NUMINAMATH_CALUDE_sunday_school_average_class_size_l946_94688

/-- The average class size in a Sunday school with two classes -/
theorem sunday_school_average_class_size 
  (three_year_olds : ℕ) 
  (four_year_olds : ℕ) 
  (five_year_olds : ℕ) 
  (six_year_olds : ℕ) 
  (h1 : three_year_olds = 13)
  (h2 : four_year_olds = 20)
  (h3 : five_year_olds = 15)
  (h4 : six_year_olds = 22) :
  (three_year_olds + four_year_olds + five_year_olds + six_year_olds) / 2 = 35 := by
  sorry

#check sunday_school_average_class_size

end NUMINAMATH_CALUDE_sunday_school_average_class_size_l946_94688


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l946_94682

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : a > 0
  pos_b : b > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- The right focus of a hyperbola -/
def right_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- Represents a point on the hyperbola -/
structure HyperbolaPoint (h : Hyperbola a b) where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2 / a^2 - y^2 / b^2 = 1

/-- Theorem: Eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (h : Hyperbola a b) 
  (P Q : HyperbolaPoint h) (F₁ : ℝ × ℝ) :
  (∃ (line : ℝ → ℝ × ℝ), 
    line 0 = right_focus h ∧ 
    (∃ t₁ t₂, line t₁ = (P.x, P.y) ∧ line t₂ = (Q.x, Q.y)) ∧
    ((P.x - Q.x) * (P.x - F₁.1) + (P.y - Q.y) * (P.y - F₁.2) = 0) ∧
    ((P.x - Q.x)^2 + (P.y - Q.y)^2 = (P.x - F₁.1)^2 + (P.y - F₁.2)^2)) →
  eccentricity h = Real.sqrt (5 - 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l946_94682


namespace NUMINAMATH_CALUDE_tom_last_year_games_l946_94642

/-- Represents the number of hockey games Tom attended in various scenarios -/
structure HockeyGames where
  this_year : ℕ
  missed_this_year : ℕ
  total_two_years : ℕ

/-- Calculates the number of hockey games Tom attended last year -/
def games_last_year (g : HockeyGames) : ℕ :=
  g.total_two_years - g.this_year

/-- Theorem stating that Tom attended 9 hockey games last year -/
theorem tom_last_year_games (g : HockeyGames) 
  (h1 : g.this_year = 4)
  (h2 : g.missed_this_year = 7)
  (h3 : g.total_two_years = 13) :
  games_last_year g = 9 := by
  sorry


end NUMINAMATH_CALUDE_tom_last_year_games_l946_94642


namespace NUMINAMATH_CALUDE_ticket_order_solution_l946_94660

/-- Represents the ticket order information -/
structure TicketOrder where
  childPrice : ℚ
  adultPrice : ℚ
  discountThreshold : ℕ
  discountRate : ℚ
  childrenExcess : ℕ
  totalBill : ℚ

/-- Calculates the number of adult and children tickets -/
def calculateTickets (order : TicketOrder) : ℕ × ℕ :=
  sorry

/-- Checks if the discount was applied -/
def wasDiscountApplied (order : TicketOrder) (adultTickets childTickets : ℕ) : Bool :=
  sorry

theorem ticket_order_solution (order : TicketOrder)
    (h1 : order.childPrice = 7.5)
    (h2 : order.adultPrice = 12)
    (h3 : order.discountThreshold = 20)
    (h4 : order.discountRate = 0.1)
    (h5 : order.childrenExcess = 8)
    (h6 : order.totalBill = 138) :
    let (adultTickets, childTickets) := calculateTickets order
    adultTickets = 4 ∧ childTickets = 12 ∧ ¬wasDiscountApplied order adultTickets childTickets :=
  sorry

end NUMINAMATH_CALUDE_ticket_order_solution_l946_94660


namespace NUMINAMATH_CALUDE_probability_factor_less_than_seven_l946_94600

def factors (n : ℕ) : Finset ℕ :=
  Finset.filter (· ∣ n) (Finset.range (n + 1))

theorem probability_factor_less_than_seven :
  let all_factors := factors 60
  let factors_less_than_seven := all_factors.filter (· < 7)
  (factors_less_than_seven.card : ℚ) / all_factors.card = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_probability_factor_less_than_seven_l946_94600


namespace NUMINAMATH_CALUDE_ac_over_b_squared_eq_one_l946_94634

/-- A quadratic equation with real coefficients -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  has_imaginary_roots : ∃ (x₁ x₂ : ℂ), a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 ∧ x₁ ≠ x₂ ∧ x₁.im ≠ 0 ∧ x₂.im ≠ 0
  x₁_cubed_real : ∃ (x₁ : ℂ), (a * x₁^2 + b * x₁ + c = 0) ∧ (∃ (r : ℝ), x₁^3 = r)

/-- Theorem stating that ac/b^2 = 1 for a quadratic equation satisfying the given conditions -/
theorem ac_over_b_squared_eq_one (eq : QuadraticEquation) : eq.a * eq.c / eq.b^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ac_over_b_squared_eq_one_l946_94634


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l946_94615

theorem absolute_value_inequality (a : ℝ) : 
  (∃ x : ℝ, |x - a| + |x + 1| < 2) ↔ (-3 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l946_94615


namespace NUMINAMATH_CALUDE_problem_solution_l946_94627

theorem problem_solution :
  let M : ℕ := 3009 / 3
  let N : ℕ := (2 * M) / 3
  let X : ℤ := M - N
  X = 335 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l946_94627


namespace NUMINAMATH_CALUDE_e_percentage_of_d_l946_94631

-- Define the variables
variable (a b c d e : ℝ)

-- Define the relationships between the variables
def relationship_d : Prop := d = 0.4 * a ∧ d = 0.35 * b
def relationship_e : Prop := e = 0.5 * b ∧ e = 0.2 * c
def relationship_c : Prop := c = 0.3 * a ∧ c = 0.25 * b

-- Theorem statement
theorem e_percentage_of_d 
  (hd : relationship_d a b d)
  (he : relationship_e b c e)
  (hc : relationship_c a b c) :
  e / d = 0.15 := by sorry

end NUMINAMATH_CALUDE_e_percentage_of_d_l946_94631


namespace NUMINAMATH_CALUDE_evaluate_expression_l946_94697

theorem evaluate_expression (a : ℝ) : 
  let x := a + 9
  (x - a + 5) = 14 := by
sorry

end NUMINAMATH_CALUDE_evaluate_expression_l946_94697


namespace NUMINAMATH_CALUDE_gain_percentage_cloth_sale_l946_94695

/-- Calculates the gain percentage given the total quantity sold and the profit quantity -/
def gainPercentage (totalQuantity : ℕ) (profitQuantity : ℕ) : ℚ :=
  (profitQuantity : ℚ) / (totalQuantity : ℚ)

/-- Theorem: The gain percentage is 1/6 when selling 60 meters of cloth and gaining the selling price of 10 meters as profit -/
theorem gain_percentage_cloth_sale : 
  gainPercentage 60 10 = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_gain_percentage_cloth_sale_l946_94695


namespace NUMINAMATH_CALUDE_complex_equation_solution_l946_94623

theorem complex_equation_solution :
  ∃ x : ℤ, x - (28 - (37 - (15 - 17))) = 56 ∧ x = 45 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l946_94623


namespace NUMINAMATH_CALUDE_exists_ratio_eq_rational_l946_94630

def u : ℕ → ℚ
  | 0 => 1
  | n + 1 => if n % 2 = 0 then u (n / 2) + u ((n - 1) / 2) else u (n / 2)

theorem exists_ratio_eq_rational (k : ℚ) (hk : k > 0) :
  ∃ n : ℕ, u n / u (n + 1) = k :=
by sorry

end NUMINAMATH_CALUDE_exists_ratio_eq_rational_l946_94630


namespace NUMINAMATH_CALUDE_geometric_sequence_y_value_l946_94658

def is_geometric_sequence (a b c d e : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r ∧ e = d * r

theorem geometric_sequence_y_value (x y z : ℝ) :
  is_geometric_sequence 1 x y z 9 → y = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_y_value_l946_94658


namespace NUMINAMATH_CALUDE_integer_solution_for_equation_l946_94604

theorem integer_solution_for_equation (a b c : ℤ) :
  (∀ x : ℤ, (x - a) * (x - 15) + 1 = (x + b) * (x + c)) →
  (a = 13 ∨ a = 17) := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_for_equation_l946_94604


namespace NUMINAMATH_CALUDE_complex_number_equality_l946_94674

theorem complex_number_equality (z : ℂ) (h : z * Complex.I = 2 - 2 * Complex.I) : z = -2 - 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l946_94674


namespace NUMINAMATH_CALUDE_johny_east_south_difference_l946_94643

/-- Represents Johny's travel distances in different directions -/
structure TravelDistances where
  south : ℝ
  east : ℝ
  north : ℝ

/-- Johny's travel conditions -/
def johny_travel : TravelDistances → Prop :=
  λ d => d.south = 40 ∧
         d.east > d.south ∧
         d.north = 2 * d.east ∧
         d.south + d.east + d.north = 220

/-- The theorem to prove -/
theorem johny_east_south_difference (d : TravelDistances) 
  (h : johny_travel d) : d.east - d.south = 40 :=
by
  sorry


end NUMINAMATH_CALUDE_johny_east_south_difference_l946_94643


namespace NUMINAMATH_CALUDE_a_4_equals_zero_l946_94698

/-- Given a sequence {aₙ} with general term aₙ = n² - 3n - 4 for n ∈ ℕ*, prove that a₄ = 0 -/
theorem a_4_equals_zero (a : ℕ+ → ℤ) (h : ∀ n : ℕ+, a n = n.val ^ 2 - 3 * n.val - 4) :
  a 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_a_4_equals_zero_l946_94698


namespace NUMINAMATH_CALUDE_prime_divides_sum_of_squares_l946_94665

theorem prime_divides_sum_of_squares (p a b : ℤ) : 
  Prime p → p % 4 = 3 → (a^2 + b^2) % p = 0 → p ∣ a ∧ p ∣ b := by
  sorry

end NUMINAMATH_CALUDE_prime_divides_sum_of_squares_l946_94665


namespace NUMINAMATH_CALUDE_original_number_proof_l946_94689

theorem original_number_proof (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 5 * (1 / x)) :
  x = Real.sqrt 2 / 20 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l946_94689


namespace NUMINAMATH_CALUDE_money_distribution_l946_94629

theorem money_distribution (total : ℕ) (vasim_share : ℕ) : 
  vasim_share = 1500 →
  ∃ (faruk_share ranjith_share : ℕ),
    faruk_share + vasim_share + ranjith_share = total ∧
    5 * faruk_share = 3 * vasim_share ∧
    6 * faruk_share = 3 * ranjith_share ∧
    ranjith_share - faruk_share = 900 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l946_94629


namespace NUMINAMATH_CALUDE_power_product_equality_l946_94684

theorem power_product_equality : 3^5 * 7^5 = 4084101 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l946_94684


namespace NUMINAMATH_CALUDE_find_2a_plus_b_l946_94657

-- Define the functions
def f (a b x : ℝ) : ℝ := 2 * a * x - 3 * b
def g (x : ℝ) : ℝ := 5 * x + 4
def h (a b x : ℝ) : ℝ := g (f a b x)

-- State the theorem
theorem find_2a_plus_b (a b : ℝ) :
  (∀ x, h a b (2 * x - 9) = x) →
  2 * a + b = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_find_2a_plus_b_l946_94657


namespace NUMINAMATH_CALUDE_curve_is_hyperbola_with_foci_on_x_axis_l946_94692

/-- The curve represented by the equation x²/(sin θ + 3) + y²/(sin θ - 2) = 1 -/
def curve (x y θ : ℝ) : Prop :=
  x^2 / (Real.sin θ + 3) + y^2 / (Real.sin θ - 2) = 1

/-- The curve is a hyperbola with foci on the x-axis -/
theorem curve_is_hyperbola_with_foci_on_x_axis :
  ∀ x y θ, curve x y θ → 
    (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ x^2/a^2 - y^2/b^2 = 1) ∧ 
    (∃ c : ℝ, c > 0 ∧ (∃ f₁ f₂ : ℝ × ℝ, f₁.1 = c ∧ f₁.2 = 0 ∧ f₂.1 = -c ∧ f₂.2 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_curve_is_hyperbola_with_foci_on_x_axis_l946_94692


namespace NUMINAMATH_CALUDE_system_solution_l946_94616

theorem system_solution : ∃ (x y : ℝ), x + y = 8 ∧ x - 3*y = 4 ∧ x = 7 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l946_94616


namespace NUMINAMATH_CALUDE_intersection_when_m_3_union_equals_A_iff_l946_94686

-- Define sets A and B
def A : Set ℝ := {x | |x - 1| < 2}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 1 < 0}

-- Theorem for part 1
theorem intersection_when_m_3 : 
  A ∩ B 3 = {x | 2 < x ∧ x < 3} := by sorry

-- Theorem for part 2
theorem union_equals_A_iff (m : ℝ) : 
  A ∪ B m = A ↔ 0 ≤ m ∧ m ≤ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_3_union_equals_A_iff_l946_94686


namespace NUMINAMATH_CALUDE_calculate_expression_l946_94664

theorem calculate_expression (y : ℝ) (h : y ≠ 0) :
  (18 * y^3) * (9 * y^2) * (1 / (6 * y)^3) = (3 / 4) * y^2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l946_94664


namespace NUMINAMATH_CALUDE_apple_products_total_cost_l946_94611

/-- Calculates the total cost of an iPhone and iWatch after discounts and cashback -/
theorem apple_products_total_cost 
  (iphone_price : ℝ) 
  (iwatch_price : ℝ) 
  (iphone_discount : ℝ) 
  (iwatch_discount : ℝ) 
  (cashback_rate : ℝ) 
  (h1 : iphone_price = 800) 
  (h2 : iwatch_price = 300) 
  (h3 : iphone_discount = 0.15) 
  (h4 : iwatch_discount = 0.10) 
  (h5 : cashback_rate = 0.02) : 
  ℝ := by
  sorry

#check apple_products_total_cost

end NUMINAMATH_CALUDE_apple_products_total_cost_l946_94611


namespace NUMINAMATH_CALUDE_speed_ratio_A_to_B_l946_94656

-- Define the work completion rates for A and B
def work_rate_B : ℚ := 1 / 12
def work_rate_A_and_B : ℚ := 1 / 4

-- Define A's work rate in terms of B's
def work_rate_A : ℚ := work_rate_A_and_B - work_rate_B

-- Theorem statement
theorem speed_ratio_A_to_B : 
  work_rate_A / work_rate_B = 2 := by
  sorry

end NUMINAMATH_CALUDE_speed_ratio_A_to_B_l946_94656


namespace NUMINAMATH_CALUDE_remainder_theorem_l946_94670

/-- The polynomial P(x) = 5x^4 - 13x^3 + 3x^2 - x + 15 -/
def P (x : ℝ) : ℝ := 5*x^4 - 13*x^3 + 3*x^2 - x + 15

/-- The divisor polynomial d(x) = 3x - 9 -/
def d (x : ℝ) : ℝ := 3*x - 9

/-- Theorem stating that the remainder when P(x) is divided by d(x) is 93 -/
theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ x, P x = d x * q x + 93 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l946_94670


namespace NUMINAMATH_CALUDE_seats_formula_l946_94639

/-- The number of seats in the n-th row of a cinema -/
def seats (n : ℕ) : ℕ :=
  18 + 3 * (n - 1)

/-- Theorem: The number of seats in the n-th row is 3n + 15 -/
theorem seats_formula (n : ℕ) : seats n = 3 * n + 15 := by
  sorry

end NUMINAMATH_CALUDE_seats_formula_l946_94639


namespace NUMINAMATH_CALUDE_barn_painting_area_l946_94659

theorem barn_painting_area (width length height : ℝ) 
  (h_width : width = 10)
  (h_length : length = 13)
  (h_height : height = 5) :
  2 * (width * height + length * height) + width * length = 590 :=
by sorry

end NUMINAMATH_CALUDE_barn_painting_area_l946_94659


namespace NUMINAMATH_CALUDE_real_estate_problem_l946_94652

-- Define the constants
def total_sets : ℕ := 80
def cost_A : ℕ := 90
def price_A : ℕ := 102
def cost_B : ℕ := 60
def price_B : ℕ := 70
def min_funds : ℕ := 5700
def max_A : ℕ := 32

-- Define the variables
variable (x : ℕ) -- number of Type A sets
variable (W : ℕ → ℕ) -- profit function
variable (a : ℚ) -- price reduction for Type A

-- Define the theorem
theorem real_estate_problem :
  (∀ x, W x = 2 * x + 800) ∧
  (x ≥ 30 ∧ x ≤ 32) ∧
  (∀ a, 0 < a ∧ a ≤ 3 →
    (0 < a ∧ a < 2 → x = 32) ∧
    (a = 2 → true) ∧
    (2 < a ∧ a ≤ 3 → x = 30)) :=
sorry

end NUMINAMATH_CALUDE_real_estate_problem_l946_94652


namespace NUMINAMATH_CALUDE_sum_of_right_angles_l946_94641

/-- A rectangle has 4 right angles -/
def rectangle_right_angles : ℕ := 4

/-- A square has 4 right angles -/
def square_right_angles : ℕ := 4

/-- The sum of right angles in a rectangle and a square -/
def total_right_angles : ℕ := rectangle_right_angles + square_right_angles

theorem sum_of_right_angles : total_right_angles = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_right_angles_l946_94641


namespace NUMINAMATH_CALUDE_number_problem_l946_94667

theorem number_problem : ∃ x : ℚ, 34 + 3 * x = 49 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l946_94667


namespace NUMINAMATH_CALUDE_average_daily_attendance_l946_94650

def monday_attendance : ℕ := 10
def tuesday_attendance : ℕ := 15
def wednesday_to_friday_attendance : ℕ := 10
def total_days : ℕ := 5

def total_attendance : ℕ := monday_attendance + tuesday_attendance + 3 * wednesday_to_friday_attendance

theorem average_daily_attendance :
  (total_attendance : ℚ) / total_days = 11 := by sorry

end NUMINAMATH_CALUDE_average_daily_attendance_l946_94650


namespace NUMINAMATH_CALUDE_proportion_solution_l946_94622

theorem proportion_solution (x y : ℝ) : 
  (0.75 : ℝ) / x = 5 / y ∧ x = 1.65 → y = 11 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l946_94622


namespace NUMINAMATH_CALUDE_logarithmic_equation_solution_l946_94633

theorem logarithmic_equation_solution (x : ℝ) : 
  x > 0 → 
  (7.3113 * (Real.log 4 / Real.log x) + 
   2 * (Real.log 4 / Real.log (4 * x)) + 
   3 * (Real.log 4 / Real.log (16 * x)) = 0) ↔ 
  (x = 1/2 ∨ x = 1/8) := by
  sorry

end NUMINAMATH_CALUDE_logarithmic_equation_solution_l946_94633


namespace NUMINAMATH_CALUDE_gardener_tree_rows_l946_94610

/-- Proves that the initial number of rows is 24 given the gardener's tree planting conditions -/
theorem gardener_tree_rows : ∀ r : ℕ, 
  (42 * r = 28 * (r + 12)) → r = 24 := by
  sorry

end NUMINAMATH_CALUDE_gardener_tree_rows_l946_94610


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l946_94676

/-- A geometric sequence with specific properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) ∧
  a 2 * a 3 = 5 ∧
  a 5 * a 6 = 10

/-- Theorem stating the property of the 8th and 9th terms -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 8 * a 9 = 20 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l946_94676


namespace NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l946_94644

theorem largest_n_satisfying_inequality : 
  ∀ n : ℤ, (1/4 : ℚ) + (n : ℚ)/6 < 3/2 ↔ n ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l946_94644


namespace NUMINAMATH_CALUDE_max_payment_is_31_l946_94671

/-- Represents a four-digit number of the form 20** -/
def FourDigitNumber := { n : ℕ // 2000 ≤ n ∧ n ≤ 2099 }

/-- Calculates the payment for a given divisor -/
def payment (d : ℕ) : ℕ :=
  match d with
  | 1 => 1
  | 3 => 3
  | 5 => 5
  | 7 => 7
  | 9 => 9
  | 11 => 11
  | _ => 0

/-- Calculates the total payment for a number based on its divisibility -/
def totalPayment (n : FourDigitNumber) : ℕ :=
  (payment 1) +
  (if n.val % 3 = 0 then payment 3 else 0) +
  (if n.val % 5 = 0 then payment 5 else 0) +
  (if n.val % 7 = 0 then payment 7 else 0) +
  (if n.val % 9 = 0 then payment 9 else 0) +
  (if n.val % 11 = 0 then payment 11 else 0)

theorem max_payment_is_31 :
  ∃ (n : FourDigitNumber), totalPayment n = 31 ∧
  ∀ (m : FourDigitNumber), totalPayment m ≤ 31 :=
sorry

end NUMINAMATH_CALUDE_max_payment_is_31_l946_94671


namespace NUMINAMATH_CALUDE_inequality_proof_l946_94694

theorem inequality_proof (a b c d p q : ℝ) 
  (h1 : a * b + c * d = 2 * p * q) 
  (h2 : a * c ≥ p^2) 
  (h3 : p^2 > 0) : 
  b * d ≤ q^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l946_94694


namespace NUMINAMATH_CALUDE_repeating_decimal_problem_l946_94603

theorem repeating_decimal_problem (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) :
  72 * ((1 + (100 * a + 10 * b + c : ℕ) / 999 : ℚ) - (1 + (a / 10 + b / 100 + c / 1000 : ℚ))) = (3 / 5 : ℚ) →
  100 * a + 10 * b + c = 833 := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_problem_l946_94603


namespace NUMINAMATH_CALUDE_max_digits_product_four_digit_numbers_l946_94624

theorem max_digits_product_four_digit_numbers :
  ∀ a b : ℕ, 1000 ≤ a ∧ a ≤ 9999 → 1000 ≤ b ∧ b ≤ 9999 →
  ∃ n : ℕ, n ≤ 8 ∧ a * b < 10^n :=
by sorry

end NUMINAMATH_CALUDE_max_digits_product_four_digit_numbers_l946_94624


namespace NUMINAMATH_CALUDE_product_equals_fraction_l946_94655

/-- The decimal representation of a real number with digits 1, 4, 5 repeating after the decimal point -/
def repeating_decimal : ℚ := 145 / 999

/-- The product of the repeating decimal and 11 -/
def product : ℚ := 11 * repeating_decimal

theorem product_equals_fraction : product = 1595 / 999 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_fraction_l946_94655


namespace NUMINAMATH_CALUDE_boys_can_be_truthful_l946_94601

/-- Represents the possible grades a student can receive -/
inductive Grade
  | Three
  | Four
  | Five

/-- Compares two grades -/
def Grade.gt (a b : Grade) : Prop :=
  match a, b with
  | Five, Three => True
  | Five, Four => True
  | Four, Three => True
  | _, _ => False

/-- Represents the grades of a student for three tests -/
structure StudentGrades :=
  (test1 : Grade)
  (test2 : Grade)
  (test3 : Grade)

/-- Checks if one student has higher grades than another for at least two tests -/
def higherGradesInTwoTests (a b : StudentGrades) : Prop :=
  (a.test1.gt b.test1 ∧ a.test2.gt b.test2) ∨
  (a.test1.gt b.test1 ∧ a.test3.gt b.test3) ∨
  (a.test2.gt b.test2 ∧ a.test3.gt b.test3)

/-- The main theorem stating that there exists a set of grades satisfying all conditions -/
theorem boys_can_be_truthful :
  ∃ (valera seryozha dima : StudentGrades),
    higherGradesInTwoTests valera seryozha ∧
    higherGradesInTwoTests seryozha dima ∧
    higherGradesInTwoTests dima valera :=
  sorry

end NUMINAMATH_CALUDE_boys_can_be_truthful_l946_94601


namespace NUMINAMATH_CALUDE_intersection_distance_squared_is_96_l946_94663

/-- The square of the distance between intersection points of two circles -/
def intersection_distance_squared (c1_center c2_center : ℝ × ℝ) (r1 r2 : ℝ) : ℝ :=
  let (x1, y1) := c1_center
  let (x2, y2) := c2_center
  -- Definition of the function, to be implemented
  0

/-- The theorem stating the square of the distance between intersection points -/
theorem intersection_distance_squared_is_96 :
  intersection_distance_squared (3, 2) (3, -4) 5 7 = 96 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_squared_is_96_l946_94663


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l946_94636

theorem expression_simplification_and_evaluation :
  let x : ℝ := 6 * Real.sin (30 * π / 180) - Real.sqrt 2 * Real.cos (45 * π / 180)
  ((x / (x - 2) - x / (x + 2)) / (4 * x / (x - 2))) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l946_94636


namespace NUMINAMATH_CALUDE_julia_parrot_weeks_l946_94646

/-- Represents the problem of determining how long Julia has had her parrot -/
theorem julia_parrot_weeks : 
  ∀ (total_weekly_cost rabbit_weekly_cost total_spent rabbit_weeks : ℕ),
  total_weekly_cost = 30 →
  rabbit_weekly_cost = 12 →
  rabbit_weeks = 5 →
  total_spent = 114 →
  ∃ (parrot_weeks : ℕ),
    parrot_weeks * (total_weekly_cost - rabbit_weekly_cost) = 
      total_spent - (rabbit_weeks * rabbit_weekly_cost) ∧
    parrot_weeks = 3 :=
by sorry

end NUMINAMATH_CALUDE_julia_parrot_weeks_l946_94646


namespace NUMINAMATH_CALUDE_total_distinct_plants_l946_94618

-- Define the flower beds as finite sets
variable (A B C D : Finset ℕ)

-- Define the cardinalities of the sets
variable (hA : A.card = 550)
variable (hB : B.card = 500)
variable (hC : C.card = 400)
variable (hD : D.card = 350)

-- Define the intersections
variable (hAB : (A ∩ B).card = 60)
variable (hAC : (A ∩ C).card = 110)
variable (hAD : (A ∩ D).card = 70)
variable (hABC : (A ∩ B ∩ C).card = 30)

-- Define the empty intersections
variable (hBC : (B ∩ C).card = 0)
variable (hBD : (B ∩ D).card = 0)

-- State the theorem
theorem total_distinct_plants :
  (A ∪ B ∪ C ∪ D).card = 1590 :=
sorry

end NUMINAMATH_CALUDE_total_distinct_plants_l946_94618


namespace NUMINAMATH_CALUDE_max_value_on_curve_l946_94696

theorem max_value_on_curve :
  ∀ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 4 →
  ∃ M : ℝ, M = 17 ∧ ∀ x' y' : ℝ, (x' - 1)^2 + (y' - 1)^2 = 4 → 3*x' + 4*y' ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_curve_l946_94696


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l946_94626

/-- Definition of a geometric progression for three real numbers -/
def is_geometric_progression (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r

/-- The condition b^2 = ac -/
def condition (a b c : ℝ) : Prop := b^2 = a * c

/-- Theorem stating that the condition is necessary but not sufficient -/
theorem condition_necessary_not_sufficient :
  (∀ a b c : ℝ, is_geometric_progression a b c → condition a b c) ∧
  ¬(∀ a b c : ℝ, condition a b c → is_geometric_progression a b c) :=
sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l946_94626


namespace NUMINAMATH_CALUDE_more_karabases_than_barabases_l946_94625

/-- Represents the inhabitants of Perra-Terra -/
inductive Inhabitant
| Karabas
| Barabas

/-- The number of acquaintances each type of inhabitant has -/
def acquaintances (i : Inhabitant) : Nat × Nat :=
  match i with
  | Inhabitant.Karabas => (6, 9)  -- (Other Karabases, Barabases)
  | Inhabitant.Barabas => (10, 7) -- (Karabases, Other Barabases)

theorem more_karabases_than_barabases (K B : Nat) 
  (hK : K > 0) (hB : B > 0) 
  (h_acquaintances : K * (acquaintances Inhabitant.Karabas).2 = B * (acquaintances Inhabitant.Barabas).1) :
  K > B := by
  sorry

#check more_karabases_than_barabases

end NUMINAMATH_CALUDE_more_karabases_than_barabases_l946_94625


namespace NUMINAMATH_CALUDE_geometric_sequence_min_sum_l946_94672

/-- A positive geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_min_sum
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : GeometricSequence a)
  (h_prod : a 3 * a 5 = 64) :
  ∃ (min : ℝ), min = 16 ∧ ∀ (a' : ℕ → ℝ),
    GeometricSequence a' → (∀ n, a' n > 0) → a' 3 * a' 5 = 64 →
    a' 1 + a' 7 ≥ min :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_min_sum_l946_94672


namespace NUMINAMATH_CALUDE_cookie_difference_l946_94653

/-- Proves that the difference between the number of cookies in 8 boxes and 9 bags is 33,
    given that each box contains 12 cookies and each bag contains 7 cookies. -/
theorem cookie_difference :
  let cookies_per_box : ℕ := 12
  let cookies_per_bag : ℕ := 7
  let num_boxes : ℕ := 8
  let num_bags : ℕ := 9
  (num_boxes * cookies_per_box) - (num_bags * cookies_per_bag) = 33 := by
  sorry

end NUMINAMATH_CALUDE_cookie_difference_l946_94653


namespace NUMINAMATH_CALUDE_problem_solution_l946_94645

theorem problem_solution : (42 / (9 - 3 * 2)) * 4 = 56 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l946_94645


namespace NUMINAMATH_CALUDE_mutually_exclusive_not_contradictory_l946_94691

structure Ball :=
  (color : String)

def Bag : Finset Ball := sorry

axiom bag_composition : 
  (Bag.filter (λ b => b.color = "red")).card = 2 ∧ 
  (Bag.filter (λ b => b.color = "black")).card = 2

def Draw : Finset Ball := sorry

axiom draw_size : Draw.card = 2

def exactly_one_black : Prop :=
  (Draw.filter (λ b => b.color = "black")).card = 1

def exactly_two_black : Prop :=
  (Draw.filter (λ b => b.color = "black")).card = 2

theorem mutually_exclusive_not_contradictory :
  (¬(exactly_one_black ∧ exactly_two_black)) ∧
  (∃ draw : Finset Ball, draw.card = 2 ∧ ¬exactly_one_black ∧ ¬exactly_two_black) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_not_contradictory_l946_94691


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l946_94621

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2
  arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- Main theorem: If S_6 / S_3 = 4 for an arithmetic sequence, then S_9 / S_6 = 9/4 -/
theorem arithmetic_sequence_sum_ratio 
  (seq : ArithmeticSequence) 
  (h : seq.S 6 / seq.S 3 = 4) : 
  seq.S 9 / seq.S 6 = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l946_94621


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l946_94612

/-- Given vectors in ℝ² -/
def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (2, -1)

/-- Function to check if two vectors are parallel -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), v.1 * w.2 = t * v.2 * w.1

/-- Main theorem -/
theorem parallel_vectors_k_value :
  ∃ (k : ℝ), are_parallel (a.1 + k * c.1, a.2 + k * c.2) (2 * b.1 - a.1, 2 * b.2 - a.2) ∧ k = 16 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l946_94612


namespace NUMINAMATH_CALUDE_paint_usage_proof_l946_94605

theorem paint_usage_proof (total_paint : ℝ) (second_week_fraction : ℝ) (total_used : ℝ) 
  (h1 : total_paint = 360)
  (h2 : second_week_fraction = 1/6)
  (h3 : total_used = 135) :
  ∃ (first_week_fraction : ℝ),
    first_week_fraction * total_paint + 
    second_week_fraction * (total_paint - first_week_fraction * total_paint) = total_used ∧
    first_week_fraction = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_paint_usage_proof_l946_94605


namespace NUMINAMATH_CALUDE_inequality_solution_set_l946_94635

theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, (m + 1) * x^2 - (m - 1) * x + 3 * (m - 1) < 0) → m < -13/11 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l946_94635


namespace NUMINAMATH_CALUDE_perpendicular_distance_approx_l946_94609

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a rectangular parallelepiped -/
structure Parallelepiped where
  length : ℝ
  width : ℝ
  height : ℝ
  d : Point3D
  a : Point3D
  b : Point3D
  c : Point3D

/-- Calculates the perpendicular distance from a point to a plane defined by three points -/
def perpendicularDistance (p : Point3D) (a b c : Point3D) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem perpendicular_distance_approx (p : Parallelepiped) : 
  p.length = 5 ∧ p.width = 3 ∧ p.height = 2 ∧
  p.d = ⟨0, 0, 0⟩ ∧ p.a = ⟨5, 0, 0⟩ ∧ p.b = ⟨0, 3, 0⟩ ∧ p.c = ⟨0, 0, 2⟩ →
  abs (perpendicularDistance p.d p.a p.b p.c - 1.9) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_distance_approx_l946_94609


namespace NUMINAMATH_CALUDE_train_length_l946_94661

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (cross_time : ℝ) (h1 : speed_kmh = 90) (h2 : cross_time = 9) :
  speed_kmh * (1000 / 3600) * cross_time = 225 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l946_94661


namespace NUMINAMATH_CALUDE_lemon_problem_l946_94679

theorem lemon_problem (levi jayden eli ian : ℕ) : 
  levi = 5 →
  jayden > levi →
  jayden * 3 = eli →
  eli * 2 = ian →
  levi + jayden + eli + ian = 115 →
  jayden - levi = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_lemon_problem_l946_94679


namespace NUMINAMATH_CALUDE_missing_village_population_l946_94648

def village_count : Nat := 7
def known_populations : List Nat := [803, 900, 1100, 1023, 980, 1249]
def average_population : Nat := 1000

theorem missing_village_population :
  village_count * average_population - known_populations.sum = 945 := by
  sorry

end NUMINAMATH_CALUDE_missing_village_population_l946_94648


namespace NUMINAMATH_CALUDE_no_exact_table_count_l946_94606

theorem no_exact_table_count : ¬∃ (t : ℕ), 
  3 * (8 * t) + 4 * (2 * t) + 4 * t = 656 := by
  sorry

end NUMINAMATH_CALUDE_no_exact_table_count_l946_94606


namespace NUMINAMATH_CALUDE_tangent_line_at_point_l946_94678

-- Define the curve
def curve (x y : ℝ) : Prop := x^3 - y = 0

-- Define the point of tangency
def point : ℝ × ℝ := (-2, -8)

-- Define the proposed tangent line equation
def tangent_line (x y : ℝ) : Prop := 12*x - y + 16 = 0

-- Theorem statement
theorem tangent_line_at_point :
  ∀ x y : ℝ,
  curve x y →
  (x, y) = point →
  tangent_line x y :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_l946_94678


namespace NUMINAMATH_CALUDE_square_of_1031_l946_94608

theorem square_of_1031 : (1031 : ℕ)^2 = 1060961 := by sorry

end NUMINAMATH_CALUDE_square_of_1031_l946_94608


namespace NUMINAMATH_CALUDE_courtyard_length_l946_94699

/-- Proves that a courtyard with given dimensions and number of bricks has a specific length -/
theorem courtyard_length 
  (width : ℝ) 
  (brick_length : ℝ) 
  (brick_width : ℝ) 
  (num_bricks : ℕ) : 
  width = 16 →
  brick_length = 0.2 →
  brick_width = 0.1 →
  num_bricks = 24000 →
  (width * (num_bricks * brick_length * brick_width / width)) = 30 := by
sorry

end NUMINAMATH_CALUDE_courtyard_length_l946_94699


namespace NUMINAMATH_CALUDE_perpendicular_vectors_imply_k_zero_l946_94602

/-- Given vectors a, b, and c in ℝ², prove that if a-c is perpendicular to b, then k = 0 -/
theorem perpendicular_vectors_imply_k_zero (a b c : ℝ × ℝ) (h : a.1 = 3 ∧ a.2 = 1) 
  (h' : b.1 = 1 ∧ b.2 = 3) (h'' : c.1 = k ∧ c.2 = 2) 
  (h''' : (a.1 - c.1) * b.1 + (a.2 - c.2) * b.2 = 0) : 
  k = 0 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_imply_k_zero_l946_94602


namespace NUMINAMATH_CALUDE_least_positive_difference_l946_94614

def geometric_sequence (a₁ : ℝ) (r : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => r * geometric_sequence a₁ r n

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => arithmetic_sequence a₁ d n + d

def sequence_A : ℕ → ℝ := geometric_sequence 3 2

def sequence_B : ℕ → ℝ := arithmetic_sequence 15 30

def valid_term_A (n : ℕ) : Prop := sequence_A n ≤ 300

def valid_term_B (n : ℕ) : Prop := sequence_B n ≤ 300

theorem least_positive_difference :
  ∃ (m n : ℕ), valid_term_A m ∧ valid_term_B n ∧
    ∀ (i j : ℕ), valid_term_A i → valid_term_B j →
      |sequence_A m - sequence_B n| ≤ |sequence_A i - sequence_B j| ∧
      |sequence_A m - sequence_B n| = 3 :=
sorry

end NUMINAMATH_CALUDE_least_positive_difference_l946_94614


namespace NUMINAMATH_CALUDE_selling_price_calculation_l946_94683

/-- Calculates the selling price of an article given its cost price and profit percentage -/
def selling_price (cost_price : ℚ) (profit_percentage : ℚ) : ℚ :=
  cost_price * (1 + profit_percentage / 100)

/-- Theorem: The selling price of an article with cost price 480 and profit percentage 25% is 600 -/
theorem selling_price_calculation :
  selling_price 480 25 = 600 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l946_94683


namespace NUMINAMATH_CALUDE_cuboid_volume_with_margin_eq_l946_94607

/-- The volume of points inside or within two units of a cuboid with dimensions 5 by 6 by 8 units -/
def cuboid_volume_with_margin : ℝ := sorry

/-- The dimensions of the cuboid -/
def cuboid_dimensions : Fin 3 → ℕ
  | 0 => 5
  | 1 => 6
  | 2 => 8
  | _ => 0

/-- The margin around the cuboid -/
def margin : ℕ := 2

/-- Theorem stating that the volume of points inside or within two units of the cuboid 
    is equal to (2136 + 140π)/3 cubic units -/
theorem cuboid_volume_with_margin_eq : 
  cuboid_volume_with_margin = (2136 + 140 * Real.pi) / 3 := by sorry

end NUMINAMATH_CALUDE_cuboid_volume_with_margin_eq_l946_94607


namespace NUMINAMATH_CALUDE_pizza_slices_per_person_l946_94681

theorem pizza_slices_per_person 
  (small_pizza_slices : ℕ) 
  (large_pizza_slices : ℕ) 
  (slices_eaten_per_person : ℕ) 
  (num_people : ℕ) :
  small_pizza_slices = 8 →
  large_pizza_slices = 14 →
  slices_eaten_per_person = 9 →
  num_people = 2 →
  (small_pizza_slices + large_pizza_slices - slices_eaten_per_person * num_people) / num_people = 2 := by
sorry

end NUMINAMATH_CALUDE_pizza_slices_per_person_l946_94681


namespace NUMINAMATH_CALUDE_polynomial_factorization_sum_l946_94687

theorem polynomial_factorization_sum (a b c : ℝ) : 
  (∀ x, x^2 + 17*x + 52 = (x + a) * (x + b)) →
  (∀ x, x^2 + 7*x - 60 = (x + b) * (x - c)) →
  a + b + c = 27 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_sum_l946_94687


namespace NUMINAMATH_CALUDE_at_least_two_acute_angles_l946_94680

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_180 : angles 0 + angles 1 + angles 2 = 180
  all_positive : ∀ i, angles i > 0

-- Define an acute angle
def is_acute (angle : ℝ) : Prop := angle < 90

-- Define the theorem
theorem at_least_two_acute_angles (t : Triangle) : 
  ∃ i j, i ≠ j ∧ is_acute (t.angles i) ∧ is_acute (t.angles j) :=
sorry

end NUMINAMATH_CALUDE_at_least_two_acute_angles_l946_94680


namespace NUMINAMATH_CALUDE_lcm_ratio_implies_gcd_l946_94651

theorem lcm_ratio_implies_gcd (X Y : ℕ+) : 
  Nat.lcm X Y = 180 → X * 6 = Y * 5 → Nat.gcd X Y = 6 := by
  sorry

end NUMINAMATH_CALUDE_lcm_ratio_implies_gcd_l946_94651


namespace NUMINAMATH_CALUDE_lunch_combinations_eq_27_l946_94628

/-- Represents a category of food items in the cafeteria -/
structure FoodCategory where
  options : Finset String
  size_eq_three : options.card = 3

/-- Represents the cafeteria menu -/
structure CafeteriaMenu where
  main_dishes : FoodCategory
  beverages : FoodCategory
  snacks : FoodCategory

/-- A function to calculate the number of distinct lunch combinations -/
def count_lunch_combinations (menu : CafeteriaMenu) : ℕ :=
  menu.main_dishes.options.card * menu.beverages.options.card * menu.snacks.options.card

/-- Theorem stating that the number of distinct lunch combinations is 27 -/
theorem lunch_combinations_eq_27 (menu : CafeteriaMenu) :
  count_lunch_combinations menu = 27 := by
  sorry

#check lunch_combinations_eq_27

end NUMINAMATH_CALUDE_lunch_combinations_eq_27_l946_94628


namespace NUMINAMATH_CALUDE_fifth_term_is_correct_l946_94662

/-- An arithmetic sequence with the given first four terms -/
def arithmetic_sequence (x y : ℚ) : ℕ → ℚ
| 0 => 2*x + y
| 1 => 2*x - y
| 2 => 2*x*y
| 3 => 2*x / y
| n + 4 => arithmetic_sequence x y 3 + (n + 1) * (arithmetic_sequence x y 1 - arithmetic_sequence x y 0)

/-- The theorem stating that the fifth term of the sequence is -77/10 -/
theorem fifth_term_is_correct (x y : ℚ) :
  arithmetic_sequence x y 0 = 2*x + y →
  arithmetic_sequence x y 1 = 2*x - y →
  arithmetic_sequence x y 2 = 2*x*y →
  arithmetic_sequence x y 3 = 2*x / y →
  arithmetic_sequence x y 4 = -77/10 :=
by sorry

end NUMINAMATH_CALUDE_fifth_term_is_correct_l946_94662


namespace NUMINAMATH_CALUDE_platform_length_calculation_l946_94685

/-- Calculates the length of a platform given train characteristics and crossing time -/
theorem platform_length_calculation (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) :
  train_length = 140 →
  train_speed_kmph = 84 →
  crossing_time = 16 →
  ∃ (platform_length : ℝ), abs (platform_length - 233.33) < 0.01 := by
  sorry

#check platform_length_calculation

end NUMINAMATH_CALUDE_platform_length_calculation_l946_94685


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l946_94617

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (5 * x + 2 * x^2 + 8) = 12 ↔ x = 8 ∨ x = -17/2 := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l946_94617


namespace NUMINAMATH_CALUDE_seven_rings_four_fingers_l946_94690

/-- The number of ways to arrange rings on fingers -/
def ring_arrangements (total_rings : ℕ) (fingers : ℕ) : ℕ :=
  Nat.choose total_rings fingers * 
  Nat.factorial fingers * 
  Nat.choose (total_rings - 1) (fingers - 1)

/-- Theorem stating the number of ring arrangements for 7 rings on 4 fingers -/
theorem seven_rings_four_fingers : 
  ring_arrangements 7 4 = 29400 := by
  sorry

end NUMINAMATH_CALUDE_seven_rings_four_fingers_l946_94690


namespace NUMINAMATH_CALUDE_smallest_integer_solution_eight_satisfies_smallest_integer_is_eight_l946_94613

theorem smallest_integer_solution : ∀ x : ℤ, x < 3 * x - 15 → x ≥ 8 :=
by
  sorry

theorem eight_satisfies : (8 : ℤ) < 3 * 8 - 15 :=
by
  sorry

theorem smallest_integer_is_eight : 
  (∀ x : ℤ, x < 3 * x - 15 → x ≥ 8) ∧ (8 < 3 * 8 - 15) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_eight_satisfies_smallest_integer_is_eight_l946_94613


namespace NUMINAMATH_CALUDE_crazy_silly_school_series_remaining_books_l946_94668

/-- Given a series of books, calculate the number of books remaining to be read -/
def booksRemaining (totalBooks readBooks : ℕ) : ℕ :=
  totalBooks - readBooks

/-- Theorem: In a series of 32 books, if 17 have been read, 15 remain to be read -/
theorem crazy_silly_school_series_remaining_books :
  booksRemaining 32 17 = 15 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_series_remaining_books_l946_94668


namespace NUMINAMATH_CALUDE_zoey_holiday_months_l946_94620

/-- The number of holidays Zoey takes per month -/
def holidays_per_month : ℕ := 2

/-- The total number of holidays Zoey took -/
def total_holidays : ℕ := 24

/-- The number of months Zoey took holidays for -/
def months_of_holidays : ℕ := total_holidays / holidays_per_month

/-- Theorem: The number of months Zoey took holidays for is 12 -/
theorem zoey_holiday_months : months_of_holidays = 12 := by
  sorry

end NUMINAMATH_CALUDE_zoey_holiday_months_l946_94620


namespace NUMINAMATH_CALUDE_building_height_is_270_l946_94693

/-- Calculates the height of a building with specified floor heights -/
def building_height (total_stories : ℕ) (first_half_height : ℕ) (height_increase : ℕ) : ℕ :=
  let first_half := (total_stories / 2) * first_half_height
  let second_half := (total_stories / 2) * (first_half_height + height_increase)
  first_half + second_half

/-- Proves that the height of the specified building is 270 feet -/
theorem building_height_is_270 :
  building_height 20 12 3 = 270 := by
  sorry

#eval building_height 20 12 3

end NUMINAMATH_CALUDE_building_height_is_270_l946_94693


namespace NUMINAMATH_CALUDE_quadratic_vertex_l946_94673

/-- The quadratic function f(x) = 2(x - 4)^2 + 5 -/
def f (x : ℝ) : ℝ := 2 * (x - 4)^2 + 5

/-- The vertex of the quadratic function f -/
def vertex : ℝ × ℝ := (4, 5)

theorem quadratic_vertex :
  (∀ x : ℝ, f x ≥ f (vertex.1)) ∧ f (vertex.1) = vertex.2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l946_94673


namespace NUMINAMATH_CALUDE_phantom_ink_problem_l946_94669

/-- The cost of a single black printer ink -/
def black_ink_cost : ℕ := 11

/-- The amount Phantom's mom gave him -/
def initial_amount : ℕ := 50

/-- The number of black printer inks bought -/
def black_ink_count : ℕ := 2

/-- The number of red printer inks bought -/
def red_ink_count : ℕ := 3

/-- The cost of each red printer ink -/
def red_ink_cost : ℕ := 15

/-- The number of yellow printer inks bought -/
def yellow_ink_count : ℕ := 2

/-- The cost of each yellow printer ink -/
def yellow_ink_cost : ℕ := 13

/-- The additional amount Phantom needs -/
def additional_amount : ℕ := 43

theorem phantom_ink_problem :
  black_ink_cost * black_ink_count +
  red_ink_cost * red_ink_count +
  yellow_ink_cost * yellow_ink_count =
  initial_amount + additional_amount :=
sorry

end NUMINAMATH_CALUDE_phantom_ink_problem_l946_94669
