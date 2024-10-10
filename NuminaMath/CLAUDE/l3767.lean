import Mathlib

namespace fans_with_all_items_l3767_376788

/-- The number of fans in the stadium -/
def total_fans : ℕ := 5000

/-- The interval for t-shirt vouchers -/
def t_shirt_interval : ℕ := 60

/-- The interval for cap vouchers -/
def cap_interval : ℕ := 45

/-- The interval for water bottle vouchers -/
def water_bottle_interval : ℕ := 40

/-- Theorem: The number of fans receiving all three items is equal to the floor of total_fans divided by the LCM of the three intervals -/
theorem fans_with_all_items (total_fans t_shirt_interval cap_interval water_bottle_interval : ℕ) :
  (total_fans / Nat.lcm (Nat.lcm t_shirt_interval cap_interval) water_bottle_interval : ℕ) = 13 :=
sorry

end fans_with_all_items_l3767_376788


namespace blue_lipstick_count_l3767_376735

theorem blue_lipstick_count (total_students : ℕ) 
  (h1 : total_students = 200)
  (h2 : ∃ colored_lipstick : ℕ, colored_lipstick = total_students / 2)
  (h3 : ∃ red_lipstick : ℕ, red_lipstick = (total_students / 2) / 4)
  (h4 : ∃ blue_lipstick : ℕ, blue_lipstick = ((total_students / 2) / 4) / 5) :
  ∃ blue_lipstick : ℕ, blue_lipstick = 5 := by
  sorry

end blue_lipstick_count_l3767_376735


namespace no_natural_squares_diff_2018_l3767_376758

theorem no_natural_squares_diff_2018 : ¬∃ (a b : ℕ), a^2 - b^2 = 2018 := by
  sorry

end no_natural_squares_diff_2018_l3767_376758


namespace x_equation_solution_l3767_376709

theorem x_equation_solution (x : ℝ) (h : x + 1/x = Real.sqrt 5) :
  x^12 - 7*x^8 + x^4 = 343 := by
  sorry

end x_equation_solution_l3767_376709


namespace hemisphere_diameter_l3767_376740

-- Define the cube
def cube_side_length : ℝ := 2

-- Define the hemisphere properties
structure Hemisphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

-- Define the cube with hemispheres
structure CubeWithHemispheres where
  side_length : ℝ
  hemispheres : List Hemisphere
  hemispheres_touch : Bool

-- Theorem statement
theorem hemisphere_diameter (cube : CubeWithHemispheres) 
  (h1 : cube.side_length = cube_side_length)
  (h2 : cube.hemispheres.length = 6)
  (h3 : cube.hemispheres_touch = true) :
  ∀ h ∈ cube.hemispheres, 2 * h.radius = Real.sqrt 2 :=
sorry

end hemisphere_diameter_l3767_376740


namespace sphere_surface_area_from_cube_l3767_376779

theorem sphere_surface_area_from_cube (cube_edge_length : ℝ) (sphere_radius : ℝ) : 
  cube_edge_length = 2 →
  sphere_radius^2 = 3 →
  4 * Real.pi * sphere_radius^2 = 12 * Real.pi :=
by
  sorry

end sphere_surface_area_from_cube_l3767_376779


namespace wine_purchase_problem_l3767_376717

theorem wine_purchase_problem :
  ∃ (x y n m : ℕ), 
    5 * x + 8 * y = n ^ 2 ∧
    n ^ 2 + 60 = m ^ 2 ∧
    x + y = m :=
by sorry

end wine_purchase_problem_l3767_376717


namespace cinema_uses_systematic_sampling_l3767_376773

/-- Represents a sampling method --/
inductive SamplingMethod
| Lottery
| Stratified
| RandomNumberTable
| Systematic

/-- Represents a cinema with rows and seats per row --/
structure Cinema where
  rows : Nat
  seatsPerRow : Nat

/-- Represents a selection rule for seats --/
structure SelectionRule where
  endDigit : Nat

/-- Determines if a sampling method is systematic based on cinema layout and selection rule --/
def isSystematicSampling (c : Cinema) (r : SelectionRule) : Prop :=
  r.endDigit = c.seatsPerRow % 10 ∧ c.seatsPerRow % 10 ≠ 0

/-- Theorem stating that the given cinema scenario uses systematic sampling --/
theorem cinema_uses_systematic_sampling (c : Cinema) (r : SelectionRule) :
  c.rows = 50 → c.seatsPerRow = 30 → r.endDigit = 8 →
  isSystematicSampling c r ∧ SamplingMethod.Systematic = SamplingMethod.Systematic := by
  sorry

end cinema_uses_systematic_sampling_l3767_376773


namespace three_divisors_iff_prime_square_l3767_376756

/-- A positive integer has exactly three distinct divisors if and only if it is the square of a prime number. -/
theorem three_divisors_iff_prime_square (n : ℕ) :
  (∃! (s : Finset ℕ), s.card = 3 ∧ ∀ d ∈ s, d ∣ n) ↔ ∃ p : ℕ, Nat.Prime p ∧ n = p^2 :=
sorry

end three_divisors_iff_prime_square_l3767_376756


namespace dow_jones_problem_l3767_376790

/-- The Dow Jones Industrial Average problem -/
theorem dow_jones_problem (end_value : ℝ) (percent_fall : ℝ) :
  end_value = 8722 →
  percent_fall = 2 →
  (1 - percent_fall / 100) * 8900 = end_value :=
by
  sorry

end dow_jones_problem_l3767_376790


namespace distance_between_vertices_l3767_376771

-- Define the two parabolas
def parabola1 (x : ℝ) : ℝ := x^2 - 6*x + 13
def parabola2 (x : ℝ) : ℝ := x^2 + 2*x + 4

-- Define the vertex of a parabola
def vertex (f : ℝ → ℝ) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem distance_between_vertices : 
  distance (vertex parabola1) (vertex parabola2) = Real.sqrt 17 := by sorry

end distance_between_vertices_l3767_376771


namespace k_range_unique_triangle_l3767_376713

/-- Represents an acute triangle ABC with specific properties -/
structure AcuteTriangle where
  /-- Side length AB -/
  k : ℝ
  /-- Angle C in radians -/
  angleC : ℝ
  /-- Angle A is 60 degrees (π/3 radians) -/
  angleA_eq : angleA = π/3
  /-- Side length BC is 6 -/
  bc_eq : bc = 6
  /-- Triangle is acute -/
  acute : 0 < angleC ∧ angleC < π/2
  /-- Sine rule holds -/
  sine_rule : k = 4 * Real.sqrt 3 * Real.sin angleC

/-- The range of k for the specific acute triangle -/
theorem k_range (t : AcuteTriangle) : 2 * Real.sqrt 3 < t.k ∧ t.k < 4 * Real.sqrt 3 := by
  sorry

/-- There exists only one such triangle -/
theorem unique_triangle : ∃! t : AcuteTriangle, True := by
  sorry

end k_range_unique_triangle_l3767_376713


namespace probability_of_common_books_l3767_376774

def total_books : ℕ := 12
def books_to_choose : ℕ := 6
def common_books : ℕ := 3

theorem probability_of_common_books :
  (Nat.choose total_books common_books * 
   Nat.choose (total_books - common_books) (books_to_choose - common_books) * 
   Nat.choose (total_books - common_books) (books_to_choose - common_books)) / 
  (Nat.choose total_books books_to_choose * Nat.choose total_books books_to_choose) = 220 / 1215 := by
  sorry

end probability_of_common_books_l3767_376774


namespace unsold_books_l3767_376749

theorem unsold_books (total_amount : ℝ) (price_per_book : ℝ) (fraction_sold : ℝ) :
  total_amount = 500 ∧
  price_per_book = 5 ∧
  fraction_sold = 2/3 →
  (1 - fraction_sold) * (total_amount / (price_per_book * fraction_sold)) = 50 := by
  sorry

end unsold_books_l3767_376749


namespace combined_mixture_ratio_l3767_376724

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℚ
  water : ℚ

/-- Combines two mixtures -/
def combine_mixtures (m1 m2 : Mixture) : Mixture :=
  { milk := m1.milk + m2.milk,
    water := m1.water + m2.water }

/-- Calculates the ratio of milk to water in a mixture -/
def milk_water_ratio (m : Mixture) : ℚ × ℚ :=
  (m.milk, m.water)

theorem combined_mixture_ratio :
  let m1 : Mixture := { milk := 4, water := 1 }
  let m2 : Mixture := { milk := 7, water := 3 }
  let combined := combine_mixtures m1 m2
  milk_water_ratio combined = (11, 4) := by
  sorry

end combined_mixture_ratio_l3767_376724


namespace negative_root_implies_inequality_l3767_376747

theorem negative_root_implies_inequality (a : ℝ) : 
  (∃ x : ℝ, x - 3*a + 9 = 0 ∧ x < 0) → (a - 4) * (a - 5) > 0 := by
  sorry

end negative_root_implies_inequality_l3767_376747


namespace max_side_length_triangle_l3767_376765

theorem max_side_length_triangle (a b c : ℕ) : 
  a < b ∧ b < c ∧                -- Three different side lengths
  a + b + c = 24 ∧               -- Perimeter is 24
  a ≥ 4 ∧                        -- Shortest side is at least 4
  c < a + b →                    -- Triangle inequality
  c ≤ 11 :=                      -- Maximum side length is 11
by sorry

end max_side_length_triangle_l3767_376765


namespace quadratic_equal_roots_l3767_376796

theorem quadratic_equal_roots (x : ℝ) : 
  (∃ r : ℝ, (x^2 + 2*x + 1 = 0) ↔ (x = r ∧ x = r)) :=
by sorry

end quadratic_equal_roots_l3767_376796


namespace special_pair_characterization_l3767_376770

/-- A pair of integers is special if it is of the form (n, n-1) or (n-1, n) for some positive integer n. -/
def IsSpecialPair (p : ℤ × ℤ) : Prop :=
  ∃ n : ℤ, n > 0 ∧ (p = (n, n - 1) ∨ p = (n - 1, n))

/-- The sum of two pairs -/
def PairSum (p q : ℤ × ℤ) : ℤ × ℤ :=
  (p.1 + q.1, p.2 + q.2)

/-- A pair can be expressed as a sum of special pairs -/
def CanExpressAsSumOfSpecialPairs (p : ℤ × ℤ) : Prop :=
  ∃ (k : ℕ) (specialPairs : Fin k → ℤ × ℤ),
    k ≥ 2 ∧
    (∀ i, IsSpecialPair (specialPairs i)) ∧
    (∀ i j, i ≠ j → specialPairs i ≠ specialPairs j) ∧
    p = Finset.sum Finset.univ (λ i => specialPairs i)

theorem special_pair_characterization (n m : ℤ) 
    (h_positive : n > 0 ∧ m > 0)
    (h_not_special : ¬IsSpecialPair (n, m)) :
    CanExpressAsSumOfSpecialPairs (n, m) ↔ n + m ≥ (n - m)^2 := by
  sorry

end special_pair_characterization_l3767_376770


namespace exact_pairing_l3767_376744

/-- The number of workers processing large gears to match pairs exactly -/
def workers_large_gears : ℕ := 18

/-- The total number of workers in the workshop -/
def total_workers : ℕ := 34

/-- The number of large gears processed by one worker per day -/
def large_gears_per_worker : ℕ := 20

/-- The number of small gears processed by one worker per day -/
def small_gears_per_worker : ℕ := 15

/-- The number of large gears in a pair -/
def large_gears_per_pair : ℕ := 3

/-- The number of small gears in a pair -/
def small_gears_per_pair : ℕ := 2

theorem exact_pairing :
  workers_large_gears * large_gears_per_worker * small_gears_per_pair =
  (total_workers - workers_large_gears) * small_gears_per_worker * large_gears_per_pair :=
by sorry

end exact_pairing_l3767_376744


namespace work_completion_problem_l3767_376702

theorem work_completion_problem (first_group_days : ℕ) (second_group_men : ℕ) (second_group_days : ℕ) :
  first_group_days = 18 →
  second_group_men = 108 →
  second_group_days = 6 →
  ∃ (first_group_men : ℕ), first_group_men * first_group_days = second_group_men * second_group_days ∧ first_group_men = 36 :=
by
  sorry


end work_completion_problem_l3767_376702


namespace zahra_kimmie_ratio_l3767_376766

def kimmie_earnings : ℚ := 450
def total_savings : ℚ := 375

theorem zahra_kimmie_ratio (zahra_earnings : ℚ) 
  (h1 : zahra_earnings < kimmie_earnings)
  (h2 : total_savings = (1/2) * kimmie_earnings + (1/2) * zahra_earnings) :
  zahra_earnings / kimmie_earnings = 2/3 := by
sorry

end zahra_kimmie_ratio_l3767_376766


namespace largest_three_digit_congruence_l3767_376721

theorem largest_three_digit_congruence :
  ∃ (n : ℕ), n = 993 ∧ 
  (∀ m : ℕ, m ≤ 999 → 45 * m ≡ 270 [MOD 315] → m ≤ n) ∧
  45 * n ≡ 270 [MOD 315] :=
by sorry

end largest_three_digit_congruence_l3767_376721


namespace gigi_banana_consumption_l3767_376715

theorem gigi_banana_consumption (total : ℕ) (days : ℕ) (increase : ℕ) (last_day : ℕ) :
  total = 150 →
  days = 7 →
  increase = 4 →
  (∃ first : ℚ, (days : ℚ) / 2 * (2 * first + (days - 1) * increase) = total) →
  last_day = (days - 1) * increase + (total * 2 / days - (days - 1) * increase) / 2 →
  last_day = 33 :=
by sorry

end gigi_banana_consumption_l3767_376715


namespace regular_hexagon_side_length_l3767_376743

/-- The length of a side in a regular hexagon given the distance between opposite sides -/
theorem regular_hexagon_side_length (d : ℝ) (h : d = 20) : 
  let s := d * 2 / Real.sqrt 3
  s = 40 * Real.sqrt 3 / 3 := by sorry

end regular_hexagon_side_length_l3767_376743


namespace product_equality_l3767_376734

theorem product_equality : 100 * 19.98 * 1.998 * 999 = 3988008 := by
  sorry

end product_equality_l3767_376734


namespace rent_spending_percentage_l3767_376795

theorem rent_spending_percentage (x : ℝ) : 
  x > 0 ∧ x < 100 ∧ 
  x + (x - 0.2 * x) + 28 = 100 → 
  x = 40 := by
sorry

end rent_spending_percentage_l3767_376795


namespace sanchez_problem_l3767_376786

theorem sanchez_problem (x y : ℕ+) : x - y = 3 → x * y = 56 → x + y = 17 := by sorry

end sanchez_problem_l3767_376786


namespace repeating_decimal_sum_l3767_376728

theorem repeating_decimal_sum : 
  (1/3 : ℚ) + (4/999 : ℚ) + (5/9999 : ℚ) = (3378/9999 : ℚ) := by sorry

end repeating_decimal_sum_l3767_376728


namespace stock_price_after_two_years_l3767_376762

/-- The stock price after two years of changes -/
theorem stock_price_after_two_years 
  (initial_price : ℝ) 
  (first_year_increase : ℝ) 
  (second_year_decrease : ℝ) 
  (h1 : initial_price = 120)
  (h2 : first_year_increase = 1.2)
  (h3 : second_year_decrease = 0.3) :
  initial_price * (1 + first_year_increase) * (1 - second_year_decrease) = 184.8 := by
  sorry

end stock_price_after_two_years_l3767_376762


namespace arithmetic_sequence_sum_l3767_376706

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) :
  arithmetic_sequence a →
  a 0 = 3 →
  a 1 = 10 →
  a 2 = 17 →
  a 5 = 32 →
  a 3 + a 4 = 55 := by
sorry

end arithmetic_sequence_sum_l3767_376706


namespace delta_equation_solution_l3767_376722

-- Define the Δ operation
def delta (a b : ℝ) : ℝ := a * b + a + b

-- Theorem statement
theorem delta_equation_solution :
  ∀ p : ℝ, delta p 3 = 39 → p = 9 := by
  sorry

end delta_equation_solution_l3767_376722


namespace transform_equation_l3767_376753

theorem transform_equation (m n x y : ℚ) :
  m + x = n + y → m = n → x = y := by
  sorry

end transform_equation_l3767_376753


namespace point_P_coordinates_l3767_376707

/-- Given a point P with coordinates (3m+6, m-3), prove its coordinates under different conditions --/
theorem point_P_coordinates (m : ℝ) :
  let P : ℝ × ℝ := (3*m + 6, m - 3)
  -- Condition 1: P lies on the angle bisector in the first and third quadrants
  (P.1 = P.2 → P = (-7.5, -7.5)) ∧
  -- Condition 2: The ordinate of P is 5 greater than the abscissa
  (P.2 = P.1 + 5 → P = (-15, -10)) ∧
  -- Condition 3: P lies on the line passing through A(3, -2) and parallel to the y-axis
  (P.1 = 3 → P = (3, -4)) := by sorry

end point_P_coordinates_l3767_376707


namespace eccentricity_for_one_and_nine_l3767_376791

/-- The eccentricity of a curve given two positive numbers -/
def eccentricity_of_curve (x y : ℝ) : Set ℝ :=
  let a := (x + y) / 2
  let b := Real.sqrt (x * y)
  let e₁ := Real.sqrt (a - b) / Real.sqrt a
  let e₂ := Real.sqrt (a + b) / Real.sqrt a
  {e₁, e₂}

/-- Theorem: The eccentricity of the curve for numbers 1 and 9 -/
theorem eccentricity_for_one_and_nine :
  eccentricity_of_curve 1 9 = {Real.sqrt 10 / 5, 2 * Real.sqrt 10 / 5} :=
by sorry

end eccentricity_for_one_and_nine_l3767_376791


namespace overtime_pay_rate_l3767_376720

def regular_pay_rate : ℝ := 10
def regular_hours : ℝ := 40
def total_hours : ℝ := 60
def total_earnings : ℝ := 700

theorem overtime_pay_rate :
  ∃ (overtime_rate : ℝ),
    regular_pay_rate * regular_hours +
    overtime_rate * (total_hours - regular_hours) =
    total_earnings ∧
    overtime_rate = 15 :=
by sorry

end overtime_pay_rate_l3767_376720


namespace pfd_product_theorem_l3767_376789

/-- Partial fraction decomposition coefficients -/
structure PFDCoefficients where
  A : ℚ
  B : ℚ
  C : ℚ

/-- The partial fraction decomposition of (x^2 - 25) / ((x - 1)(x + 3)(x - 4)) -/
def partial_fraction_decomposition : (ℚ → ℚ) → PFDCoefficients → Prop :=
  λ f coeffs =>
    ∀ x, x ≠ 1 ∧ x ≠ -3 ∧ x ≠ 4 →
      f x = coeffs.A / (x - 1) + coeffs.B / (x + 3) + coeffs.C / (x - 4)

/-- The original rational function -/
def original_function (x : ℚ) : ℚ :=
  (x^2 - 25) / ((x - 1) * (x + 3) * (x - 4))

theorem pfd_product_theorem :
  ∃ coeffs : PFDCoefficients,
    partial_fraction_decomposition original_function coeffs ∧
    coeffs.A * coeffs.B * coeffs.C = 24/49 := by
  sorry

end pfd_product_theorem_l3767_376789


namespace tunneled_cube_surface_area_l3767_376703

/-- Represents a cube with its dimensions and composition -/
structure Cube where
  side_length : ℕ
  sub_cube_side : ℕ
  sub_cube_count : ℕ

/-- Represents the tunneling operation on the cube -/
structure TunneledCube extends Cube where
  removed_layers : ℕ
  removed_edge_units : ℕ

/-- Calculates the surface area of a tunneled cube -/
def surface_area (tc : TunneledCube) : ℕ :=
  sorry

/-- The main theorem stating the surface area of the specific tunneled cube -/
theorem tunneled_cube_surface_area :
  let original_cube : Cube := {
    side_length := 12,
    sub_cube_side := 3,
    sub_cube_count := 64
  }
  let tunneled_cube : TunneledCube := {
    side_length := original_cube.side_length,
    sub_cube_side := original_cube.sub_cube_side,
    sub_cube_count := original_cube.sub_cube_count,
    removed_layers := 2,
    removed_edge_units := 1
  }
  surface_area tunneled_cube = 2496 := by
  sorry

end tunneled_cube_surface_area_l3767_376703


namespace problem_solution_l3767_376727

theorem problem_solution (a b c : ℝ) : 
  b = 15 → 
  c = 3 → 
  a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1) → 
  a * b * c = 3 → 
  a = 6 := by
sorry

end problem_solution_l3767_376727


namespace initial_players_correct_l3767_376716

/-- The initial number of players in a video game -/
def initial_players : ℕ := 8

/-- The number of players who quit the game -/
def players_quit : ℕ := 3

/-- The number of lives each remaining player has -/
def lives_per_player : ℕ := 3

/-- The total number of lives after some players quit -/
def total_lives : ℕ := 15

/-- Theorem stating that the initial number of players is correct -/
theorem initial_players_correct : 
  lives_per_player * (initial_players - players_quit) = total_lives :=
by sorry

end initial_players_correct_l3767_376716


namespace square_last_digit_six_implies_second_last_odd_l3767_376737

theorem square_last_digit_six_implies_second_last_odd (n : ℕ) : 
  n^2 % 100 ≥ 6 ∧ n^2 % 100 < 16 → (n^2 / 10) % 2 = 1 := by
  sorry

end square_last_digit_six_implies_second_last_odd_l3767_376737


namespace wire_around_square_field_l3767_376778

theorem wire_around_square_field (area : ℝ) (wire_length : ℝ) : 
  area = 69696 → wire_length = 15840 → 
  (wire_length / (4 * Real.sqrt area)) = 15 := by
  sorry

end wire_around_square_field_l3767_376778


namespace intersection_in_interval_l3767_376754

theorem intersection_in_interval :
  ∃ x₀ : ℝ, 0 < x₀ ∧ x₀ < 1 ∧ x₀^3 = (1/2)^x₀ := by sorry

end intersection_in_interval_l3767_376754


namespace spiral_stripe_length_l3767_376732

/-- The length of a spiral stripe on a cylindrical water tower -/
theorem spiral_stripe_length 
  (circumference height : ℝ) 
  (h_circumference : circumference = 18) 
  (h_height : height = 24) :
  Real.sqrt (circumference^2 + height^2) = 30 := by sorry

end spiral_stripe_length_l3767_376732


namespace uranus_appearance_time_l3767_376769

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  inv : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : ℕ) : Time :=
  let totalMinutes := t.minutes + m
  let newHours := t.hours + totalMinutes / 60
  let newMinutes := totalMinutes % 60
  ⟨newHours % 24, newMinutes, by sorry⟩

/-- Calculates the difference in minutes between two times -/
def minutesBetween (t1 t2 : Time) : ℕ :=
  (t2.hours - t1.hours) * 60 + (t2.minutes - t1.minutes)

theorem uranus_appearance_time
  (marsDisappearance : Time)
  (jupiterDelay : ℕ)
  (uranusDelay : ℕ)
  (h1 : marsDisappearance = ⟨0, 10, by sorry⟩)  -- 12:10 AM
  (h2 : jupiterDelay = 161)  -- 2 hours and 41 minutes
  (h3 : uranusDelay = 196)  -- 3 hours and 16 minutes
  : minutesBetween ⟨6, 0, by sorry⟩ (addMinutes (addMinutes marsDisappearance jupiterDelay) uranusDelay) = 7 :=
by sorry

end uranus_appearance_time_l3767_376769


namespace product_digits_count_l3767_376764

theorem product_digits_count : ∃ n : ℕ, 
  (1002000000000000000 * 999999999999999999 : ℕ) ≥ 10^37 ∧ 
  (1002000000000000000 * 999999999999999999 : ℕ) < 10^38 :=
by sorry

end product_digits_count_l3767_376764


namespace sphere_surface_area_l3767_376741

theorem sphere_surface_area (r : ℝ) (h : r = 3) : 4 * Real.pi * r^2 = 36 * Real.pi := by
  sorry

end sphere_surface_area_l3767_376741


namespace absolute_value_inequality_l3767_376784

theorem absolute_value_inequality (a b c : ℝ) (h : |a + b| < -c) :
  (∃! n : ℕ, n = 2 ∧
    (a < -b - c) ∧
    (a + b > c) ∧
    ¬(a + c < b) ∧
    ¬(|a| + c < b)) :=
by sorry

end absolute_value_inequality_l3767_376784


namespace max_angle_on_perp_bisector_l3767_376757

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the angle between three points
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Define the perpendicular bisector of a line segment
def perpBisector (p1 p2 : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

theorem max_angle_on_perp_bisector 
  (O A : ℝ × ℝ) (r : ℝ) 
  (h_circle : Circle O r)
  (h_interior : A ∈ interior (Circle O r))
  (h_different : A ≠ O) :
  ∃ P : ℝ × ℝ, P ∈ Circle O r ∧ 
    P ∈ perpBisector O A ∧
    ∀ Q : ℝ × ℝ, Q ∈ Circle O r → angle O P A ≥ angle O Q A :=
sorry

end max_angle_on_perp_bisector_l3767_376757


namespace lara_cookies_count_l3767_376755

/-- Calculates the total number of cookies baked by Lara --/
def total_cookies (
  num_trays : ℕ
  ) (
  large_rows_per_tray : ℕ
  ) (
  medium_rows_per_tray : ℕ
  ) (
  small_rows_per_tray : ℕ
  ) (
  large_cookies_per_row : ℕ
  ) (
  medium_cookies_per_row : ℕ
  ) (
  small_cookies_per_row : ℕ
  ) (
  extra_large_cookies : ℕ
  ) : ℕ :=
  (large_rows_per_tray * large_cookies_per_row * num_trays + extra_large_cookies) +
  (medium_rows_per_tray * medium_cookies_per_row * num_trays) +
  (small_rows_per_tray * small_cookies_per_row * num_trays)

theorem lara_cookies_count :
  total_cookies 4 5 4 6 6 7 8 6 = 430 := by
  sorry

end lara_cookies_count_l3767_376755


namespace min_value_of_reciprocal_sum_l3767_376767

theorem min_value_of_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 3) :
  1/a + 4/b ≥ 3 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 3 ∧ 1/a₀ + 4/b₀ = 3 :=
by sorry

end min_value_of_reciprocal_sum_l3767_376767


namespace service_fee_is_24_percent_l3767_376793

/-- Calculates the service fee percentage given the cost of food, tip, and total amount spent. -/
def service_fee_percentage (food_cost tip total_spent : ℚ) : ℚ :=
  ((total_spent - food_cost - tip) / food_cost) * 100

/-- Theorem stating that the service fee percentage is 24% given the problem conditions. -/
theorem service_fee_is_24_percent :
  let food_cost : ℚ := 50
  let tip : ℚ := 5
  let total_spent : ℚ := 61
  service_fee_percentage food_cost tip total_spent = 24 := by
  sorry

end service_fee_is_24_percent_l3767_376793


namespace x_squared_plus_reciprocal_l3767_376714

theorem x_squared_plus_reciprocal (x : ℝ) (h : 47 = x^4 + 1/x^4) : x^2 + 1/x^2 = 7 := by
  sorry

end x_squared_plus_reciprocal_l3767_376714


namespace sales_prediction_at_34_l3767_376739

/-- Represents the linear regression equation for predicting cold drink sales based on temperature -/
def predict_sales (x : ℝ) : ℝ := 2 * x + 60

/-- Theorem stating that when the temperature is 34°C, the predicted sales volume is 128 cups -/
theorem sales_prediction_at_34 :
  predict_sales 34 = 128 := by
  sorry

end sales_prediction_at_34_l3767_376739


namespace relationship_between_exponents_l3767_376745

theorem relationship_between_exponents 
  (a b c d : ℝ) 
  (x y q z : ℝ) 
  (h1 : a^(2*x) = c^(2*q)) 
  (h2 : a^(2*x) = b^2) 
  (h3 : c^(3*y) = a^(3*z)) 
  (h4 : c^(3*y) = d^2) 
  (h5 : a ≠ 0) 
  (h6 : b ≠ 0) 
  (h7 : c ≠ 0) 
  (h8 : d ≠ 0) : 
  x * y = q * z := by
sorry

end relationship_between_exponents_l3767_376745


namespace A_n_is_integer_l3767_376701

theorem A_n_is_integer (a b n : ℕ) (h1 : a > b) (h2 : b > 0) 
  (θ : Real) (h3 : 0 < θ) (h4 : θ < Real.pi / 2) 
  (h5 : Real.sin θ = (2 * a * b : ℝ) / ((a^2 + b^2) : ℝ)) :
  ∃ k : ℤ, ((a^2 + b^2 : ℕ)^n : ℝ) * Real.sin (n * θ) = k := by
  sorry

#check A_n_is_integer

end A_n_is_integer_l3767_376701


namespace complex_modulus_l3767_376719

theorem complex_modulus (z : ℂ) : z / (1 + Complex.I) = -3 * Complex.I → Complex.abs z = 3 * Real.sqrt 2 := by
  sorry

end complex_modulus_l3767_376719


namespace tan_eleven_pi_fourths_l3767_376736

theorem tan_eleven_pi_fourths : Real.tan (11 * π / 4) = -1 := by sorry

end tan_eleven_pi_fourths_l3767_376736


namespace triangle_side_length_l3767_376798

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a = √3, b = 3, and B = 2A, then c = 2√3 -/
theorem triangle_side_length (A B C : Real) (a b c : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧          -- Sum of angles in a triangle
  a = Real.sqrt 3 ∧        -- Given: a = √3
  b = 3 ∧                  -- Given: b = 3
  B = 2 * A →              -- Given: B = 2A
  c = 2 * Real.sqrt 3 :=   -- Conclusion: c = 2√3
by sorry

end triangle_side_length_l3767_376798


namespace max_value_theorem_l3767_376760

theorem max_value_theorem (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 ≠ 0) :
  (|a + 2*b + 3*c| / Real.sqrt (a^2 + b^2 + c^2)) ≤ Real.sqrt 2 ∧
  ∃ (a' b' c' : ℝ), a' + b' + c' = 0 ∧ a'^2 + b'^2 + c'^2 ≠ 0 ∧
    |a' + 2*b' + 3*c'| / Real.sqrt (a'^2 + b'^2 + c'^2) = Real.sqrt 2 :=
by sorry

end max_value_theorem_l3767_376760


namespace max_intersections_theorem_l3767_376723

/-- A convex polygon in a plane -/
structure ConvexPolygon where
  sides : ℕ
  convex : Bool

/-- Two nested convex polygons Q1 and Q2 -/
structure NestedPolygons where
  Q1 : ConvexPolygon
  Q2 : ConvexPolygon
  m : ℕ
  h_m_ge_3 : m ≥ 3
  h_Q1_sides : Q1.sides = m
  h_Q2_sides : Q2.sides = 2 * m
  h_nested : Bool
  h_no_shared_segment : Bool
  h_both_convex : Q1.convex ∧ Q2.convex

/-- The maximum number of intersections between two nested convex polygons -/
def max_intersections (np : NestedPolygons) : ℕ := 2 * np.m^2

/-- Theorem stating the maximum number of intersections -/
theorem max_intersections_theorem (np : NestedPolygons) :
  max_intersections np = 2 * np.m^2 :=
sorry

end max_intersections_theorem_l3767_376723


namespace sin_plus_cos_equals_sqrt_a_plus_one_l3767_376780

theorem sin_plus_cos_equals_sqrt_a_plus_one (θ : Real) (a : Real) 
  (h1 : 0 < θ ∧ θ < π / 2) -- θ is an acute angle
  (h2 : Real.sin (2 * θ) = a) : -- sin 2θ = a
  Real.sin θ + Real.cos θ = Real.sqrt (a + 1) := by sorry

end sin_plus_cos_equals_sqrt_a_plus_one_l3767_376780


namespace max_a_value_l3767_376718

noncomputable def f (x : ℝ) : ℝ := 1 - Real.sqrt (x + 1)

noncomputable def g (a x : ℝ) : ℝ := Real.log (a * x^2 - 3 * x + 1)

theorem max_a_value :
  ∃ (a : ℝ), ∀ (a' : ℝ),
    (∀ (x₁ : ℝ), x₁ ≥ 0 → ∃ (x₂ : ℝ), f x₁ = g a' x₂) →
    a' ≤ a ∧
    (∀ (x₁ : ℝ), x₁ ≥ 0 → ∃ (x₂ : ℝ), f x₁ = g a x₂) ∧
    a = 9/4 :=
sorry

end max_a_value_l3767_376718


namespace original_fraction_l3767_376794

theorem original_fraction (x y : ℚ) : 
  (x * (1 + 12/100)) / (y * (1 - 2/100)) = 6/7 → x/y = 3/4 := by
  sorry

end original_fraction_l3767_376794


namespace percentage_relation_l3767_376785

theorem percentage_relation (a b c : ℝ) 
  (h1 : c = 0.14 * a) 
  (h2 : c = 0.40 * b) : 
  b = 0.35 * a := by
sorry

end percentage_relation_l3767_376785


namespace inequality_statement_not_always_true_l3767_376748

theorem inequality_statement_not_always_true :
  ¬ (∀ a b c : ℝ, a < b → a * c^2 < b * c^2) :=
sorry

end inequality_statement_not_always_true_l3767_376748


namespace exists_sixth_root_of_3_30_sixth_root_of_3_30_correct_l3767_376763

theorem exists_sixth_root_of_3_30 : ∃ n : ℕ, n^6 = 3^30 :=
by
  -- The proof would go here
  sorry

def sixth_root_of_3_30 : ℕ :=
  -- The definition of the actual value would go here
  -- We're not providing the implementation as per the instructions
  sorry

-- This theorem ensures that our defined value actually satisfies the property
theorem sixth_root_of_3_30_correct : (sixth_root_of_3_30)^6 = 3^30 :=
by
  -- The proof would go here
  sorry

end exists_sixth_root_of_3_30_sixth_root_of_3_30_correct_l3767_376763


namespace mixing_hcl_solutions_l3767_376733

/-- Represents a hydrochloric acid solution --/
structure HClSolution where
  mass : ℝ
  concentration : ℝ

/-- Calculates the mass of pure HCl in a solution --/
def HClMass (solution : HClSolution) : ℝ :=
  solution.mass * solution.concentration

theorem mixing_hcl_solutions
  (solution1 : HClSolution)
  (solution2 : HClSolution)
  (mixed : HClSolution)
  (h1 : solution1.concentration = 0.3)
  (h2 : solution2.concentration = 0.1)
  (h3 : mixed.concentration = 0.15)
  (h4 : mixed.mass = 600)
  (h5 : solution1.mass + solution2.mass = mixed.mass)
  (h6 : HClMass solution1 + HClMass solution2 = HClMass mixed) :
  solution1.mass = 150 ∧ solution2.mass = 450 := by
  sorry

end mixing_hcl_solutions_l3767_376733


namespace intersection_of_A_and_B_l3767_376783

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x > 0}
def B : Set ℝ := {x : ℝ | x < 4}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 < x ∧ x < 4} :=
sorry

end intersection_of_A_and_B_l3767_376783


namespace joseph_total_distance_l3767_376792

/-- The total distance Joseph ran over 3 days, given he ran 900 meters each day. -/
def total_distance (distance_per_day : ℕ) (days : ℕ) : ℕ :=
  distance_per_day * days

/-- Theorem stating that Joseph ran 2700 meters in total. -/
theorem joseph_total_distance :
  total_distance 900 3 = 2700 := by
  sorry

end joseph_total_distance_l3767_376792


namespace milk_delivery_theorem_l3767_376746

/-- Calculates the number of jars of milk good for sale given the delivery conditions --/
def goodJarsForSale (
  normalDelivery : ℕ
  ) (jarsPerCarton : ℕ
  ) (cartonShortage : ℕ
  ) (damagedJarsPerCarton : ℕ
  ) (cartonsWithDamagedJars : ℕ
  ) (totallyDamagedCartons : ℕ
  ) : ℕ :=
  let deliveredCartons := normalDelivery - cartonShortage
  let totalJars := deliveredCartons * jarsPerCarton
  let damagedJars := damagedJarsPerCarton * cartonsWithDamagedJars + totallyDamagedCartons * jarsPerCarton
  totalJars - damagedJars

/-- Theorem stating that under the given conditions, there are 565 jars of milk good for sale --/
theorem milk_delivery_theorem :
  goodJarsForSale 50 20 20 3 5 1 = 565 := by
  sorry

end milk_delivery_theorem_l3767_376746


namespace triangle_ratio_theorem_l3767_376797

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_ratio_theorem (abc : Triangle) :
  abc.a = 5 →
  Real.cos abc.B = 4/5 →
  (1/2) * abc.a * abc.c * Real.sin abc.B = 12 →
  (abc.a + abc.c) / (Real.sin abc.A + Real.sin abc.C) = 25/3 := by
  sorry

end triangle_ratio_theorem_l3767_376797


namespace shaded_area_sum_l3767_376700

/-- Represents the shaded area in each level of the square division pattern -/
def shadedAreaSeries : ℕ → ℚ
  | 0 => 1/4
  | n+1 => (1/4) * shadedAreaSeries n

/-- The sum of the infinite geometric series representing the total shaded area -/
def totalShadedArea : ℚ := 1/3

/-- Theorem stating that the sum of the infinite geometric series is 1/3 -/
theorem shaded_area_sum : 
  (∑' n, shadedAreaSeries n) = totalShadedArea := by
  sorry

#check shaded_area_sum

end shaded_area_sum_l3767_376700


namespace basketball_spectators_l3767_376782

/-- Proves the number of children at a basketball match -/
theorem basketball_spectators (total : ℕ) (men : ℕ) (women : ℕ) (children : ℕ) : 
  total = 10000 →
  men = 7000 →
  children = 5 * women →
  total = men + women + children →
  children = 2500 := by
sorry

end basketball_spectators_l3767_376782


namespace intersection_A_B_l3767_376799

-- Define the sets A and B
def A : Set ℝ := {x | |x - 1| > 2}
def B : Set ℝ := {x | x * (x - 5) < 0}

-- State the theorem
theorem intersection_A_B :
  A ∩ B = {x : ℝ | 3 < x ∧ x < 5} :=
by sorry

end intersection_A_B_l3767_376799


namespace max_wrappers_l3767_376705

theorem max_wrappers (andy_wrappers : ℕ) (total_wrappers : ℕ) (max_wrappers : ℕ) : 
  andy_wrappers = 34 → total_wrappers = 49 → max_wrappers = total_wrappers - andy_wrappers →
  max_wrappers = 15 :=
by
  sorry

end max_wrappers_l3767_376705


namespace parabola_vertex_on_x_axis_l3767_376772

/-- A parabola with equation y = x^2 - 8x + m has its vertex on the x-axis if and only if m = 16 -/
theorem parabola_vertex_on_x_axis (m : ℝ) :
  (∃ x : ℝ, x^2 - 8*x + m = 0 ∧ 
   ∀ t : ℝ, t^2 - 8*t + m ≥ 0) ↔ 
  m = 16 :=
sorry

end parabola_vertex_on_x_axis_l3767_376772


namespace product_inequality_l3767_376712

theorem product_inequality (a b m : ℕ) : 
  (a + b = 40 → a * b ≤ 20^2) ∧ 
  (a + b = m → a * b ≤ (m / 2)^2) :=
by sorry

end product_inequality_l3767_376712


namespace max_value_quadratic_l3767_376776

theorem max_value_quadratic (r : ℝ) : 
  -3 * r^2 + 30 * r + 8 ≤ 83 ∧ ∃ r : ℝ, -3 * r^2 + 30 * r + 8 = 83 := by
  sorry

end max_value_quadratic_l3767_376776


namespace min_absolute_sum_l3767_376787

theorem min_absolute_sum (x : ℝ) : 
  |x + 1| + |x + 3| + |x + 6| ≥ 5 ∧ ∃ y : ℝ, |y + 1| + |y + 3| + |y + 6| = 5 :=
by sorry

end min_absolute_sum_l3767_376787


namespace pond_to_field_area_ratio_l3767_376711

/-- Represents a rectangular field with a square pond -/
structure FieldWithPond where
  field_length : ℝ
  field_width : ℝ
  pond_side : ℝ
  length_is_double_width : field_length = 2 * field_width
  length_is_96 : field_length = 96
  pond_side_is_8 : pond_side = 8

/-- The ratio of the pond area to the field area is 1:72 -/
theorem pond_to_field_area_ratio (f : FieldWithPond) :
  (f.pond_side^2) / (f.field_length * f.field_width) = 1 / 72 := by
  sorry

end pond_to_field_area_ratio_l3767_376711


namespace same_solution_implies_c_value_l3767_376750

theorem same_solution_implies_c_value (y : ℝ) (c : ℝ) : 
  (3 * y - 9 = 0) ∧ (c * y + 15 = 3) → c = -4 := by
  sorry

end same_solution_implies_c_value_l3767_376750


namespace fraction_to_decimal_l3767_376751

theorem fraction_to_decimal : (45 : ℚ) / 64 = 0.703125 := by sorry

end fraction_to_decimal_l3767_376751


namespace distance_circle_center_to_line_l3767_376729

theorem distance_circle_center_to_line :
  let line_eq : ℝ → ℝ → Prop := λ x y => x + y = 6
  let circle_eq : ℝ → ℝ → Prop := λ x y => x^2 + (y - 2)^2 = 4
  let circle_center : ℝ × ℝ := (0, 2)
  ∃ d : ℝ, d = 2 * Real.sqrt 2 ∧
    d = (|0 + 2 - 6|) / Real.sqrt ((1:ℝ)^2 + 1^2) :=
by sorry

end distance_circle_center_to_line_l3767_376729


namespace prob_A_or_B_l3767_376761

/- Given probabilities -/
def P_A : ℝ := 0.4
def P_B : ℝ := 0.65
def P_A_and_B : ℝ := 0.25

/- Theorem to prove -/
theorem prob_A_or_B : P_A + P_B - P_A_and_B = 0.8 := by
  sorry

end prob_A_or_B_l3767_376761


namespace polynomial_division_l3767_376738

def dividend (x : ℚ) : ℚ := 10*x^4 + 5*x^3 - 9*x^2 + 7*x + 2
def divisor (x : ℚ) : ℚ := 3*x^2 + 2*x + 1
def quotient (x : ℚ) : ℚ := (10/3)*x^2 - (5/9)*x - 193/243
def remainder (x : ℚ) : ℚ := (592/27)*x + 179/27

theorem polynomial_division :
  ∀ x : ℚ, dividend x = divisor x * quotient x + remainder x :=
by sorry

end polynomial_division_l3767_376738


namespace equation_solutions_l3767_376704

-- Define the equation
def equation (x : ℂ) : Prop :=
  (x - 4)^4 + (x - 6)^4 = 16

-- Define the set of solutions
def solution_set : Set ℂ :=
  {5 + Complex.I * Real.sqrt 7, 5 - Complex.I * Real.sqrt 7, 6, 4}

-- Theorem statement
theorem equation_solutions :
  ∀ x : ℂ, equation x ↔ x ∈ solution_set :=
by sorry

end equation_solutions_l3767_376704


namespace line_through_points_l3767_376726

/-- A line passing through two points (3,1) and (7,13) has equation y = ax + b. This theorem proves that a - b = 11. -/
theorem line_through_points (a b : ℝ) : 
  (1 = a * 3 + b) → (13 = a * 7 + b) → a - b = 11 := by sorry

end line_through_points_l3767_376726


namespace no_geometric_triple_in_arithmetic_sequence_l3767_376759

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

-- Define the property of containing 1 and √2
def contains_one_and_sqrt_two (a : ℕ → ℝ) : Prop :=
  ∃ k l : ℕ, a k = 1 ∧ a l = Real.sqrt 2

-- Define a geometric sequence of three terms
def geometric_sequence (x y z : ℝ) : Prop :=
  y^2 = x * z

-- Main theorem
theorem no_geometric_triple_in_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : contains_one_and_sqrt_two a) : 
  ¬∃ i j k : ℕ, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ geometric_sequence (a i) (a j) (a k) :=
sorry

end no_geometric_triple_in_arithmetic_sequence_l3767_376759


namespace right_triangle_angle_sum_l3767_376742

theorem right_triangle_angle_sum (A B C : Real) : 
  (A + B + C = 180) → (C = 90) → (B = 55) → (A = 35) := by
  sorry

end right_triangle_angle_sum_l3767_376742


namespace sqrt_neg_four_squared_plus_cube_root_neg_eight_equals_two_l3767_376768

theorem sqrt_neg_four_squared_plus_cube_root_neg_eight_equals_two :
  Real.sqrt ((-4)^2) + ((-8 : ℝ) ^ (1/3 : ℝ)) = 2 := by
  sorry

end sqrt_neg_four_squared_plus_cube_root_neg_eight_equals_two_l3767_376768


namespace intersection_of_A_and_B_l3767_376708

def A : Set ℝ := {x | x + 2 > 0}
def B : Set ℝ := {-3, -2, -1, 0}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0} := by
  sorry

end intersection_of_A_and_B_l3767_376708


namespace evaluate_expression_l3767_376752

theorem evaluate_expression (c x y z : ℚ) :
  c = -2 →
  x = 2/5 →
  y = 3/5 →
  z = -3 →
  c * x^3 * y^4 * z^2 = -11664/78125 := by
  sorry

end evaluate_expression_l3767_376752


namespace caps_collection_total_l3767_376777

theorem caps_collection_total (A B C : ℕ) : 
  A = (B + C) / 2 →
  B = (A + C) / 3 →
  C = 150 →
  A + B + C = 360 := by
sorry

end caps_collection_total_l3767_376777


namespace triangle_formation_l3767_376731

/-- Triangle inequality theorem: the sum of any two sides must be greater than the third side --/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if a set of three line segments can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ satisfies_triangle_inequality a b c

theorem triangle_formation :
  can_form_triangle 4 3 6 ∧
  ¬can_form_triangle 1 2 3 ∧
  ¬can_form_triangle 7 8 16 ∧
  ¬can_form_triangle 9 10 20 :=
by sorry

end triangle_formation_l3767_376731


namespace cube_sum_equals_four_l3767_376781

theorem cube_sum_equals_four (x y : ℝ) 
  (h1 : x + y = 1) 
  (h2 : x^2 + y^2 = 3) : 
  x^3 + y^3 = 4 := by
sorry

end cube_sum_equals_four_l3767_376781


namespace prob_genuine_given_equal_weights_l3767_376730

/-- Represents a bag of coins -/
structure CoinBag where
  total : ℕ
  genuine : ℕ
  counterfeit : ℕ

/-- Represents the result of selecting coins -/
inductive Selection
  | AllGenuine
  | Mixed
  | AllCounterfeit

/-- Calculates the probability of selecting all genuine coins -/
def prob_all_genuine (bag : CoinBag) : ℚ :=
  (bag.genuine.choose 2 : ℚ) * ((bag.genuine - 2).choose 2 : ℚ) /
  ((bag.total.choose 2 : ℚ) * ((bag.total - 2).choose 2 : ℚ))

/-- Calculates the probability of equal weights -/
def prob_equal_weights (bag : CoinBag) : ℚ :=
  sorry  -- Actual calculation would go here

/-- The main theorem to prove -/
theorem prob_genuine_given_equal_weights (bag : CoinBag) 
  (h1 : bag.total = 12)
  (h2 : bag.genuine = 9)
  (h3 : bag.counterfeit = 3) :
  prob_all_genuine bag / prob_equal_weights bag = 42 / 165 := by
  sorry

end prob_genuine_given_equal_weights_l3767_376730


namespace trig_identity_l3767_376775

theorem trig_identity : (2 * Real.cos (10 * π / 180) - Real.sin (20 * π / 180)) / Real.sin (70 * π / 180) = Real.sqrt 3 := by
  sorry

end trig_identity_l3767_376775


namespace system_solution_l3767_376725

theorem system_solution (x y : ℝ) : 
  x + 2*y = 8 → 2*x + y = -5 → x + y = 1 := by sorry

end system_solution_l3767_376725


namespace ring_sector_area_proof_l3767_376710

/-- The area of a ring-shaped sector formed by two concentric circles with radii 13 and 7, and a common central angle θ -/
def ring_sector_area (θ : Real) : Real :=
  60 * θ

/-- Theorem: The area of a ring-shaped sector formed by two concentric circles
    with radii 13 and 7, and a common central angle θ, is equal to 60θ -/
theorem ring_sector_area_proof (θ : Real) :
  ring_sector_area θ = 60 * θ := by
  sorry

#check ring_sector_area_proof

end ring_sector_area_proof_l3767_376710
