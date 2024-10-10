import Mathlib

namespace man_speed_on_bridge_l2074_207425

/-- Calculates the speed of a man crossing a bridge -/
theorem man_speed_on_bridge (bridge_length : ℝ) (crossing_time : ℝ) : 
  bridge_length = 2500 →  -- bridge length in meters
  crossing_time = 15 →    -- crossing time in minutes
  bridge_length / (crossing_time / 60) / 1000 = 10 := by
  sorry

end man_speed_on_bridge_l2074_207425


namespace f_properties_l2074_207490

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + |x - 3|

-- Theorem statement
theorem f_properties : 
  (∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min ∧ f x_min = 4) ∧ 
  (∀ (a : ℝ), (∃ (x : ℝ), f x < a) ↔ a > 4) ∧
  (∀ (a b : ℝ), (∀ (x : ℝ), f x < a ↔ b < x ∧ x < 7/2) → a + b = 3.5) :=
by sorry

end f_properties_l2074_207490


namespace smallest_prime_after_seven_nonprimes_l2074_207438

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def consecutive_nonprimes (start : ℕ) (count : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ start → k < start + count → ¬(is_prime k)

theorem smallest_prime_after_seven_nonprimes :
  (∃ start : ℕ, consecutive_nonprimes start 7) ∧
  (∀ p : ℕ, p < 97 → ¬(is_prime p ∧ ∃ start : ℕ, consecutive_nonprimes start 7 ∧ start + 7 ≤ p)) ∧
  is_prime 97 :=
sorry

end smallest_prime_after_seven_nonprimes_l2074_207438


namespace max_d_value_l2074_207452

/-- Represents a 6-digit number of the form 7d7,33e -/
def SixDigitNumber (d e : ℕ) : ℕ := 700000 + d * 10000 + 7000 + 330 + e

/-- Checks if a natural number is a digit (0-9) -/
def isDigit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

/-- The main theorem stating the maximum value of d -/
theorem max_d_value : 
  ∃ (d e : ℕ), isDigit d ∧ isDigit e ∧ 
  (SixDigitNumber d e) % 33 = 0 ∧
  ∀ (d' e' : ℕ), isDigit d' ∧ isDigit e' ∧ (SixDigitNumber d' e') % 33 = 0 → d' ≤ d :=
sorry

end max_d_value_l2074_207452


namespace absolute_value_and_exponents_l2074_207451

theorem absolute_value_and_exponents :
  |(-4 : ℝ)| + (π - Real.sqrt 2)^(0 : ℝ) - (1/2 : ℝ)^(-1 : ℝ) = 3 := by
  sorry

end absolute_value_and_exponents_l2074_207451


namespace snooker_ticket_difference_l2074_207445

theorem snooker_ticket_difference :
  ∀ (vip_tickets gen_tickets : ℕ),
    vip_tickets + gen_tickets = 320 →
    40 * vip_tickets + 15 * gen_tickets = 7500 →
    gen_tickets - vip_tickets = 104 := by
  sorry

end snooker_ticket_difference_l2074_207445


namespace april_flower_sale_earnings_l2074_207485

/-- April's flower sale earnings calculation --/
theorem april_flower_sale_earnings : 
  ∀ (initial_roses final_roses price_per_rose : ℕ),
  initial_roses = 9 →
  final_roses = 4 →
  price_per_rose = 7 →
  (initial_roses - final_roses) * price_per_rose = 35 :=
by
  sorry

end april_flower_sale_earnings_l2074_207485


namespace max_area_inscribed_rectangle_l2074_207429

-- Define a circle with radius R
variable (R : ℝ) (hR : R > 0)

-- Define an inscribed rectangle with side lengths x and y
def inscribed_rectangle (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x^2 + y^2 = (2*R)^2

-- Define the area of a rectangle
def rectangle_area (x y : ℝ) : ℝ := x * y

-- Theorem: The area of any inscribed rectangle is less than or equal to 2R^2
theorem max_area_inscribed_rectangle (x y : ℝ) 
  (h : inscribed_rectangle R x y) : rectangle_area x y ≤ 2 * R^2 := by
  sorry

-- Note: The actual proof is omitted and replaced with 'sorry'

end max_area_inscribed_rectangle_l2074_207429


namespace percentage_difference_l2074_207443

theorem percentage_difference (p q : ℝ) (h : p = 1.25 * q) : 
  (p - q) / p = 0.2 := by
sorry

end percentage_difference_l2074_207443


namespace equation_solution_existence_l2074_207400

theorem equation_solution_existence (n : ℤ) : 
  (∃ x y z : ℤ, x^2 + y^2 + z^2 - x*y - y*z - z*x = n) → 
  (∃ a b : ℤ, a^2 + b^2 - a*b = n) := by
sorry

end equation_solution_existence_l2074_207400


namespace rectangular_garden_area_l2074_207427

theorem rectangular_garden_area :
  ∀ (length width area : ℝ),
    length = 175 →
    width = 12 →
    area = length * width →
    area = 2100 := by
  sorry

end rectangular_garden_area_l2074_207427


namespace movie_theater_popcorn_ratio_l2074_207482

/-- Movie theater revenue calculation and customer ratio --/
theorem movie_theater_popcorn_ratio 
  (matinee_price evening_price opening_price popcorn_price : ℕ)
  (matinee_customers evening_customers opening_customers : ℕ)
  (total_revenue : ℕ)
  (h_matinee : matinee_price = 5)
  (h_evening : evening_price = 7)
  (h_opening : opening_price = 10)
  (h_popcorn : popcorn_price = 10)
  (h_matinee_cust : matinee_customers = 32)
  (h_evening_cust : evening_customers = 40)
  (h_opening_cust : opening_customers = 58)
  (h_total_rev : total_revenue = 1670) :
  (total_revenue - (matinee_price * matinee_customers + 
                    evening_price * evening_customers + 
                    opening_price * opening_customers)) / popcorn_price = 
  (matinee_customers + evening_customers + opening_customers) / 2 :=
by sorry

end movie_theater_popcorn_ratio_l2074_207482


namespace distance_to_point_l2074_207423

-- Define the point
def point : ℝ × ℝ := (-12, 5)

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem distance_to_point : Real.sqrt ((point.1 - origin.1)^2 + (point.2 - origin.2)^2) = 13 := by
  sorry

end distance_to_point_l2074_207423


namespace author_writing_speed_l2074_207460

/-- Calculates the words written per hour, given the total words, total hours, and break hours -/
def wordsPerHour (totalWords : ℕ) (totalHours : ℕ) (breakHours : ℕ) : ℕ :=
  totalWords / (totalHours - breakHours)

/-- Theorem stating that under the given conditions, the author wrote at least 705 words per hour -/
theorem author_writing_speed :
  let totalWords : ℕ := 60000
  let totalHours : ℕ := 100
  let breakHours : ℕ := 15
  wordsPerHour totalWords totalHours breakHours ≥ 705 := by
  sorry

#eval wordsPerHour 60000 100 15

end author_writing_speed_l2074_207460


namespace recipe_flour_amount_l2074_207493

/-- The number of cups of flour Mary has already put in -/
def flour_already_added : ℕ := 2

/-- The number of cups of flour Mary needs to add -/
def flour_to_add : ℕ := 5

/-- The total number of cups of flour in the recipe -/
def total_flour : ℕ := flour_already_added + flour_to_add

theorem recipe_flour_amount : total_flour = 7 := by
  sorry

end recipe_flour_amount_l2074_207493


namespace mushroom_soup_production_l2074_207424

theorem mushroom_soup_production (total_required : ℕ) (team1_production : ℕ) (team2_production : ℕ) 
  (h1 : total_required = 280)
  (h2 : team1_production = 90)
  (h3 : team2_production = 120) :
  total_required - (team1_production + team2_production) = 70 :=
by sorry

end mushroom_soup_production_l2074_207424


namespace q_div_p_eq_90_l2074_207437

/-- The number of cards in the box -/
def total_cards : ℕ := 50

/-- The number of distinct numbers on the cards -/
def distinct_numbers : ℕ := 10

/-- The number of cards for each number -/
def cards_per_number : ℕ := 5

/-- The number of cards drawn -/
def cards_drawn : ℕ := 4

/-- The probability of drawing all cards with the same number -/
def p : ℚ := (distinct_numbers * Nat.choose cards_per_number cards_drawn) / Nat.choose total_cards cards_drawn

/-- The probability of drawing three cards with one number and one card with a different number -/
def q : ℚ := (distinct_numbers * Nat.choose cards_per_number 3 * (distinct_numbers - 1) * Nat.choose cards_per_number 1) / Nat.choose total_cards cards_drawn

/-- The main theorem stating that q/p = 90 -/
theorem q_div_p_eq_90 : q / p = 90 := by
  sorry

end q_div_p_eq_90_l2074_207437


namespace floor_divisibility_implies_integer_l2074_207487

/-- Floor function -/
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

/-- Property: For all integers m and n, if m divides n, then ⌊mr⌋ divides ⌊nr⌋ -/
def floor_divisibility_property (r : ℝ) : Prop :=
  ∀ (m n : ℤ), m ∣ n → (floor (m * r) : ℤ) ∣ (floor (n * r) : ℤ)

/-- Theorem: If r ≥ 0 satisfies the floor divisibility property, then r is an integer -/
theorem floor_divisibility_implies_integer (r : ℝ) (h1 : r ≥ 0) (h2 : floor_divisibility_property r) : ∃ (n : ℤ), r = n := by
  sorry

end floor_divisibility_implies_integer_l2074_207487


namespace jessica_seashells_l2074_207444

theorem jessica_seashells (initial_shells : ℝ) (given_away : ℝ) (remaining : ℝ) : 
  initial_shells = 8.5 → 
  given_away = 6.25 → 
  remaining = initial_shells - given_away → 
  remaining = 2.25 := by
sorry

end jessica_seashells_l2074_207444


namespace least_number_with_conditions_l2074_207430

theorem least_number_with_conditions : ∃ n : ℕ, 
  (n = 1262) ∧ 
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 7 → n % k = 2) ∧
  (n % 13 = 0) ∧
  (∀ m : ℕ, m < n → ¬(∀ k : ℕ, 2 ≤ k ∧ k ≤ 7 → m % k = 2) ∨ m % 13 ≠ 0) :=
by sorry

end least_number_with_conditions_l2074_207430


namespace video_game_enemies_l2074_207480

/-- The number of points earned per defeated enemy -/
def points_per_enemy : ℕ := 3

/-- The number of enemies left undefeated -/
def undefeated_enemies : ℕ := 2

/-- The total points earned -/
def total_points : ℕ := 12

/-- The total number of enemies in the level -/
def total_enemies : ℕ := 6

theorem video_game_enemies :
  total_enemies = (total_points / points_per_enemy) + undefeated_enemies :=
sorry

end video_game_enemies_l2074_207480


namespace max_value_on_ellipse_l2074_207416

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop :=
  x^2 + 3*x*y + 2*y^2 - 14*x - 21*y + 49 = 0

-- Define the function to be maximized
def f (x y : ℝ) : ℝ := x + y

-- Theorem statement
theorem max_value_on_ellipse :
  ∃ (x₀ y₀ : ℝ), ellipse x₀ y₀ ∧
  (∀ (x y : ℝ), ellipse x y → f x y ≤ f x₀ y₀) ∧
  f x₀ y₀ = 343 / 88 := by
  sorry

end max_value_on_ellipse_l2074_207416


namespace line_slope_intercept_sum_l2074_207450

/-- A line with slope 3 passing through (-2, 4) has m + b = 13 when written as y = mx + b -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  m = 3 → 
  4 = m * (-2) + b → 
  m + b = 13 := by sorry

end line_slope_intercept_sum_l2074_207450


namespace robin_total_pieces_l2074_207446

/-- The number of gum packages Robin has -/
def gum_packages : ℕ := 21

/-- The number of candy packages Robin has -/
def candy_packages : ℕ := 45

/-- The number of mint packages Robin has -/
def mint_packages : ℕ := 30

/-- The number of gum pieces in each gum package -/
def gum_pieces_per_package : ℕ := 9

/-- The number of candy pieces in each candy package -/
def candy_pieces_per_package : ℕ := 12

/-- The number of mint pieces in each mint package -/
def mint_pieces_per_package : ℕ := 8

/-- The total number of pieces Robin has -/
def total_pieces : ℕ := gum_packages * gum_pieces_per_package + 
                        candy_packages * candy_pieces_per_package + 
                        mint_packages * mint_pieces_per_package

theorem robin_total_pieces : total_pieces = 969 := by
  sorry

end robin_total_pieces_l2074_207446


namespace ellipse_minor_axis_length_l2074_207410

/-- The length of the minor axis of an ellipse with semi-focal distance 2 and eccentricity 1/2 is 2√3. -/
theorem ellipse_minor_axis_length : 
  ∀ (c a b : ℝ), 
  c = 2 → -- semi-focal distance
  a / c = 2 → -- derived from eccentricity e = 1/2
  b ^ 2 = a ^ 2 - c ^ 2 → -- relationship between a, b, and c in an ellipse
  b = 2 * Real.sqrt 3 := by
sorry

end ellipse_minor_axis_length_l2074_207410


namespace inequality_solution_l2074_207477

theorem inequality_solution (x : ℝ) : 
  (2 * x + 2) / (3 * x + 1) < (x - 3) / (x + 4) ↔ 
  (x > -Real.sqrt 11 ∧ x < -1/3) ∨ (x > Real.sqrt 11) :=
by sorry

end inequality_solution_l2074_207477


namespace polynomial_expansion_coefficient_l2074_207428

theorem polynomial_expansion_coefficient (x : ℝ) : 
  ∃ (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ), 
    (x^10 = a + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + 
            a₅*(x-1)^5 + a₆*(x-1)^6 + a₇*(x-1)^7 + a₈*(x-1)^8 + 
            a₉*(x-1)^9 + a₁₀*(x-1)^10) ∧ 
    a₇ = 120 := by
  sorry

end polynomial_expansion_coefficient_l2074_207428


namespace triangle_max_perimeter_l2074_207419

theorem triangle_max_perimeter (a b c : ℝ) (A B C : ℝ) :
  c = Real.sqrt 3 →
  (Real.sqrt 3 + a) * (Real.sin C - Real.sin A) = (a + b) * Real.sin B →
  a > 0 →
  b > 0 →
  (a + b + c : ℝ) ≤ 2 + Real.sqrt 3 :=
by sorry

end triangle_max_perimeter_l2074_207419


namespace f_property_l2074_207478

theorem f_property (f : ℝ → ℝ) 
  (h : ∀ x y z : ℝ, f (x^2 + y * f z + z) = x * f x + z * f y + f z) :
  (∃ a b : ℝ, (∀ x : ℝ, f x = a ∨ f x = b) ∧ f 5 = a ∨ f 5 = b) ∧
  (∃ s : ℝ, s = (f 5 + f 5) ∧ s = 5) :=
sorry

end f_property_l2074_207478


namespace car_speed_calculation_l2074_207489

theorem car_speed_calculation (v : ℝ) : v > 0 → (1 / v) * 3600 = (1 / 80) * 3600 + 10 → v = 3600 / 55 := by
  sorry

end car_speed_calculation_l2074_207489


namespace test_score_ranges_l2074_207453

theorem test_score_ranges (range1 range2 range3 : ℕ) : 
  range1 ≤ range2 ∧ range2 ≤ range3 →  -- Assuming ranges are ordered
  range1 ≥ 30 →                        -- Minimum range is 30
  range3 = 32 →                        -- One range is 32
  range2 = 18 :=                       -- Prove second range is 18
by sorry

end test_score_ranges_l2074_207453


namespace prism_128_cubes_ratio_l2074_207441

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Checks if the given dimensions form a valid prism of 128 cubes -/
def is_valid_prism (d : PrismDimensions) : Prop :=
  d.width * d.length * d.height = 128

/-- Checks if the given dimensions have the ratio 1:1:2 -/
def has_ratio_1_1_2 (d : PrismDimensions) : Prop :=
  d.width = d.length ∧ d.height = 2 * d.width

/-- Theorem stating that a valid prism of 128 cubes has dimensions with ratio 1:1:2 -/
theorem prism_128_cubes_ratio :
  ∀ d : PrismDimensions, is_valid_prism d → has_ratio_1_1_2 d :=
by sorry

end prism_128_cubes_ratio_l2074_207441


namespace reimbursement_is_correct_l2074_207413

/-- Represents the type of client --/
inductive ClientType
| Industrial
| Commercial

/-- Represents the day of the week --/
inductive DayType
| Weekday
| Weekend

/-- Calculates the reimbursement rate based on client type and day type --/
def reimbursementRate (client : ClientType) (day : DayType) : ℚ :=
  match client, day with
  | ClientType.Industrial, DayType.Weekday => 36/100
  | ClientType.Commercial, DayType.Weekday => 42/100
  | _, DayType.Weekend => 45/100

/-- Represents a day's travel --/
structure DayTravel where
  miles : ℕ
  client : ClientType
  day : DayType

/-- Calculates the reimbursement for a single day --/
def dailyReimbursement (travel : DayTravel) : ℚ :=
  (travel.miles : ℚ) * reimbursementRate travel.client travel.day

/-- The week's travel schedule --/
def weekSchedule : List DayTravel := [
  ⟨18, ClientType.Industrial, DayType.Weekday⟩,
  ⟨26, ClientType.Commercial, DayType.Weekday⟩,
  ⟨20, ClientType.Industrial, DayType.Weekday⟩,
  ⟨20, ClientType.Commercial, DayType.Weekday⟩,
  ⟨16, ClientType.Industrial, DayType.Weekday⟩,
  ⟨12, ClientType.Commercial, DayType.Weekend⟩
]

/-- Calculates the total reimbursement for the week --/
def totalReimbursement (schedule : List DayTravel) : ℚ :=
  schedule.map dailyReimbursement |>.sum

/-- Theorem: The total reimbursement for the given week schedule is $44.16 --/
theorem reimbursement_is_correct : totalReimbursement weekSchedule = 4416/100 := by
  sorry

end reimbursement_is_correct_l2074_207413


namespace solve_equation_l2074_207402

theorem solve_equation (x : ℝ) (h : 3 * x = (26 - x) + 26) : x = 13 := by
  sorry

end solve_equation_l2074_207402


namespace parabola_directrix_tangent_circle_l2074_207432

/-- Given a parabola y^2 = 2px (p > 0) with directrix x = -p/2, 
    if the directrix is tangent to the circle (x - 3)^2 + y^2 = 16, then p = 2 -/
theorem parabola_directrix_tangent_circle (p : ℝ) (h1 : p > 0) : 
  (∀ x y : ℝ, y^2 = 2*p*x → x = -p/2 → (x - 3)^2 + y^2 = 16) → p = 2 := by
  sorry

end parabola_directrix_tangent_circle_l2074_207432


namespace ninth_term_of_specific_arithmetic_sequence_l2074_207474

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) : ℕ → ℚ :=
  λ n => a₁ + (n - 1 : ℚ) * d

theorem ninth_term_of_specific_arithmetic_sequence :
  ∃ (d : ℚ), 
    let seq := arithmetic_sequence (3/4) d
    seq 1 = 3/4 ∧ seq 17 = 1/2 ∧ seq 9 = 5/8 :=
by
  sorry

end ninth_term_of_specific_arithmetic_sequence_l2074_207474


namespace line_point_k_value_l2074_207495

/-- A line contains the points (3, 5), (-3, k), and (-9, -2). The value of k is 3/2. -/
theorem line_point_k_value (k : ℚ) : 
  (∃ (m b : ℚ), 5 = m * 3 + b ∧ k = m * (-3) + b ∧ -2 = m * (-9) + b) → k = 3/2 := by
  sorry

end line_point_k_value_l2074_207495


namespace rectangle_area_difference_l2074_207408

/-- Given a large rectangle of dimensions A × B and a small rectangle of dimensions a × b,
    where the small rectangle is entirely contained within the large rectangle,
    this theorem proves that the absolute difference between the total area of the parts
    of the small rectangle outside the large rectangle and the area of the large rectangle
    not covered by the small rectangle is equal to 572, given specific dimensions. -/
theorem rectangle_area_difference (A B a b : ℝ) 
    (h1 : A = 20) (h2 : B = 30) (h3 : a = 4) (h4 : b = 7)
    (h5 : a ≤ A ∧ b ≤ B) : 
    |0 - (A * B - a * b)| = 572 := by
  sorry

end rectangle_area_difference_l2074_207408


namespace safe_game_probabilities_l2074_207464

/-- The probability of opening all safes given the number of safes and initially opened safes. -/
def P (m n : ℕ) : ℚ :=
  sorry

theorem safe_game_probabilities (n : ℕ) (h : n ≥ 2) :
  P 2 3 = 2/3 ∧
  (∀ k, P 1 k = 1/k) ∧
  (∀ k ≥ 2, P 2 k = (2/k) * P 1 (k-1) + ((k-2)/k) * P 2 (k-1)) ∧
  (∀ k ≥ 2, P 2 k = 2/k) :=
sorry

end safe_game_probabilities_l2074_207464


namespace area_of_three_presentable_set_l2074_207458

/-- A complex number is three-presentable if there exists a complex number w
    with |w| = 3 such that z = w - 1/w -/
def ThreePresentable (z : ℂ) : Prop :=
  ∃ w : ℂ, Complex.abs w = 3 ∧ z = w - 1 / w

/-- T is the set of all three-presentable complex numbers -/
def T : Set ℂ :=
  {z : ℂ | ThreePresentable z}

/-- The area of a set in the complex plane -/
noncomputable def AreaInside (S : Set ℂ) : ℝ := sorry

theorem area_of_three_presentable_set :
  AreaInside T = (80 / 9) * Real.pi := by sorry

end area_of_three_presentable_set_l2074_207458


namespace xy_positive_l2074_207462

theorem xy_positive (x y : ℝ) (h1 : x * y > 1) (h2 : x + y ≥ 0) : x > 0 ∧ y > 0 := by
  sorry

end xy_positive_l2074_207462


namespace equilateral_triangle_cd_product_l2074_207449

/-- An equilateral triangle with vertices at (0,0), (c, 20), and (d, 51) -/
structure EquilateralTriangle where
  c : ℝ
  d : ℝ
  is_equilateral : (c^2 + 20^2 = d^2 + 51^2) ∧ 
                   (c^2 + 20^2 = c^2 + d^2 + 51^2 - 2*c*d - 2*20*51) ∧
                   (d^2 + 51^2 = c^2 + d^2 + 51^2 - 2*c*d - 2*20*51)

/-- The product of c and d in the equilateral triangle equals -5822/3 -/
theorem equilateral_triangle_cd_product (t : EquilateralTriangle) : t.c * t.d = -5822/3 := by
  sorry

end equilateral_triangle_cd_product_l2074_207449


namespace perpendicular_lines_b_value_l2074_207440

-- Define the slopes of the two lines
def slope1 : ℚ := -1/5
def slope2 (b : ℚ) : ℚ := -b/4

-- Define the perpendicularity condition
def perpendicular (b : ℚ) : Prop := slope1 * slope2 b = -1

-- Theorem statement
theorem perpendicular_lines_b_value : 
  ∀ b : ℚ, perpendicular b → b = -20 := by
  sorry

end perpendicular_lines_b_value_l2074_207440


namespace farm_has_eleven_goats_l2074_207404

/-- Represents the number of animals on a farm -/
structure Farm where
  goats : ℕ
  cows : ℕ
  pigs : ℕ

/-- Defines the properties of the farm in the problem -/
def ProblemFarm (f : Farm) : Prop :=
  (f.pigs = 2 * f.cows) ∧ 
  (f.cows = f.goats + 4) ∧ 
  (f.goats + f.cows + f.pigs = 56)

/-- Theorem stating that a farm satisfying the problem conditions has 11 goats -/
theorem farm_has_eleven_goats (f : Farm) (h : ProblemFarm f) : f.goats = 11 := by
  sorry


end farm_has_eleven_goats_l2074_207404


namespace egor_can_always_achieve_two_roots_egor_cannot_always_achieve_more_than_two_roots_l2074_207455

/-- Represents a player in the polynomial coefficient game -/
inductive Player
| Igor
| Egor

/-- Represents the state of the game after each move -/
structure GameState where
  coefficients : Vector ℤ 100
  currentPlayer : Player

/-- A strategy for a player in the game -/
def Strategy := GameState → ℕ → ℤ

/-- The game play function -/
def play (igorStrategy : Strategy) (egorStrategy : Strategy) : Vector ℤ 100 := sorry

/-- Counts the number of distinct integer roots of a polynomial -/
def countDistinctIntegerRoots (coeffs : Vector ℤ 100) : ℕ := sorry

/-- The main theorem stating Egor can always achieve 2 distinct integer roots -/
theorem egor_can_always_achieve_two_roots :
  ∃ (egorStrategy : Strategy),
    ∀ (igorStrategy : Strategy),
      countDistinctIntegerRoots (play igorStrategy egorStrategy) ≥ 2 := sorry

/-- The main theorem stating Egor cannot always achieve more than 2 distinct integer roots -/
theorem egor_cannot_always_achieve_more_than_two_roots :
  ∀ (egorStrategy : Strategy),
    ∃ (igorStrategy : Strategy),
      countDistinctIntegerRoots (play igorStrategy egorStrategy) ≤ 2 := sorry

end egor_can_always_achieve_two_roots_egor_cannot_always_achieve_more_than_two_roots_l2074_207455


namespace modified_factor_tree_l2074_207467

theorem modified_factor_tree (P X Y G Z : ℕ) : 
  P = X * Y ∧
  X = 7 * G ∧
  Y = 11 * Z ∧
  G = 7 * 4 ∧
  Z = 11 * 4 →
  P = 94864 := by
sorry

end modified_factor_tree_l2074_207467


namespace largest_package_size_l2074_207417

theorem largest_package_size (ming_pencils catherine_pencils lucas_pencils : ℕ) 
  (h_ming : ming_pencils = 48)
  (h_catherine : catherine_pencils = 36)
  (h_lucas : lucas_pencils = 60) :
  Nat.gcd ming_pencils (Nat.gcd catherine_pencils lucas_pencils) = 12 := by
  sorry

end largest_package_size_l2074_207417


namespace shortest_side_right_triangle_l2074_207456

theorem shortest_side_right_triangle (a b c : ℝ) (ha : a = 9) (hb : b = 12) 
  (hright : a^2 + c^2 = b^2) : 
  c = Real.sqrt (b^2 - a^2) :=
sorry

end shortest_side_right_triangle_l2074_207456


namespace cylinder_in_hemisphere_height_l2074_207472

theorem cylinder_in_hemisphere_height (r c h : ℝ) : 
  r > 0 → c > 0 → h > 0 →
  r = 7 → c = 3 →
  h^2 = r^2 - c^2 →
  h = Real.sqrt 40 := by
sorry

end cylinder_in_hemisphere_height_l2074_207472


namespace tetrahedron_surface_area_l2074_207465

/-- Given a regular tetrahedron with a square cross-section of area m², 
    its surface area is 4m²√3. -/
theorem tetrahedron_surface_area (m : ℝ) (h : m > 0) : 
  let square_area : ℝ := m^2
  let tetrahedron_surface_area : ℝ := 4 * m^2 * Real.sqrt 3
  square_area = m^2 → tetrahedron_surface_area = 4 * m^2 * Real.sqrt 3 :=
by sorry

end tetrahedron_surface_area_l2074_207465


namespace special_arithmetic_sequence_general_term_l2074_207421

/-- An arithmetic sequence with a1 = 4 and a1, a5, a13 forming a geometric sequence -/
structure SpecialArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d
  a1_eq_4 : a 1 = 4
  geometric_subsequence : ∃ r : ℝ, a 5 = a 1 * r ∧ a 13 = a 5 * r

/-- The general term formula for the special arithmetic sequence -/
def general_term (seq : SpecialArithmeticSequence) (n : ℕ) : ℝ :=
  n + 3

theorem special_arithmetic_sequence_general_term (seq : SpecialArithmeticSequence) :
  ∀ n : ℕ, seq.a n = general_term seq n ∨ seq.a n = 4 := by
  sorry

end special_arithmetic_sequence_general_term_l2074_207421


namespace fixed_point_of_function_l2074_207459

theorem fixed_point_of_function (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 4 + a^(x - 1)
  f 1 = 5 := by sorry

end fixed_point_of_function_l2074_207459


namespace lemon_heads_in_package_l2074_207481

/-- The number of Lemon Heads Louis ate -/
def total_lemon_heads : ℕ := 54

/-- The number of whole boxes Louis ate -/
def whole_boxes : ℕ := 9

/-- The number of Lemon Heads in one package -/
def lemon_heads_per_package : ℕ := total_lemon_heads / whole_boxes

theorem lemon_heads_in_package : lemon_heads_per_package = 6 := by
  sorry

end lemon_heads_in_package_l2074_207481


namespace triangle_construction_feasibility_l2074_207414

/-- Given a triangle with sides a and c, and angle condition α = 2β, 
    the triangle construction is feasible if and only if a > (2/3)c -/
theorem triangle_construction_feasibility (a c : ℝ) (α β : ℝ) 
  (h_positive_a : a > 0) (h_positive_c : c > 0) (h_angle : α = 2 * β) :
  (∃ b : ℝ, b > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) ↔ a > (2/3) * c := by
sorry

end triangle_construction_feasibility_l2074_207414


namespace forces_form_hyperboloid_rulings_l2074_207499

-- Define a 3D vector
def Vector3D := ℝ × ℝ × ℝ

-- Define a line in 3D space
structure Line3D where
  point : Vector3D
  direction : Vector3D

-- Define a force as a line with a magnitude
structure Force where
  line : Line3D
  magnitude : ℝ

-- Define the concept of equilibrium
def is_equilibrium (forces : List Force) : Prop := sorry

-- Define the concept of non-coplanarity
def are_non_coplanar (lines : List Line3D) : Prop := sorry

-- Define the concept of a hyperboloid
def is_hyperboloid_ruling (lines : List Line3D) : Prop := sorry

-- The main theorem
theorem forces_form_hyperboloid_rulings 
  (forces : List Force) 
  (h_count : forces.length = 4)
  (h_equilibrium : is_equilibrium forces)
  (h_non_coplanar : are_non_coplanar (forces.map Force.line)) :
  is_hyperboloid_ruling (forces.map Force.line) := by sorry

end forces_form_hyperboloid_rulings_l2074_207499


namespace unique_number_with_pairable_divisors_l2074_207479

def is_own_divisor (d n : ℕ) : Prop :=
  d ∣ n ∧ d ≠ 1 ∧ d ≠ n

def has_pairable_own_divisors (n : ℕ) : Prop :=
  ∃ (f : ℕ → ℕ), 
    (∀ d, is_own_divisor d n → is_own_divisor (f d) n) ∧
    (∀ d, is_own_divisor d n → f (f d) = d) ∧
    (∀ d, is_own_divisor d n → (f d = d + 545 ∨ f d = d - 545))

theorem unique_number_with_pairable_divisors :
  ∃! n : ℕ, has_pairable_own_divisors n ∧ n = 1094 :=
sorry

end unique_number_with_pairable_divisors_l2074_207479


namespace parabola_point_x_coordinate_l2074_207448

/-- The x-coordinate of a point on the parabola y^2 = 4x that is 4 units away from the focus -/
theorem parabola_point_x_coordinate (x y : ℝ) : 
  y^2 = 4*x →                            -- Point (x, y) is on the parabola y^2 = 4x
  (x - 1)^2 + y^2 = 4^2 →                -- Distance from (x, y) to focus (1, 0) is 4
  x = 3 := by
sorry

end parabola_point_x_coordinate_l2074_207448


namespace circle_and_line_intersection_l2074_207483

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 4)^2 = 25

-- Define the moving line l
def line_l (m x y : ℝ) : Prop :=
  (m + 2) * x + (2 * m + 1) * y - 7 * m - 8 = 0

-- Theorem statement
theorem circle_and_line_intersection :
  -- Given conditions
  (circle_C (-2) 1) ∧  -- Circle C passes through A(-2, 1)
  (circle_C 5 0) ∧     -- Circle C passes through B(5, 0)
  (∃ x y : ℝ, circle_C x y ∧ y = 2 * x) →  -- Center of C is on y = 2x
  -- Conclusions
  ((∀ x y : ℝ, circle_C x y ↔ (x - 2)^2 + (y - 4)^2 = 25) ∧
   (∃ min_PQ : ℝ, 
     (min_PQ = 4 * Real.sqrt 5) ∧
     (∀ m x1 y1 x2 y2 : ℝ,
       (circle_C x1 y1 ∧ circle_C x2 y2 ∧ 
        line_l m x1 y1 ∧ line_l m x2 y2) →
       ((x1 - x2)^2 + (y1 - y2)^2 ≥ min_PQ^2))))
  := by sorry

end circle_and_line_intersection_l2074_207483


namespace num_outfits_is_480_l2074_207498

/-- Number of shirts available --/
def num_shirts : ℕ := 8

/-- Number of ties available --/
def num_ties : ℕ := 5

/-- Number of pants available --/
def num_pants : ℕ := 3

/-- Number of belts available --/
def num_belts : ℕ := 4

/-- Number of belts that can be worn with a tie --/
def num_belts_with_tie : ℕ := 2

/-- Calculate the number of different outfits --/
def num_outfits : ℕ :=
  let outfits_without_tie := num_shirts * num_pants * (num_belts + 1)
  let outfits_with_tie := num_shirts * num_pants * num_ties * (num_belts_with_tie + 1)
  outfits_without_tie + outfits_with_tie

/-- Theorem stating that the number of different outfits is 480 --/
theorem num_outfits_is_480 : num_outfits = 480 := by
  sorry

end num_outfits_is_480_l2074_207498


namespace complex_number_problem_l2074_207484

theorem complex_number_problem (a b : ℝ) (i : ℂ) : 
  (a - 2*i) * i = b - i → a^2 + b^2 = 5 := by sorry

end complex_number_problem_l2074_207484


namespace quadratic_equation_theorem_l2074_207447

/-- Quadratic equation parameters -/
structure QuadraticParams where
  m : ℝ

/-- Roots of the quadratic equation -/
structure QuadraticRoots where
  x1 : ℝ
  x2 : ℝ

/-- Theorem about the quadratic equation x^2 - (2m+3)x + m^2 + 2 = 0 -/
theorem quadratic_equation_theorem (p : QuadraticParams) (r : QuadraticRoots) : 
  /- The equation has real roots if and only if m ≥ -1/12 -/
  (∃ (x : ℝ), x^2 - (2*p.m + 3)*x + p.m^2 + 2 = 0) ↔ p.m ≥ -1/12 ∧
  
  /- If x1 and x2 are the roots of the equation and satisfy the given condition, then m = 13 -/
  (r.x1^2 - (2*p.m + 3)*r.x1 + p.m^2 + 2 = 0 ∧
   r.x2^2 - (2*p.m + 3)*r.x2 + p.m^2 + 2 = 0 ∧
   r.x1^2 + r.x2^2 = 3*r.x1*r.x2 - 14) →
  p.m = 13 := by
  sorry

end quadratic_equation_theorem_l2074_207447


namespace exactly_four_even_probability_l2074_207466

def num_dice : ℕ := 8
def num_even : ℕ := 4
def prob_even : ℚ := 2/3
def prob_odd : ℚ := 1/3

theorem exactly_four_even_probability :
  (Nat.choose num_dice num_even) * (prob_even ^ num_even) * (prob_odd ^ (num_dice - num_even)) = 1120/6561 := by
  sorry

end exactly_four_even_probability_l2074_207466


namespace increasing_function_inequality_l2074_207491

theorem increasing_function_inequality (f : ℝ → ℝ) (h_increasing : Monotone f) :
  (∀ x : ℝ, f 4 < f (2^x)) → {x : ℝ | x > 2}.Nonempty := by
  sorry

end increasing_function_inequality_l2074_207491


namespace smallest_n_for_grape_contest_l2074_207409

theorem smallest_n_for_grape_contest : ∃ (c : ℕ+), 
  (c : ℕ) * (89 - c + 1) = 2009 ∧ 
  89 ≥ 2 * (c - 1) ∧
  ∀ (n : ℕ), n < 89 → ¬(∃ (d : ℕ+), (d : ℕ) * (n - d + 1) = 2009 ∧ n ≥ 2 * (d - 1)) :=
by sorry

end smallest_n_for_grape_contest_l2074_207409


namespace eunjis_rank_l2074_207435

/-- Given that Minyoung arrived 33rd in a race and Eunji arrived 11 places after Minyoung,
    prove that Eunji's rank is 44th. -/
theorem eunjis_rank (minyoungs_rank : ℕ) (places_after : ℕ) 
  (h1 : minyoungs_rank = 33) 
  (h2 : places_after = 11) : 
  minyoungs_rank + places_after = 44 := by
  sorry

end eunjis_rank_l2074_207435


namespace triangle_inequality_l2074_207454

theorem triangle_inequality (a b c : ℝ) (x y z : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) 
  (h4 : 0 ≤ x ∧ x ≤ π) (h5 : 0 ≤ y ∧ y ≤ π) (h6 : 0 ≤ z ∧ z ≤ π) 
  (h7 : x + y + z = π) : 
  b * c + c * a - a * b < b * c * Real.cos x + c * a * Real.cos y + a * b * Real.cos z ∧
  b * c * Real.cos x + c * a * Real.cos y + a * b * Real.cos z ≤ (a^2 + b^2 + c^2) / 2 :=
by sorry

end triangle_inequality_l2074_207454


namespace rainfall_difference_l2074_207439

/-- The number of Mondays -/
def numMondays : ℕ := 7

/-- The number of Tuesdays -/
def numTuesdays : ℕ := 9

/-- The amount of rain on each Monday in centimeters -/
def rainPerMonday : ℝ := 1.5

/-- The amount of rain on each Tuesday in centimeters -/
def rainPerTuesday : ℝ := 2.5

/-- The difference in total rainfall between Tuesdays and Mondays -/
theorem rainfall_difference : 
  (numTuesdays : ℝ) * rainPerTuesday - (numMondays : ℝ) * rainPerMonday = 12 := by
  sorry

end rainfall_difference_l2074_207439


namespace intersection_of_lines_l2074_207433

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (1/5, 2/5)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := y = -3 * x + 1

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := y + 1 = 7 * x

theorem intersection_of_lines :
  let (x, y) := intersection_point
  (line1 x y ∧ line2 x y) ∧
  ∀ x' y', (line1 x' y' ∧ line2 x' y') → (x' = x ∧ y' = y) :=
by sorry

end intersection_of_lines_l2074_207433


namespace slope_angle_range_l2074_207463

/-- Given two lines L1 and L2, where L1 has slope k and y-intercept -b,
    and their intersection point M is in the first quadrant,
    prove that the slope angle α of L1 is between arctan(-2/3) and π/2. -/
theorem slope_angle_range (k b : ℝ) :
  let L1 := λ x y : ℝ => y = k * x - b
  let L2 := λ x y : ℝ => 2 * x + 3 * y - 6 = 0
  let M := (((3 * b + 6) / (2 + 3 * k)), ((6 * k + 2 * b) / (2 + 3 * k)))
  let α := Real.arctan k
  (M.1 > 0 ∧ M.2 > 0) → (α > Real.arctan (-2/3) ∧ α < π/2) := by
  sorry

end slope_angle_range_l2074_207463


namespace circle_covering_theorem_l2074_207415

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A set of n points in the plane -/
def PointSet (n : ℕ) := Fin n → Point

/-- A circle in the plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Predicate to check if a point is covered by a circle -/
def covered (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 ≤ c.radius^2

/-- Predicate to check if a set of points can be covered by a circle -/
def canBeCovered (S : Set Point) (c : Circle) : Prop :=
  ∀ p ∈ S, covered p c

theorem circle_covering_theorem (n : ℕ) (points : PointSet n) :
  (∀ (i j k : Fin n), ∃ (c : Circle), c.radius = 1 ∧ 
    canBeCovered {points i, points j, points k} c) →
  ∃ (c : Circle), c.radius = 1 ∧ canBeCovered (Set.range points) c :=
sorry

end circle_covering_theorem_l2074_207415


namespace sum_of_xy_is_30_l2074_207496

-- Define the matrix
def matrix (x y : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![2, 5, 10],
    ![4, x, y],
    ![4, y, x]]

-- State the theorem
theorem sum_of_xy_is_30 (x y : ℝ) (h1 : x ≠ y) (h2 : Matrix.det (matrix x y) = 0) :
  x + y = 30 := by
  sorry

end sum_of_xy_is_30_l2074_207496


namespace min_value_of_x_l2074_207406

theorem min_value_of_x (x : ℝ) (h1 : x > 0) (h2 : Real.log x / Real.log 3 ≥ 1 + (1/3) * (Real.log x / Real.log 3)) :
  x ≥ 3 * Real.sqrt 3 ∧ ∀ y : ℝ, y > 0 → Real.log y / Real.log 3 ≥ 1 + (1/3) * (Real.log y / Real.log 3) → y ≥ x :=
sorry

end min_value_of_x_l2074_207406


namespace min_value_trig_expression_l2074_207461

theorem min_value_trig_expression (γ δ : ℝ) :
  ∃ (min : ℝ), min = 36 ∧
  ∀ (γ' δ' : ℝ), (3 * Real.cos γ' + 4 * Real.sin δ' - 7)^2 + 
    (3 * Real.sin γ' + 4 * Real.cos δ' - 12)^2 ≥ min ∧
  ∃ (γ₀ δ₀ : ℝ), (3 * Real.cos γ₀ + 4 * Real.sin δ₀ - 7)^2 + 
    (3 * Real.sin γ₀ + 4 * Real.cos δ₀ - 12)^2 = min :=
by sorry

end min_value_trig_expression_l2074_207461


namespace initial_average_equals_correct_average_l2074_207405

/-- The number of values in the set -/
def n : ℕ := 10

/-- The correct average of the numbers -/
def correct_average : ℚ := 401/10

/-- The difference between the first incorrectly copied number and its actual value -/
def first_error : ℤ := 17

/-- The difference between the second incorrectly copied number and its actual value -/
def second_error : ℤ := 13 - 31

/-- The sum of all errors in the incorrectly copied numbers -/
def total_error : ℤ := first_error + second_error

theorem initial_average_equals_correct_average :
  let S := n * correct_average
  let initial_average := (S + total_error) / n
  initial_average = correct_average := by sorry

end initial_average_equals_correct_average_l2074_207405


namespace students_liking_both_desserts_l2074_207422

/-- Proves the number of students liking both apple pie and chocolate cake in a class --/
theorem students_liking_both_desserts 
  (total_students : ℕ) 
  (apple_pie_fans : ℕ) 
  (chocolate_cake_fans : ℕ) 
  (neither_fans : ℕ) 
  (only_cookies_fans : ℕ) 
  (h1 : total_students = 50)
  (h2 : apple_pie_fans = 22)
  (h3 : chocolate_cake_fans = 20)
  (h4 : neither_fans = 10)
  (h5 : only_cookies_fans = 5)
  : ∃ (both_fans : ℕ), both_fans = 7 := by
  sorry


end students_liking_both_desserts_l2074_207422


namespace range_of_x_when_not_p_range_of_m_for_not_q_sufficient_not_necessary_l2074_207476

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - x - 2 ≤ 0
def q (x m : ℝ) : Prop := x^2 - x - m^2 - m ≤ 0

-- Theorem for the range of x when ¬p is true
theorem range_of_x_when_not_p (x : ℝ) :
  ¬(p x) ↔ (x > 2 ∨ x < -1) :=
sorry

-- Theorem for the range of m when ¬q is a sufficient but not necessary condition for ¬p
theorem range_of_m_for_not_q_sufficient_not_necessary (m : ℝ) :
  (∀ x, ¬(q x m) → ¬(p x)) ∧ ¬(∀ x, q x m → p x) ↔ (m > 1 ∨ m < -2) :=
sorry

end range_of_x_when_not_p_range_of_m_for_not_q_sufficient_not_necessary_l2074_207476


namespace pineapple_shipping_cost_l2074_207411

/-- The shipping cost for a dozen pineapples, given the initial cost and total cost per pineapple. -/
theorem pineapple_shipping_cost 
  (initial_cost : ℚ)  -- Cost of each pineapple before shipping
  (total_cost : ℚ)    -- Total cost of each pineapple including shipping
  (h1 : initial_cost = 1.25)  -- Each pineapple costs $1.25
  (h2 : total_cost = 3)       -- Each pineapple ends up costing $3
  : (12 : ℚ) * (total_cost - initial_cost) = 21 := by
  sorry

#check pineapple_shipping_cost

end pineapple_shipping_cost_l2074_207411


namespace one_seventh_comparison_l2074_207469

theorem one_seventh_comparison : (1 : ℚ) / 7 - 142857142857 / 1000000000000 = 1 / (7 * 1000000000000) := by
  sorry

end one_seventh_comparison_l2074_207469


namespace quadratic_roots_condition_l2074_207434

theorem quadratic_roots_condition (a : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 + 2*x + a = 0 ∧ y^2 + 2*y + a = 0) → a < 1 :=
by sorry

end quadratic_roots_condition_l2074_207434


namespace point_location_l2074_207492

theorem point_location (x y : ℝ) (h : (x + y)^2 = x^2 + y^2 - 2) :
  (x * y = -1) ∧ (x > 0 ∧ y < 0 ∨ x < 0 ∧ y > 0) :=
by sorry

end point_location_l2074_207492


namespace simplify_and_evaluate_l2074_207471

theorem simplify_and_evaluate (a b : ℝ) (ha : a = Real.sqrt 3 - Real.sqrt 11) (hb : b = Real.sqrt 3 + Real.sqrt 11) :
  (a^2 - b^2) / (a^2 * b - a * b^2) / (1 + (a^2 + b^2) / (2 * a * b)) = Real.sqrt 3 / 3 := by
  sorry

end simplify_and_evaluate_l2074_207471


namespace brick_length_calculation_l2074_207486

theorem brick_length_calculation (courtyard_length courtyard_width : ℝ)
  (brick_width : ℝ) (total_bricks : ℕ) (h1 : courtyard_length = 18)
  (h2 : courtyard_width = 16) (h3 : brick_width = 0.1)
  (h4 : total_bricks = 14400) :
  let courtyard_area : ℝ := courtyard_length * courtyard_width * 10000
  let brick_area : ℝ := courtyard_area / total_bricks
  brick_area / brick_width = 20 := by sorry

end brick_length_calculation_l2074_207486


namespace ox_and_sheep_cost_l2074_207468

theorem ox_and_sheep_cost (ox sheep : ℚ) 
  (h1 : 5 * ox + 2 * sheep = 10) 
  (h2 : 2 * ox + 8 * sheep = 8) : 
  ox = 16/9 ∧ sheep = 5/9 := by
  sorry

end ox_and_sheep_cost_l2074_207468


namespace point_above_line_t_range_l2074_207494

-- Define the line
def line (x y : ℝ) : Prop := x - 2*y + 4 = 0

-- Define what it means for a point to be above the line
def above_line (x y : ℝ) : Prop := x - 2*y + 4 < 0

-- Theorem statement
theorem point_above_line_t_range :
  ∀ t : ℝ, above_line (-2) t → t > 1 :=
by sorry

end point_above_line_t_range_l2074_207494


namespace subset_implies_a_range_l2074_207426

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 2*x < 0}
def N (a : ℝ) : Set ℝ := {x | x < a}

-- State the theorem
theorem subset_implies_a_range (a : ℝ) : M ⊆ N a → a ∈ Set.Ici 2 := by
  sorry

end subset_implies_a_range_l2074_207426


namespace largest_number_l2074_207431

theorem largest_number (a b c d e : ℚ) 
  (ha : a = 995/1000) 
  (hb : b = 9995/10000) 
  (hc : c = 99/100) 
  (hd : d = 999/1000) 
  (he : e = 9959/10000) : 
  b > a ∧ b > c ∧ b > d ∧ b > e := by
  sorry

end largest_number_l2074_207431


namespace part_time_employees_count_l2074_207457

def total_employees : ℕ := 65134
def full_time_employees : ℕ := 63093

theorem part_time_employees_count : total_employees - full_time_employees = 2041 := by
  sorry

end part_time_employees_count_l2074_207457


namespace opposite_rational_division_l2074_207412

theorem opposite_rational_division (a : ℚ) : 
  (a ≠ 0 → a / (-a) = -1) ∧ (a = 0 → a / (-a) = 0/0) :=
sorry

end opposite_rational_division_l2074_207412


namespace perimeter_relations_l2074_207442

variable (n : ℕ+) (r : ℝ) (hr : r > 0)

/-- Perimeter of regular n-gon circumscribed around a circle with radius r -/
noncomputable def K (n : ℕ+) (r : ℝ) : ℝ := sorry

/-- Perimeter of regular n-gon inscribed in a circle with radius r -/
noncomputable def k (n : ℕ+) (r : ℝ) : ℝ := sorry

theorem perimeter_relations (n : ℕ+) (r : ℝ) (hr : r > 0) :
  (K (2 * n) r = (2 * K n r * k n r) / (K n r + k n r)) ∧
  (k (2 * n) r = Real.sqrt ((k n r) * (K (2 * n) r))) := by sorry

end perimeter_relations_l2074_207442


namespace line_perpendicular_to_plane_and_parallel_line_l2074_207436

structure Plane where

structure Line where

def perpendicular (l : Line) (p : Plane) : Prop := sorry

def parallel (l : Line) (p : Plane) : Prop := sorry

def perpendicular_lines (l1 l2 : Line) : Prop := sorry

theorem line_perpendicular_to_plane_and_parallel_line 
  (α : Plane) (m n : Line) 
  (h1 : perpendicular m α) 
  (h2 : parallel n α) : 
  perpendicular_lines m n := by sorry

end line_perpendicular_to_plane_and_parallel_line_l2074_207436


namespace remainder_problem_l2074_207403

theorem remainder_problem (n : ℕ) : 
  n % 101 = 0 ∧ n / 101 = 347 → n % 89 = 70 := by
  sorry

end remainder_problem_l2074_207403


namespace set_equality_l2074_207475

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def M : Set ℕ := {3, 4, 5}
def N : Set ℕ := {1, 3, 6}

theorem set_equality : (Uᶜ ∪ M) ∩ (Uᶜ ∪ N) = {2, 7} := by sorry

end set_equality_l2074_207475


namespace power_of_power_l2074_207401

theorem power_of_power (a : ℝ) : (a ^ 2) ^ 3 = a ^ 6 := by
  sorry

end power_of_power_l2074_207401


namespace asymptote_sum_l2074_207418

/-- Represents a rational function -/
structure RationalFunction where
  numerator : Polynomial ℝ
  denominator : Polynomial ℝ

/-- Counts the number of holes in the graph of a rational function -/
noncomputable def count_holes (f : RationalFunction) : ℕ := sorry

/-- Counts the number of vertical asymptotes of a rational function -/
noncomputable def count_vertical_asymptotes (f : RationalFunction) : ℕ := sorry

/-- Counts the number of horizontal asymptotes of a rational function -/
noncomputable def count_horizontal_asymptotes (f : RationalFunction) : ℕ := sorry

/-- Counts the number of oblique asymptotes of a rational function -/
noncomputable def count_oblique_asymptotes (f : RationalFunction) : ℕ := sorry

/-- The main theorem -/
theorem asymptote_sum (f : RationalFunction) 
  (h : f.numerator = Polynomial.X^2 + 4*Polynomial.X + 3 ∧ 
       f.denominator = Polynomial.X^3 + 2*Polynomial.X^2 - 3*Polynomial.X) : 
  count_holes f + 2 * count_vertical_asymptotes f + 
  3 * count_horizontal_asymptotes f + 4 * count_oblique_asymptotes f = 8 := by
  sorry

end asymptote_sum_l2074_207418


namespace decagon_diagonals_l2074_207407

/-- The number of internal diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon is a polygon with 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end decagon_diagonals_l2074_207407


namespace sqrt_a_squared_plus_one_is_quadratic_radical_l2074_207470

/-- A function is a quadratic radical if it involves a square root and its radicand is non-negative for all real inputs. -/
def is_quadratic_radical (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, (∀ x, g x ≥ 0) ∧ (∀ x, f x = Real.sqrt (g x))

/-- The function f(a) = √(a² + 1) is a quadratic radical. -/
theorem sqrt_a_squared_plus_one_is_quadratic_radical :
  is_quadratic_radical (fun a : ℝ ↦ Real.sqrt (a^2 + 1)) :=
sorry

end sqrt_a_squared_plus_one_is_quadratic_radical_l2074_207470


namespace area_EFGH_extended_l2074_207420

/-- Represents a quadrilateral with extended sides -/
structure ExtendedQuadrilateral where
  ef : ℝ
  fg : ℝ
  gh : ℝ
  he : ℝ
  area : ℝ

/-- Calculates the area of the extended quadrilateral -/
def area_extended_quadrilateral (q : ExtendedQuadrilateral) : ℝ :=
  q.area + 2 * q.area

/-- Theorem stating the area of the extended quadrilateral E'F'G'H' -/
theorem area_EFGH_extended (q : ExtendedQuadrilateral)
  (h_ef : q.ef = 5)
  (h_fg : q.fg = 6)
  (h_gh : q.gh = 7)
  (h_he : q.he = 8)
  (h_area : q.area = 20) :
  area_extended_quadrilateral q = 60 := by
  sorry

end area_EFGH_extended_l2074_207420


namespace number_equation_solution_l2074_207473

theorem number_equation_solution :
  ∃ x : ℝ, (3 * x = 2 * x - 7) ∧ (x = -7) := by sorry

end number_equation_solution_l2074_207473


namespace semi_circle_perimeter_specific_semi_circle_perimeter_l2074_207488

/-- The perimeter of a semi-circle with radius r is equal to π * r + 2r -/
theorem semi_circle_perimeter (r : ℝ) (h : r > 0) :
  let perimeter := π * r + 2 * r
  perimeter = π * r + 2 * r :=
by sorry

/-- The perimeter of a semi-circle with radius 6.6 cm is approximately 33.93 cm -/
theorem specific_semi_circle_perimeter :
  let r : ℝ := 6.6
  let perimeter := π * r + 2 * r
  ∃ (approx : ℝ), abs (perimeter - approx) < 0.005 ∧ approx = 33.93 :=
by sorry

end semi_circle_perimeter_specific_semi_circle_perimeter_l2074_207488


namespace sin_squared_minus_three_sin_plus_two_range_l2074_207497

theorem sin_squared_minus_three_sin_plus_two_range :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * Real.pi →
  ∃ y : ℝ, y = Real.sin x ^ 2 - 3 * Real.sin x + 2 ∧ 0 ≤ y ∧ y ≤ 6 :=
by sorry

end sin_squared_minus_three_sin_plus_two_range_l2074_207497
