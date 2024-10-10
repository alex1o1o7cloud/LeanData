import Mathlib

namespace commute_days_calculation_l1675_167545

theorem commute_days_calculation (morning_bus afternoon_bus train_commute : ℕ) 
  (h1 : morning_bus = 8)
  (h2 : afternoon_bus = 15)
  (h3 : train_commute = 9) :
  ∃ (morning_train afternoon_train both_bus : ℕ),
    morning_train + afternoon_train = train_commute ∧
    morning_bus = afternoon_train + both_bus ∧
    afternoon_bus = morning_train + both_bus ∧
    morning_train + afternoon_train + both_bus = 16 :=
by sorry

end commute_days_calculation_l1675_167545


namespace stock_price_change_l1675_167585

/-- The final stock price after two years of changes -/
def final_stock_price (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) : ℝ :=
  initial_price * (1 + first_year_increase) * (1 - second_year_decrease)

/-- Theorem stating the final stock price after specific changes -/
theorem stock_price_change : final_stock_price 80 1.2 0.3 = 123.2 := by
  sorry

end stock_price_change_l1675_167585


namespace greatest_divisor_of_sum_first_12_terms_l1675_167531

-- Define an arithmetic sequence of positive integers
def ArithmeticSequence (a : ℕ → ℕ) : Prop :=
  ∃ (x c : ℕ), ∀ n, a n = x + n * c

-- Define the sum of the first 12 terms
def SumFirst12Terms (a : ℕ → ℕ) : ℕ :=
  (List.range 12).map a |>.sum

-- Theorem statement
theorem greatest_divisor_of_sum_first_12_terms :
  ∀ a : ℕ → ℕ, ArithmeticSequence a →
  (∃ k : ℕ, k > 6 ∧ k ∣ SumFirst12Terms a) → False :=
sorry

end greatest_divisor_of_sum_first_12_terms_l1675_167531


namespace min_value_reciprocal_sum_l1675_167576

theorem min_value_reciprocal_sum (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h_sum : 2 * m + n = 1) :
  (1 / m + 2 / n) ≥ 8 ∧ ∃ m₀ n₀ : ℝ, 0 < m₀ ∧ 0 < n₀ ∧ 2 * m₀ + n₀ = 1 ∧ 1 / m₀ + 2 / n₀ = 8 :=
by sorry

end min_value_reciprocal_sum_l1675_167576


namespace geometric_series_equality_l1675_167563

def C (n : ℕ) : ℚ := 512 * (1 - (1/2)^n) / (1 - 1/2)

def D (n : ℕ) : ℚ := 1536 * (1 - (1/(-2))^n) / (1 + 1/2)

theorem geometric_series_equality :
  ∃ (n : ℕ), n > 0 ∧ C n = D n ∧ ∀ (m : ℕ), 0 < m ∧ m < n → C m ≠ D m :=
by sorry

end geometric_series_equality_l1675_167563


namespace largest_prime_divisor_of_101011101_base_7_l1675_167538

def base_seven_to_decimal (n : ℕ) : ℕ := 
  7^8 + 7^6 + 7^4 + 7^3 + 7^2 + 1

theorem largest_prime_divisor_of_101011101_base_7 :
  ∃ (p : ℕ), Prime p ∧ p ∣ base_seven_to_decimal 101011101 ∧
  ∀ (q : ℕ), Prime q → q ∣ base_seven_to_decimal 101011101 → q ≤ p :=
by sorry

end largest_prime_divisor_of_101011101_base_7_l1675_167538


namespace alternating_ball_probability_l1675_167579

def num_black_balls : ℕ := 5
def num_white_balls : ℕ := 4
def total_balls : ℕ := num_black_balls + num_white_balls

def alternating_sequence (n : ℕ) : List Bool :=
  List.map (fun i => i % 2 = 0) (List.range n)

def is_valid_sequence (seq : List Bool) : Prop :=
  seq.length = total_balls ∧
  seq.head? = some true ∧
  seq = alternating_sequence total_balls

def num_valid_sequences : ℕ := 1

def total_outcomes : ℕ := Nat.choose total_balls num_black_balls

theorem alternating_ball_probability :
  (num_valid_sequences : ℚ) / total_outcomes = 1 / 126 :=
sorry

end alternating_ball_probability_l1675_167579


namespace max_value_of_N_l1675_167591

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def last_two_digits (n : ℕ) : ℕ := n % 100

def remove_last_two_digits (n : ℕ) : ℕ := n / 100

theorem max_value_of_N :
  ∃ N : ℕ,
    is_perfect_square N ∧
    N ≥ 100 ∧
    last_two_digits N ≠ 0 ∧
    is_perfect_square (remove_last_two_digits N) ∧
    (∀ M : ℕ, 
      (is_perfect_square M ∧
       M ≥ 100 ∧
       last_two_digits M ≠ 0 ∧
       is_perfect_square (remove_last_two_digits M)) →
      M ≤ N) ∧
    N = 1681 :=
by sorry

end max_value_of_N_l1675_167591


namespace extension_point_coordinates_l1675_167560

/-- Given points A and B, and a point P on the extension of segment AB such that |AP| = 2|PB|, 
    prove that P has specific coordinates. -/
theorem extension_point_coordinates (A B P : ℝ × ℝ) : 
  A = (2, 3) → 
  B = (4, -3) → 
  (∃ t : ℝ, t > 1 ∧ P = A + t • (B - A)) →
  ‖P - A‖ = 2 * ‖P - B‖ →
  P = (6, -9) := by
  sorry

end extension_point_coordinates_l1675_167560


namespace square_perimeters_sum_l1675_167502

theorem square_perimeters_sum (x y : ℝ) (h1 : x^2 + y^2 = 125) (h2 : x^2 - y^2 = 65) :
  4*x + 4*y = 60 := by
  sorry

end square_perimeters_sum_l1675_167502


namespace area_of_triangle_QPO_l1675_167569

-- Define the points
variable (A B C D P Q O N M : Point)
-- Define the area of the parallelogram
variable (k : ℝ)

-- Define the conditions
def is_parallelogram (A B C D : Point) : Prop := sorry

def bisects (P Q R : Point) : Prop := sorry

def intersects (L₁ L₂ P : Point) : Prop := sorry

def area (shape : Set Point) : ℝ := sorry

-- State the theorem
theorem area_of_triangle_QPO 
  (h1 : is_parallelogram A B C D)
  (h2 : bisects D N C)
  (h3 : intersects D P B)
  (h4 : bisects C M D)
  (h5 : intersects C Q A)
  (h6 : intersects D P O)
  (h7 : intersects C Q O)
  (h8 : area {A, B, C, D} = k) :
  area {Q, P, O} = 9/8 * k := sorry

end area_of_triangle_QPO_l1675_167569


namespace b_share_is_correct_l1675_167547

/-- Represents the rental information for a person -/
structure RentalInfo where
  horses : ℕ
  months : ℕ

/-- Calculates the total horse-months for a given rental information -/
def horseMonths (info : RentalInfo) : ℕ :=
  info.horses * info.months

/-- Represents the pasture rental problem -/
structure PastureRental where
  totalRent : ℚ
  a : RentalInfo
  b : RentalInfo
  c : RentalInfo

/-- Calculates the total horse-months for all renters -/
def totalHorseMonths (rental : PastureRental) : ℕ :=
  horseMonths rental.a + horseMonths rental.b + horseMonths rental.c

/-- Calculates the rent per horse-month -/
def rentPerHorseMonth (rental : PastureRental) : ℚ :=
  rental.totalRent / totalHorseMonths rental

/-- Calculates the rent for a specific renter -/
def renterShare (rental : PastureRental) (renter : RentalInfo) : ℚ :=
  (rentPerHorseMonth rental) * (horseMonths renter)

/-- The main theorem stating b's share of the rent -/
theorem b_share_is_correct (rental : PastureRental) 
  (h1 : rental.totalRent = 841)
  (h2 : rental.a = ⟨12, 8⟩)
  (h3 : rental.b = ⟨16, 9⟩)
  (h4 : rental.c = ⟨18, 6⟩) :
  renterShare rental rental.b = 348.48 := by
  sorry

end b_share_is_correct_l1675_167547


namespace fifteen_choose_three_l1675_167537

theorem fifteen_choose_three : 
  Nat.choose 15 3 = 455 := by sorry

end fifteen_choose_three_l1675_167537


namespace book_cost_problem_l1675_167506

theorem book_cost_problem (total_cost : ℝ) (loss_percent : ℝ) (gain_percent : ℝ)
  (h_total : total_cost = 600)
  (h_loss : loss_percent = 15)
  (h_gain : gain_percent = 19) :
  ∃ (cost_loss cost_gain : ℝ),
    cost_loss + cost_gain = total_cost ∧
    cost_loss * (1 - loss_percent / 100) = cost_gain * (1 + gain_percent / 100) ∧
    cost_loss = 350 := by
  sorry

end book_cost_problem_l1675_167506


namespace distance_center_to_secant_l1675_167550

/-- Given a circle O with center (0, 0) and radius 5, a tangent line AD of length 4,
    and a secant line ABC where AC = 8, the distance from the center O to the line AC is 4. -/
theorem distance_center_to_secant (O A B C D : ℝ × ℝ) : 
  let r := 5
  let circle := {p : ℝ × ℝ | (p.1 - O.1)^2 + (p.2 - O.2)^2 = r^2}
  (A ∉ circle) →
  (B ∈ circle) →
  (C ∈ circle) →
  (D ∈ circle) →
  (∀ p ∈ circle, (p.1 - A.1) * (D.1 - A.1) + (p.2 - A.2) * (D.2 - A.2) = 0) →
  (Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) = 4) →
  (Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 8) →
  (abs ((O.2 - A.2) * (C.1 - A.1) - (O.1 - A.1) * (C.2 - A.2)) / 
   Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 4) :=
by sorry

end distance_center_to_secant_l1675_167550


namespace cricket_bats_profit_percentage_l1675_167556

/-- Calculate the overall profit percentage for three cricket bats --/
theorem cricket_bats_profit_percentage
  (selling_price_A selling_price_B selling_price_C : ℝ)
  (profit_A profit_B profit_C : ℝ)
  (h1 : selling_price_A = 900)
  (h2 : selling_price_B = 1200)
  (h3 : selling_price_C = 1500)
  (h4 : profit_A = 300)
  (h5 : profit_B = 400)
  (h6 : profit_C = 500) :
  let total_cost_price := (selling_price_A - profit_A) + (selling_price_B - profit_B) + (selling_price_C - profit_C)
  let total_selling_price := selling_price_A + selling_price_B + selling_price_C
  let total_profit := total_selling_price - total_cost_price
  (total_profit / total_cost_price) * 100 = 50 := by
  sorry

end cricket_bats_profit_percentage_l1675_167556


namespace consecutive_odd_squares_same_digit_l1675_167520

theorem consecutive_odd_squares_same_digit : ∃! (n : ℕ), 
  (∃ (d : ℕ), d ∈ Finset.range 10 ∧ 
    (n - 2)^2 + n^2 + (n + 2)^2 = 1111 * d) ∧
  Odd n ∧ n = 43 := by sorry

end consecutive_odd_squares_same_digit_l1675_167520


namespace inequality_preservation_l1675_167562

theorem inequality_preservation (x y : ℝ) (h : x > y) : x / 2 > y / 2 := by
  sorry

end inequality_preservation_l1675_167562


namespace alloy_mixing_theorem_l1675_167597

/-- Represents an alloy with two metals -/
structure Alloy where
  ratio1 : ℚ
  ratio2 : ℚ

/-- Creates a new alloy by mixing two existing alloys -/
def mixAlloys (a1 : Alloy) (p1 : ℚ) (a2 : Alloy) (p2 : ℚ) : Alloy :=
  { ratio1 := (a1.ratio1 * p1 + a2.ratio1 * p2) / (p1 + p2),
    ratio2 := (a1.ratio2 * p1 + a2.ratio2 * p2) / (p1 + p2) }

theorem alloy_mixing_theorem :
  let alloy1 : Alloy := { ratio1 := 1, ratio2 := 2 }
  let alloy2 : Alloy := { ratio1 := 2, ratio2 := 3 }
  let mixedAlloy := mixAlloys alloy1 9 alloy2 35
  mixedAlloy.ratio1 / mixedAlloy.ratio2 = 17 / 27 := by
  sorry

end alloy_mixing_theorem_l1675_167597


namespace prism_surface_area_l1675_167559

/-- A right square prism with all vertices on the surface of a sphere -/
structure PrismOnSphere where
  -- The diameter of the sphere
  sphere_diameter : ℝ
  -- The side length of the base of the prism
  base_side_length : ℝ
  -- The height of the prism
  height : ℝ
  -- All vertices are on the sphere surface
  vertices_on_sphere : sphere_diameter^2 = base_side_length^2 + base_side_length^2 + height^2

/-- The surface area of a right square prism -/
def surface_area (p : PrismOnSphere) : ℝ :=
  2 * p.base_side_length^2 + 4 * p.base_side_length * p.height

/-- Theorem: The surface area of the specific prism is 2 + 4√2 -/
theorem prism_surface_area :
  ∃ (p : PrismOnSphere),
    p.sphere_diameter = 2 ∧
    p.base_side_length = 1 ∧
    surface_area p = 2 + 4 * Real.sqrt 2 :=
by sorry

end prism_surface_area_l1675_167559


namespace original_page_count_l1675_167536

/-- Represents a book with numbered pages -/
structure Book where
  pages : ℕ

/-- Calculates the total number of digits in the page numbers of remaining pages after removing even-numbered sheets -/
def remainingDigits (b : Book) : ℕ := sorry

/-- Theorem stating the possible original page counts given the remaining digit count -/
theorem original_page_count (b : Book) : 
  remainingDigits b = 845 → b.pages = 598 ∨ b.pages = 600 := by sorry

end original_page_count_l1675_167536


namespace isabel_piggy_bank_l1675_167501

theorem isabel_piggy_bank (X : ℝ) : 
  (X > 0) → 
  ((1 - 0.25) * (1 / 2) * (2 / 3) * X = 60) → 
  (X = 720) := by
sorry

end isabel_piggy_bank_l1675_167501


namespace parrot_seed_consumption_l1675_167568

/-- Given a parrot that absorbs 40% of the seeds it consumes and absorbed 8 ounces of seeds,
    prove that the total amount of seeds consumed is 20 ounces and twice that amount is 40 ounces. -/
theorem parrot_seed_consumption (absorbed_percentage : ℝ) (absorbed_amount : ℝ) 
    (h1 : absorbed_percentage = 0.40)
    (h2 : absorbed_amount = 8) : 
  ∃ (total_consumed : ℝ), 
    total_consumed * absorbed_percentage = absorbed_amount ∧ 
    total_consumed = 20 ∧ 
    2 * total_consumed = 40 := by
  sorry


end parrot_seed_consumption_l1675_167568


namespace power_of_product_equals_product_of_powers_l1675_167553

theorem power_of_product_equals_product_of_powers (a : ℝ) :
  (3 * a^3)^2 = 9 * a^6 := by
  sorry

end power_of_product_equals_product_of_powers_l1675_167553


namespace roots_bound_implies_b_bound_l1675_167505

-- Define the function f(x) = x^2 + ax + b
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- Define the theorem
theorem roots_bound_implies_b_bound
  (a b x₁ x₂ : ℝ)
  (h1 : f a b x₁ = 0)  -- x₁ is a root of f
  (h2 : f a b x₂ = 0)  -- x₂ is a root of f
  (h3 : x₁ ≠ x₂)       -- The roots are distinct
  (h4 : |x₁| + |x₂| ≤ 2) :
  b ≤ 1 := by
sorry

end roots_bound_implies_b_bound_l1675_167505


namespace alice_yard_side_length_l1675_167526

/-- Given that Alice needs to buy 12 bushes to plant around three sides of her yard,
    and each bush fills 4 feet, prove that each side of her yard is 16 feet long. -/
theorem alice_yard_side_length
  (num_bushes : ℕ)
  (bush_length : ℕ)
  (num_sides : ℕ)
  (h1 : num_bushes = 12)
  (h2 : bush_length = 4)
  (h3 : num_sides = 3) :
  (num_bushes * bush_length) / num_sides = 16 := by
  sorry

end alice_yard_side_length_l1675_167526


namespace article_cost_l1675_167572

theorem article_cost (sell_price_1 sell_price_2 : ℝ) 
  (h1 : sell_price_1 = 380)
  (h2 : sell_price_2 = 420)
  (h3 : sell_price_2 - sell_price_1 = 0.05 * cost) : cost = 800 := by
  sorry

end article_cost_l1675_167572


namespace quadratic_equation_1_quadratic_equation_2_l1675_167565

-- Equation 1
theorem quadratic_equation_1 :
  ∃ x₁ x₂ : ℝ, x₁ = 1/3 ∧ x₂ = -1 ∧ 
  (3 * x₁^2 + 2 * x₁ - 1 = 0) ∧ 
  (3 * x₂^2 + 2 * x₂ - 1 = 0) :=
sorry

-- Equation 2
theorem quadratic_equation_2 :
  ∃ x : ℝ, x = 3 ∧ 
  (x + 2) * (x - 3) = 5 * x - 15 :=
sorry

end quadratic_equation_1_quadratic_equation_2_l1675_167565


namespace john_restringing_problem_l1675_167574

/-- The number of basses John needs to restring -/
def num_basses : ℕ := 3

/-- The number of guitars John needs to restring -/
def num_guitars : ℕ := 2 * num_basses

/-- The number of 8-string guitars John needs to restring -/
def num_8string_guitars : ℕ := num_guitars - 3

/-- The total number of strings needed -/
def total_strings : ℕ := 72

theorem john_restringing_problem :
  4 * num_basses + 6 * num_guitars + 8 * num_8string_guitars = total_strings :=
by sorry

end john_restringing_problem_l1675_167574


namespace constant_ratio_problem_l1675_167507

theorem constant_ratio_problem (x y : ℝ) (k : ℝ) : 
  ((5 * x - 3) / (2 * y + 10) = k) →  -- The ratio is constant
  (y = 2 → x = 3) →                   -- When y = 2, x = 3
  (y = 5 → x = 47 / 5) :=             -- When y = 5, x = 47/5
by
  sorry

end constant_ratio_problem_l1675_167507


namespace one_less_than_negative_two_l1675_167504

theorem one_less_than_negative_two : -2 - 1 = -3 := by
  sorry

end one_less_than_negative_two_l1675_167504


namespace equal_roots_implies_c_equals_one_fourth_l1675_167534

-- Define the quadratic equation
def quadratic_equation (x c : ℝ) : Prop := x^2 + x + c = 0

-- Define the condition for two equal real roots
def has_two_equal_real_roots (c : ℝ) : Prop :=
  ∃ x : ℝ, quadratic_equation x c ∧ 
    ∀ y : ℝ, quadratic_equation y c → y = x

-- Theorem statement
theorem equal_roots_implies_c_equals_one_fourth :
  ∀ c : ℝ, has_two_equal_real_roots c → c = 1/4 :=
by sorry

end equal_roots_implies_c_equals_one_fourth_l1675_167534


namespace nabla_sum_equals_32_l1675_167586

-- Define the ∇ operation
def nabla (k m : ℕ) : ℕ := k * (k - m)

-- State the theorem
theorem nabla_sum_equals_32 : nabla 5 1 + nabla 4 1 = 32 := by
  sorry

end nabla_sum_equals_32_l1675_167586


namespace final_women_count_l1675_167575

/-- Represents the number of people in each category --/
structure Population :=
  (men : ℕ)
  (women : ℕ)
  (children : ℕ)
  (elderly : ℕ)

/-- Theorem stating the final number of women in the room --/
theorem final_women_count (initial : Population) 
  (h1 : initial.men + initial.women + initial.children + initial.elderly > 0)
  (h2 : initial.men = 4 * initial.elderly / 2)
  (h3 : initial.women = 5 * initial.elderly / 2)
  (h4 : initial.children = 3 * initial.elderly / 2)
  (h5 : initial.men + 2 = 14)
  (h6 : initial.children - 5 = 7)
  (h7 : initial.elderly - 3 = 6) :
  2 * (initial.women - 3) = 24 := by
  sorry

#check final_women_count

end final_women_count_l1675_167575


namespace sum_of_irrationals_can_be_rational_l1675_167521

theorem sum_of_irrationals_can_be_rational :
  ∃ (x y : ℝ), Irrational x ∧ Irrational y ∧ ∃ (q : ℚ), x + y = q := by
  sorry

end sum_of_irrationals_can_be_rational_l1675_167521


namespace theater_ticket_pricing_l1675_167529

/-- Theorem: Theater Ticket Pricing --/
theorem theater_ticket_pricing
  (total_tickets : ℕ)
  (total_cost : ℕ)
  (orchestra_cost : ℕ)
  (balcony_surplus : ℕ)
  (h1 : total_tickets = 370)
  (h2 : total_cost = 3320)
  (h3 : orchestra_cost = 12)
  (h4 : balcony_surplus = 190)
  : ∃ (balcony_cost : ℕ),
    balcony_cost = 8 ∧
    balcony_cost * (total_tickets - (total_tickets - balcony_surplus) / 2) +
    orchestra_cost * ((total_tickets - balcony_surplus) / 2) = total_cost :=
by sorry


end theater_ticket_pricing_l1675_167529


namespace sqrt_112_between_consecutive_integers_product_l1675_167558

theorem sqrt_112_between_consecutive_integers_product : ∃ (n : ℕ), 
  n > 0 ∧ 
  n^2 < 112 ∧ 
  (n + 1)^2 > 112 ∧ 
  n * (n + 1) = 110 := by
  sorry

end sqrt_112_between_consecutive_integers_product_l1675_167558


namespace spheres_theorem_l1675_167596

/-- The configuration of four spheres -/
structure SpheresConfiguration where
  r : ℝ  -- radius of the three smaller spheres
  R : ℝ  -- radius of the larger sphere
  h : R > r  -- condition that R is greater than r

/-- The condition for the configuration to be possible -/
def configuration_possible (c : SpheresConfiguration) : Prop :=
  c.R ≥ (2 / Real.sqrt 3 - 1) * c.r

/-- The radius of the sphere tangent to all four spheres -/
noncomputable def tangent_sphere_radius (c : SpheresConfiguration) : ℝ :=
  let numerator := c.R * (c.R + c.r - Real.sqrt (c.R^2 + 2*c.R*c.r - c.r^2/3))
  let denominator := c.r + Real.sqrt (c.R^2 + 2*c.R*c.r - c.r^2/3) - c.R
  numerator / denominator

/-- The main theorem stating the conditions and the radius of the tangent sphere -/
theorem spheres_theorem (c : SpheresConfiguration) :
  configuration_possible c ∧
  tangent_sphere_radius c = (c.R * (c.R + c.r - Real.sqrt (c.R^2 + 2*c.R*c.r - c.r^2/3))) /
                            (c.r + Real.sqrt (c.R^2 + 2*c.R*c.r - c.r^2/3) - c.R) := by
  sorry

end spheres_theorem_l1675_167596


namespace arctan_tan_difference_l1675_167589

/-- Proves that arctan(tan 70° - 2 tan 45°) = 135° --/
theorem arctan_tan_difference (θ : Real) : 
  θ ∈ Set.Icc 0 180 ∧ 
  θ * (π / 180) = Real.arctan (Real.tan (70 * π / 180) - 2 * Real.tan (45 * π / 180)) → 
  θ = 135 := by
  sorry

#check arctan_tan_difference

end arctan_tan_difference_l1675_167589


namespace yellow_balloon_ratio_l1675_167523

theorem yellow_balloon_ratio (total_balloons : ℕ) (num_colors : ℕ) (anya_balloons : ℕ) : 
  total_balloons = 672 →
  num_colors = 4 →
  anya_balloons = 84 →
  (anya_balloons : ℚ) / (total_balloons / num_colors) = 1 / 2 := by
  sorry

end yellow_balloon_ratio_l1675_167523


namespace fraction_invariance_l1675_167594

theorem fraction_invariance (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (2008 * (2 * x)) / (2007 * (2 * y)) = (2008 * x) / (2007 * y) := by
sorry

end fraction_invariance_l1675_167594


namespace sqrt_expression_equality_fraction_simplification_l1675_167548

-- Problem 1
theorem sqrt_expression_equality : 
  Real.sqrt 12 + Real.sqrt 3 * (Real.sqrt 2 - 1) = Real.sqrt 3 + Real.sqrt 6 := by
  sorry

-- Problem 2
theorem fraction_simplification (a b : ℝ) (ha : a ≠ 0) :
  (a + (2 * a * b + b^2) / a) / ((a + b) / a) = a + b := by
  sorry

end sqrt_expression_equality_fraction_simplification_l1675_167548


namespace min_squares_to_exceed_500_l1675_167527

def square (n : ℕ) : ℕ := n * n

def repeated_square (n : ℕ) (k : ℕ) : ℕ :=
  match k with
  | 0 => n
  | k + 1 => square (repeated_square n k)

theorem min_squares_to_exceed_500 :
  (∃ k : ℕ, repeated_square 2 k > 500) ∧
  (∀ k : ℕ, k < 4 → repeated_square 2 k ≤ 500) ∧
  (repeated_square 2 4 > 500) :=
by sorry

end min_squares_to_exceed_500_l1675_167527


namespace parabola_shift_theorem_l1675_167593

/-- A parabola is a function of the form f(x) = a(x - h)^2 + k, where (h, k) is the vertex -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Shifting a parabola horizontally and vertically -/
def shift (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a
    h := p.h - dx
    k := p.k - dy }

theorem parabola_shift_theorem (p : Parabola) :
  p.a = 3 ∧ p.h = 2 ∧ p.k = 1 →
  let p' := shift p 2 1
  p'.a = 3 ∧ p'.h = 0 ∧ p'.k = 0 := by
  sorry

#check parabola_shift_theorem

end parabola_shift_theorem_l1675_167593


namespace set_operations_l1675_167540

-- Define the sets A and B
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3}

-- State the theorem
theorem set_operations :
  (A ∪ B = {0, 1, 2, 3}) ∧ (A ∩ B = {1, 2}) := by
  sorry

end set_operations_l1675_167540


namespace files_deleted_l1675_167533

/-- Given Dave's initial and final number of files, prove the number of files deleted. -/
theorem files_deleted (initial_files final_files : ℕ) 
  (h1 : initial_files = 24)
  (h2 : final_files = 21) :
  initial_files - final_files = 3 := by
  sorry

#check files_deleted

end files_deleted_l1675_167533


namespace fo_greater_than_di_l1675_167500

-- Define the points
variable (F I D O : ℝ × ℝ)

-- Define the quadrilateral FIDO
def is_convex_quadrilateral (F I D O : ℝ × ℝ) : Prop := sorry

-- Define the length of a line segment
def length (A B : ℝ × ℝ) : ℝ := sorry

-- Define the angle between two line segments
def angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem fo_greater_than_di 
  (h_convex : is_convex_quadrilateral F I D O)
  (h_equal_sides : length F I = length D O)
  (h_fi_greater : length F I > length D I)
  (h_equal_angles : angle F I O = angle D I O) :
  length F O > length D I :=
sorry

end fo_greater_than_di_l1675_167500


namespace prob_adjacent_is_half_l1675_167564

/-- The number of ways to arrange n distinct objects. -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange four students with two specific students adjacent. -/
def adjacent_arrangements : ℕ := 2 * (permutations 3)

/-- The total number of ways to arrange four students. -/
def total_arrangements : ℕ := permutations 4

/-- The probability of two specific students being adjacent in a line of four students. -/
def prob_adjacent : ℚ := adjacent_arrangements / total_arrangements

theorem prob_adjacent_is_half : prob_adjacent = 1/2 := by sorry

end prob_adjacent_is_half_l1675_167564


namespace train_crossing_time_l1675_167598

/-- Given a train crossing a platform, calculate the time it takes to cross a signal pole. -/
theorem train_crossing_time (train_length platform_length platform_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_length = 1162.5)
  (h3 : platform_crossing_time = 39)
  : (train_length / ((train_length + platform_length) / platform_crossing_time)) = 8 := by
  sorry

end train_crossing_time_l1675_167598


namespace eva_test_probability_l1675_167513

theorem eva_test_probability (p_history : ℝ) (p_geography : ℝ) 
  (h_history : p_history = 5/9)
  (h_geography : p_geography = 1/3)
  (h_independent : True) -- We don't need to define independence formally for this statement
  : (1 - p_history) * (1 - p_geography) = 8/27 := by
  sorry

end eva_test_probability_l1675_167513


namespace solve_equation_l1675_167570

theorem solve_equation : ∃ y : ℕ, 400 + 2 * 20 * 5 + 25 = y ∧ y = 625 := by sorry

end solve_equation_l1675_167570


namespace largest_geometric_three_digit_l1675_167555

/-- Represents a three-digit number as a tuple of its digits -/
def ThreeDigitNumber := (Nat × Nat × Nat)

/-- Checks if all digits in a ThreeDigitNumber are distinct -/
def distinct_digits (n : ThreeDigitNumber) : Prop :=
  n.1 ≠ n.2.1 ∧ n.1 ≠ n.2.2 ∧ n.2.1 ≠ n.2.2

/-- Checks if the digits of a ThreeDigitNumber form a geometric sequence -/
def geometric_sequence (n : ThreeDigitNumber) : Prop :=
  ∃ r : Rat, r ≠ 0 ∧ n.2.1 = n.1 * r ∧ n.2.2 = n.2.1 * r

/-- Checks if a ThreeDigitNumber has no zero digits -/
def no_zero_digits (n : ThreeDigitNumber) : Prop :=
  n.1 ≠ 0 ∧ n.2.1 ≠ 0 ∧ n.2.2 ≠ 0

/-- Converts a ThreeDigitNumber to its integer representation -/
def to_int (n : ThreeDigitNumber) : Nat :=
  100 * n.1 + 10 * n.2.1 + n.2.2

/-- The main theorem stating that 842 is the largest number satisfying all conditions -/
theorem largest_geometric_three_digit :
  ∀ n : ThreeDigitNumber,
    distinct_digits n ∧ 
    geometric_sequence n ∧ 
    no_zero_digits n →
    to_int n ≤ 842 :=
  sorry

end largest_geometric_three_digit_l1675_167555


namespace problem_figure_perimeter_l1675_167517

/-- Represents a rectangle with integer side lengths -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents the figure described in the problem -/
structure Figure where
  bottom_row : Vector Rectangle 3
  middle_square : Rectangle
  side_rectangles : Vector Rectangle 2

/-- The figure described in the problem -/
def problem_figure : Figure := {
  bottom_row := ⟨[{width := 1, height := 1}, {width := 1, height := 1}, {width := 1, height := 1}], rfl⟩
  middle_square := {width := 1, height := 1}
  side_rectangles := ⟨[{width := 1, height := 2}, {width := 1, height := 2}], rfl⟩
}

/-- Calculates the perimeter of the given figure -/
def perimeter (f : Figure) : ℕ :=
  sorry

/-- Theorem stating that the perimeter of the problem figure is 13 -/
theorem problem_figure_perimeter : perimeter problem_figure = 13 :=
  sorry

end problem_figure_perimeter_l1675_167517


namespace class_size_problem_l1675_167583

theorem class_size_problem (x : ℕ) (n : ℕ) : 
  20 < x ∧ x < 30 ∧ 
  n = (0.20 : ℝ) * (5 * n) ∧
  n = (0.25 : ℝ) * (4 * n) ∧
  x = 8 * n + 2 →
  x = 26 := by sorry

end class_size_problem_l1675_167583


namespace quadratic_inequality_solution_sets_l1675_167525

/-- Given that the solution set of ax^2 + bx + c > 0 is {x | 2 < x < 3},
    prove that the solution set of ax^2 - bx + c > 0 is {x | -3 < x < -2} -/
theorem quadratic_inequality_solution_sets
  (a b c : ℝ)
  (h : ∀ x, ax^2 + b*x + c > 0 ↔ 2 < x ∧ x < 3) :
  ∀ x, ax^2 - b*x + c > 0 ↔ -3 < x ∧ x < -2 :=
sorry

end quadratic_inequality_solution_sets_l1675_167525


namespace absent_children_l1675_167532

theorem absent_children (total_children : ℕ) (total_bananas : ℕ) : 
  total_children = 610 →
  total_bananas = 610 * 2 →
  total_bananas = (610 - (total_children - (610 - 305))) * 4 →
  610 - 305 = total_children - (610 - 305) :=
by
  sorry

end absent_children_l1675_167532


namespace expression_simplification_l1675_167580

theorem expression_simplification (a : ℝ) (h1 : a^2 - 4*a + 3 = 0) (h2 : a ≠ 3) :
  (a^2 - 9) / (a^2 - 3*a) / ((a^2 + 9) / a + 6) = 1/4 := by
  sorry

end expression_simplification_l1675_167580


namespace triangle_isosceles_condition_l1675_167577

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if a/cos(A) = b/cos(B), then the triangle is isosceles. -/
theorem triangle_isosceles_condition (a b c A B C : ℝ) :
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →  -- Angles are in (0, π)
  A + B + C = π →  -- Sum of angles in a triangle
  a / Real.cos A = b / Real.cos B →  -- Given condition
  a = b  -- Conclusion: triangle is isosceles
  := by sorry

end triangle_isosceles_condition_l1675_167577


namespace restaurant_bill_theorem_l1675_167503

theorem restaurant_bill_theorem (num_teenagers : ℕ) (avg_meal_cost : ℝ) (gratuity_rate : ℝ) :
  num_teenagers = 7 →
  avg_meal_cost = 100 →
  gratuity_rate = 0.20 →
  let total_before_gratuity := num_teenagers * avg_meal_cost
  let gratuity := total_before_gratuity * gratuity_rate
  let total_bill := total_before_gratuity + gratuity
  total_bill = 840 := by sorry

end restaurant_bill_theorem_l1675_167503


namespace book_price_problem_l1675_167571

theorem book_price_problem (n : ℕ) (d : ℝ) (middle_price : ℝ) : 
  n = 40 → d = 3 → middle_price = 75 → 
  ∃ (first_price : ℝ), 
    (∀ i : ℕ, i ≤ n → 
      (first_price + d * (i - 1) = middle_price) ↔ i = n / 2) ∧
    first_price = 18 :=
by sorry

end book_price_problem_l1675_167571


namespace emily_garden_seeds_l1675_167595

theorem emily_garden_seeds (total_seeds : ℕ) (big_garden_seeds : ℕ) (small_gardens : ℕ) 
  (h1 : total_seeds = 41)
  (h2 : big_garden_seeds = 29)
  (h3 : small_gardens = 3)
  (h4 : small_gardens > 0) :
  (total_seeds - big_garden_seeds) / small_gardens = 4 := by
sorry

end emily_garden_seeds_l1675_167595


namespace regular_price_is_0_15_l1675_167518

/-- The regular price per can of soda -/
def regular_price : ℝ := sorry

/-- The discounted price per can when purchased in 24-can cases -/
def discounted_price : ℝ := 0.9 * regular_price

/-- The price of 75 cans purchased in 24-can cases -/
def price_75_cans : ℝ := 10.125

theorem regular_price_is_0_15 : regular_price = 0.15 := by
  sorry

end regular_price_is_0_15_l1675_167518


namespace product_from_hcf_lcm_l1675_167524

theorem product_from_hcf_lcm (a b : ℕ+) (h1 : Nat.gcd a b = 11) (h2 : Nat.lcm a b = 181) :
  a * b = 1991 := by
  sorry

end product_from_hcf_lcm_l1675_167524


namespace andy_total_distance_l1675_167544

/-- The total distance Andy walks given his trips to school and market -/
def total_distance (house_to_school market_to_house : ℕ) : ℕ :=
  2 * house_to_school + market_to_house

/-- Theorem stating the total distance Andy walks -/
theorem andy_total_distance :
  let house_to_school := 50
  let house_to_market := 40
  total_distance house_to_school house_to_market = 140 := by
  sorry

end andy_total_distance_l1675_167544


namespace deepak_age_l1675_167508

/-- Proves that Deepak's current age is 42 years given the specified conditions --/
theorem deepak_age (arun deepak kamal : ℕ) : 
  arun * 7 = deepak * 5 →
  kamal * 5 = deepak * 9 →
  arun + 6 = 36 →
  kamal + 6 = 2 * (deepak + 6) →
  deepak = 42 := by
sorry

end deepak_age_l1675_167508


namespace remaining_amount_proof_l1675_167509

-- Define the deposit percentage
def deposit_percentage : ℚ := 10 / 100

-- Define the deposit amount
def deposit_amount : ℚ := 55

-- Define the total cost
def total_cost : ℚ := deposit_amount / deposit_percentage

-- Define the remaining amount to be paid
def remaining_amount : ℚ := total_cost - deposit_amount

-- Theorem to prove
theorem remaining_amount_proof : remaining_amount = 495 := by
  sorry

end remaining_amount_proof_l1675_167509


namespace cherry_picking_time_l1675_167554

/-- The time spent picking cherries by 王芳 and 李丽 -/
def picking_time : ℝ := 0.25

/-- 王芳's picking rate in kg/hour -/
def wang_rate : ℝ := 8

/-- 李丽's picking rate in kg/hour -/
def li_rate : ℝ := 7

/-- Amount of cherries 王芳 gives to 李丽 after picking -/
def transfer_amount : ℝ := 0.25

theorem cherry_picking_time :
  wang_rate * picking_time - transfer_amount = li_rate * picking_time :=
by sorry

end cherry_picking_time_l1675_167554


namespace total_stickers_count_l1675_167599

/-- The number of stickers on each page -/
def stickers_per_page : ℕ := 10

/-- The number of pages -/
def number_of_pages : ℕ := 22

/-- The total number of stickers -/
def total_stickers : ℕ := stickers_per_page * number_of_pages

theorem total_stickers_count : total_stickers = 220 := by
  sorry

end total_stickers_count_l1675_167599


namespace unique_solution_square_sum_l1675_167542

theorem unique_solution_square_sum (x y : ℝ) : 
  (x - 2*y)^2 + (y - 1)^2 = 0 ↔ x = 2 ∧ y = 1 := by
sorry

end unique_solution_square_sum_l1675_167542


namespace same_label_probability_l1675_167510

def deck_size : ℕ := 50
def num_labels : ℕ := 13
def cards_per_label (i : ℕ) : ℕ :=
  if i < num_labels then 4 else if i = num_labels then 2 else 0

def total_combinations : ℕ := deck_size.choose 2

def favorable_combinations : ℕ :=
  (Finset.range num_labels).sum (λ i => (cards_per_label i).choose 2)

theorem same_label_probability :
  (favorable_combinations : ℚ) / total_combinations = 73 / 1225 := by sorry

end same_label_probability_l1675_167510


namespace sum_of_common_ratios_l1675_167522

/-- Given two nonconstant geometric sequences with different common ratios,
    prove that the sum of their common ratios is equal to k. -/
theorem sum_of_common_ratios (k a₂ a₃ b₂ b₃ : ℝ) 
  (hk : k ≠ 0)
  (ha : a₂ ≠ k ∧ a₃ ≠ a₂)  -- Ensures (k, a₂, a₃) is nonconstant
  (hb : b₂ ≠ k ∧ b₃ ≠ b₂)  -- Ensures (k, b₂, b₃) is nonconstant
  (hdiff : a₂ / k ≠ b₂ / k)  -- Ensures different common ratios
  (heq : a₃ - b₃ = k^2 * (a₂ - b₂)) :
  ∃ p q : ℝ, p ≠ q ∧ 
    a₃ = k * p^2 ∧ 
    b₃ = k * q^2 ∧ 
    a₂ = k * p ∧ 
    b₂ = k * q ∧ 
    p + q = k :=
by sorry

end sum_of_common_ratios_l1675_167522


namespace upstream_distance_is_48_l1675_167549

/-- Represents the problem of calculating the upstream distance rowed --/
def UpstreamRowingProblem (downstream_distance : ℝ) (time : ℝ) (stream_speed : ℝ) : Prop :=
  ∃ (upstream_distance : ℝ) (boat_speed : ℝ),
    downstream_distance = (boat_speed + stream_speed) * time ∧
    upstream_distance = (boat_speed - stream_speed) * time ∧
    upstream_distance = 48

/-- Theorem stating that given the problem conditions, the upstream distance is 48 km --/
theorem upstream_distance_is_48 :
  UpstreamRowingProblem 84 2 9 :=
sorry

end upstream_distance_is_48_l1675_167549


namespace time_interval_is_20_minutes_l1675_167590

/-- The time interval between cars given total time and number of cars -/
def time_interval (total_time_hours : ℕ) (num_cars : ℕ) : ℚ :=
  (total_time_hours * 60 : ℚ) / num_cars

/-- Theorem: The time interval between cars is 20 minutes -/
theorem time_interval_is_20_minutes :
  time_interval 10 30 = 20 := by
  sorry

end time_interval_is_20_minutes_l1675_167590


namespace inequalities_satisfied_l1675_167551

theorem inequalities_satisfied
  (x y z : ℝ) (a b c : ℕ)
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0)
  (hxa : x < a) (hyb : y < b) (hzc : z < c) :
  (x * y + y * z + z * x < a * b + b * c + c * a) ∧
  (x^2 + y^2 + z^2 < a^2 + b^2 + c^2) ∧
  (x * y * z < a * b * c) := by
  sorry

end inequalities_satisfied_l1675_167551


namespace circle_properties_l1675_167516

/-- Given a circle C with equation x^2 + y^2 - 2x - 2y - 2 = 0,
    prove that its radius is 2 and its center is at (1, 1) -/
theorem circle_properties (x y : ℝ) :
  x^2 + y^2 - 2*x - 2*y - 2 = 0 →
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, 1) ∧ radius = 2 ∧
    (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end circle_properties_l1675_167516


namespace book_chapters_l1675_167578

theorem book_chapters (total_pages : ℕ) (pages_per_chapter : ℕ) 
  (h1 : total_pages = 1891) 
  (h2 : pages_per_chapter = 61) : 
  total_pages / pages_per_chapter = 31 := by
  sorry

end book_chapters_l1675_167578


namespace weekly_wage_calculation_l1675_167515

def basic_daily_wage : ℕ := 200
def basic_task_quantity : ℕ := 40
def reward_per_excess : ℕ := 7
def deduction_per_incomplete : ℕ := 8
def work_days : ℕ := 5
def production_deviations : List ℤ := [5, -2, -1, 0, 4]

def total_weekly_wage : ℕ := 1039

theorem weekly_wage_calculation :
  (basic_daily_wage * work_days) +
  (production_deviations.filter (λ x => x > 0)).sum * reward_per_excess -
  (production_deviations.filter (λ x => x < 0)).sum.natAbs * deduction_per_incomplete =
  total_weekly_wage :=
sorry

end weekly_wage_calculation_l1675_167515


namespace pyramid_frustum_volume_ratio_l1675_167582

theorem pyramid_frustum_volume_ratio : 
  let base_edge : ℝ := 24
  let altitude : ℝ := 18
  let small_altitude : ℝ := altitude / 3
  let original_volume : ℝ := (1 / 3) * (base_edge ^ 2) * altitude
  let small_volume : ℝ := (1 / 3) * ((small_altitude / altitude) * base_edge) ^ 2 * small_altitude
  let frustum_volume : ℝ := original_volume - small_volume
  frustum_volume / original_volume = 32 / 33 := by sorry

end pyramid_frustum_volume_ratio_l1675_167582


namespace can_form_triangle_l1675_167512

/-- Triangle Inequality Theorem: The sum of the lengths of any two sides of a 
    triangle must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Given lengths 5, 3, and 4 can form a triangle. -/
theorem can_form_triangle : triangle_inequality 5 3 4 := by
  sorry

end can_form_triangle_l1675_167512


namespace existence_of_three_quadratic_polynomials_l1675_167546

theorem existence_of_three_quadratic_polynomials :
  ∃ (p₁ p₂ p₃ : ℝ → ℝ),
    (∃ x₁, p₁ x₁ = 0) ∧
    (∃ x₂, p₂ x₂ = 0) ∧
    (∃ x₃, p₃ x₃ = 0) ∧
    (∀ x, p₁ x + p₂ x ≠ 0) ∧
    (∀ x, p₁ x + p₃ x ≠ 0) ∧
    (∀ x, p₂ x + p₃ x ≠ 0) ∧
    (∀ x, p₁ x = (x^2 : ℝ)) ∧
    (∀ x, p₂ x = ((x - 1)^2 : ℝ)) ∧
    (∀ x, p₃ x = ((x - 2)^2 : ℝ)) :=
by sorry

end existence_of_three_quadratic_polynomials_l1675_167546


namespace square_area_calculation_l1675_167539

theorem square_area_calculation (side_length : ℝ) (h : side_length = 28) :
  side_length ^ 2 = 784 := by
  sorry

#check square_area_calculation

end square_area_calculation_l1675_167539


namespace tree_height_after_three_years_l1675_167588

/-- The height of a tree after n years, given its initial height and growth factors -/
def tree_height (initial_height : ℝ) (n : ℕ) : ℝ :=
  if n ≤ 4 then
    initial_height * 3^n
  else
    initial_height * 3^4 * 2^(n - 4)

/-- Theorem: If a tree reaches 648 feet after 7 years with the given growth pattern,
    its height after 3 years was 27 feet -/
theorem tree_height_after_three_years
  (h : tree_height (tree_height 1 3) 4 = 648) :
  tree_height 1 3 = 27 := by
  sorry

end tree_height_after_three_years_l1675_167588


namespace circle_translation_l1675_167573

-- Define the original circle
def original_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the translated circle
def translated_circle (x y : ℝ) : Prop := (x+1)^2 + (y-2)^2 = 1

-- Define the translation vector
def translation_vector : ℝ × ℝ := (-1, 2)

-- Theorem statement
theorem circle_translation :
  ∀ (x y : ℝ), original_circle x y ↔ translated_circle (x + translation_vector.1) (y + translation_vector.2) :=
by sorry

end circle_translation_l1675_167573


namespace cannot_obtain_five_equal_numbers_l1675_167552

/-- Represents the set of numbers on the board -/
def BoardNumbers : Finset Int := {2, 3, 5, 7, 11}

/-- The operation of replacing two numbers with their arithmetic mean -/
def replaceWithMean (a b : Int) : Int := (a + b) / 2

/-- Predicate to check if two numbers have the same parity -/
def sameParity (a b : Int) : Prop := a % 2 = b % 2

/-- Theorem stating that it's impossible to obtain five equal numbers -/
theorem cannot_obtain_five_equal_numbers :
  ¬ ∃ (n : Int), ∃ (k : ℕ), ∃ (operations : Fin k → Int × Int),
    (∀ i, sameParity (operations i).1 (operations i).2) ∧
    (Finset.sum BoardNumbers id = 5 * n) ∧
    (∀ x ∈ BoardNumbers, x = n) :=
sorry

end cannot_obtain_five_equal_numbers_l1675_167552


namespace tan_pi_minus_alpha_eq_neg_two_implies_result_l1675_167567

theorem tan_pi_minus_alpha_eq_neg_two_implies_result (α : ℝ) 
  (h : Real.tan (π - α) = -2) : 
  1 / (Real.cos (2 * α) + Real.cos α ^ 2) = -5/2 := by
  sorry

end tan_pi_minus_alpha_eq_neg_two_implies_result_l1675_167567


namespace coefficient_b_value_l1675_167528

-- Define the polynomial P(x)
def P (a b d c : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + d*x + c

-- Define the sum of zeros
def sum_of_zeros (a : ℝ) : ℝ := -a

-- Define the product of zeros taken three at a time
def product_of_three_zeros (d : ℝ) : ℝ := d

-- Define the sum of coefficients
def sum_of_coefficients (a b d c : ℝ) : ℝ := 1 + a + b + d + c

-- State the theorem
theorem coefficient_b_value (a b d c : ℝ) :
  sum_of_zeros a = product_of_three_zeros d ∧
  sum_of_zeros a = sum_of_coefficients a b d c ∧
  P a b d c 0 = 8 →
  b = -17 := by sorry

end coefficient_b_value_l1675_167528


namespace complement_intersection_equiv_complement_union_l1675_167511

universe u

theorem complement_intersection_equiv_complement_union {U : Type u} (M N : Set U) :
  ∀ x : U, x ∈ (M ∩ N)ᶜ ↔ x ∈ Mᶜ ∪ Nᶜ := by sorry

end complement_intersection_equiv_complement_union_l1675_167511


namespace folded_square_area_ratio_l1675_167566

/-- The ratio of the area of a square paper folded along a line connecting points
    at 1/3 and 2/3 of one side to the area of the original square is 5/6. -/
theorem folded_square_area_ratio (s : ℝ) (h : s > 0) : 
  let A := s^2
  let B := s^2 - (1/2 * (s/3) * s)
  B / A = 5/6 := by sorry

end folded_square_area_ratio_l1675_167566


namespace turtle_ratio_l1675_167519

def total_turtles : ℕ := 42
def turtles_on_sand : ℕ := 28

theorem turtle_ratio : 
  (total_turtles - turtles_on_sand) / total_turtles = 1 / 3 := by
  sorry

end turtle_ratio_l1675_167519


namespace class_size_l1675_167557

theorem class_size (top_rank bottom_rank : ℕ) (h1 : top_rank = 17) (h2 : bottom_rank = 15) :
  top_rank + bottom_rank - 1 = 31 := by
  sorry

end class_size_l1675_167557


namespace unsold_books_l1675_167530

def initial_stock : ℕ := 800
def monday_sales : ℕ := 60
def tuesday_sales : ℕ := 10
def wednesday_sales : ℕ := 20
def thursday_sales : ℕ := 44
def friday_sales : ℕ := 66

theorem unsold_books :
  initial_stock - (monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales) = 600 := by
  sorry

end unsold_books_l1675_167530


namespace a_in_S_l1675_167581

theorem a_in_S (S T : Set ℕ) (a : ℕ) 
  (h1 : S = {1, 2})
  (h2 : T = {a})
  (h3 : S ∪ T = S) :
  a ∈ S := by
  sorry

end a_in_S_l1675_167581


namespace wire_length_from_sphere_l1675_167561

/-- The length of a wire formed by melting a sphere -/
theorem wire_length_from_sphere (r : ℝ) (h : r > 0) : 
  (4 / 3 * π * 12^3) = (π * r^2 * ((4 * 12^3) / (3 * r^2))) := by
  sorry

#check wire_length_from_sphere

end wire_length_from_sphere_l1675_167561


namespace g_has_four_zeros_l1675_167587

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x + 1/x else Real.log x

noncomputable def g (x : ℝ) : ℝ :=
  f (f x + 2) + 2

theorem g_has_four_zeros :
  ∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    g a = 0 ∧ g b = 0 ∧ g c = 0 ∧ g d = 0 ∧
    (∀ x : ℝ, g x = 0 → x = a ∨ x = b ∨ x = c ∨ x = d) :=
sorry

end g_has_four_zeros_l1675_167587


namespace complement_union_theorem_l1675_167541

def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {x : ℕ | ∃ k ∈ A, x = 2 * k}

theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 2, 4, 5, 6} := by
  sorry

end complement_union_theorem_l1675_167541


namespace grade_11_count_l1675_167535

/-- Represents a school with stratified sampling -/
structure School where
  total_students : ℕ
  sample_size : ℕ
  grade_10_sample : ℕ
  grade_12_sample : ℕ

/-- Calculates the number of Grade 11 students in the school -/
def grade_11_students (s : School) : ℕ :=
  ((s.sample_size - s.grade_10_sample - s.grade_12_sample) * s.total_students) / s.sample_size

/-- Theorem stating the number of Grade 11 students in the given school -/
theorem grade_11_count (s : School) 
  (h1 : s.total_students = 900)
  (h2 : s.sample_size = 45)
  (h3 : s.grade_10_sample = 20)
  (h4 : s.grade_12_sample = 10) :
  grade_11_students s = 300 := by
  sorry

#eval grade_11_students ⟨900, 45, 20, 10⟩

end grade_11_count_l1675_167535


namespace remainder_problem_l1675_167514

theorem remainder_problem (d : ℕ) (r : ℕ) (h1 : d > 1) 
  (h2 : 1023 % d = r) (h3 : 1386 % d = r) (h4 : 2151 % d = r) : 
  d - r = 3 := by
  sorry

end remainder_problem_l1675_167514


namespace time_fraction_proof_l1675_167584

/-- Given a 24-hour day and the current time being 6, 
    prove that the fraction of time left to time already completed is 3. -/
theorem time_fraction_proof : 
  let hours_in_day : ℕ := 24
  let current_time : ℕ := 6
  let time_left : ℕ := hours_in_day - current_time
  let time_completed : ℕ := current_time
  (time_left : ℚ) / time_completed = 3 := by
  sorry

end time_fraction_proof_l1675_167584


namespace factorial_not_prime_l1675_167592

theorem factorial_not_prime (n : ℕ) (h : n > 1) : ¬ Nat.Prime (n!) := by
  sorry

end factorial_not_prime_l1675_167592


namespace highest_power_equals_carries_l1675_167543

/-- The number of carries when adding two natural numbers in a given base. -/
def num_carries (m n p : ℕ) : ℕ := sorry

/-- The highest power of p that divides the binomial coefficient (n+m choose m). -/
def highest_power_dividing_binom (n m p : ℕ) : ℕ := sorry

/-- Theorem stating the relationship between the highest power of p dividing
    (n+m choose m) and the number of carries when adding m and n in base p. -/
theorem highest_power_equals_carries (p m n : ℕ) (hp : Nat.Prime p) :
  highest_power_dividing_binom n m p = num_carries m n p :=
sorry

end highest_power_equals_carries_l1675_167543
