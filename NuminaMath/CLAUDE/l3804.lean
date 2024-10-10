import Mathlib

namespace three_numbers_sum_and_ratio_l3804_380428

theorem three_numbers_sum_and_ratio (A B C : ℝ) : 
  A + B + C = 36 →
  (A + B) / (B + C) = 2 / 3 →
  (B + C) / (A + C) = 3 / 4 →
  A = 12 ∧ B = 4 ∧ C = 20 := by
sorry

end three_numbers_sum_and_ratio_l3804_380428


namespace candy_distribution_l3804_380493

theorem candy_distribution (total_candy : ℕ) (num_students : ℕ) (candy_per_student : ℕ) : 
  total_candy = 18 → num_students = 9 → candy_per_student = total_candy / num_students → 
  candy_per_student = 2 := by
  sorry

end candy_distribution_l3804_380493


namespace min_value_on_circle_l3804_380499

theorem min_value_on_circle (x y : ℝ) (h : x^2 + y^2 - 2*x - 2*y + 1 = 0) :
  ∃ (m : ℝ), m = 4/3 ∧ ∀ (x' y' : ℝ), x'^2 + y'^2 - 2*x' - 2*y' + 1 = 0 →
    (y' - 4) / (x' - 2) ≥ m :=
by sorry

end min_value_on_circle_l3804_380499


namespace election_votes_l3804_380404

theorem election_votes (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : (60 * total_votes) / 100 - (40 * total_votes) / 100 = 240) : 
  (60 * total_votes) / 100 = 720 :=
sorry

end election_votes_l3804_380404


namespace smallest_m_value_l3804_380467

theorem smallest_m_value (m : ℕ) : 
  (∃! quad : (ℕ × ℕ × ℕ × ℕ) → Prop, 
    (∃ (n : ℕ), n = 80000 ∧ 
      (∀ a b c d : ℕ, quad (a, b, c, d) → 
        Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 100 ∧
        Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = m))) →
  m = 2250000 :=
sorry

end smallest_m_value_l3804_380467


namespace negation_of_quadratic_inequality_l3804_380402

theorem negation_of_quadratic_inequality (x : ℝ) :
  ¬(x^2 - x + 3 > 0) ↔ x^2 - x + 3 ≤ 0 := by sorry

end negation_of_quadratic_inequality_l3804_380402


namespace inequality_solution_l3804_380443

theorem inequality_solution (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ -1) :
  (x / (x + 1) + (x - 3) / (3 * x) ≥ 4) ↔ 
  (x > -1.5 ∧ x < -1) ∨ (x > -0.25) :=
by sorry

end inequality_solution_l3804_380443


namespace sequence_term_40_l3804_380431

theorem sequence_term_40 (n : ℕ+) (a : ℕ+ → ℕ) : 
  (∀ k : ℕ+, a k = 3 * k + 1) → a 13 = 40 := by
  sorry

end sequence_term_40_l3804_380431


namespace point_inside_circle_l3804_380484

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    eccentricity e = √2, right focus F(c, 0), and an equation ax² - bx - c = 0
    with roots x₁ and x₂, prove that the point P(x₁, x₂) is inside the circle x² + y² = 8 -/
theorem point_inside_circle (a b c : ℝ) (x₁ x₂ : ℝ) 
  (ha : a > 0) (hb : b > 0)
  (h_hyperbola : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (h_eccentricity : c / a = Real.sqrt 2)
  (h_focus : c > 0)
  (h_roots : a * x₁^2 - b * x₁ - c = 0 ∧ a * x₂^2 - b * x₂ - c = 0) :
  x₁^2 + x₂^2 < 8 := by
  sorry

end point_inside_circle_l3804_380484


namespace right_triangle_with_reversed_digits_l3804_380434

theorem right_triangle_with_reversed_digits : ∀ a b c : ℕ,
  a = 56 ∧ c = 65 ∧ 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ 10 ≤ c ∧ c < 100 →
  a^2 + b^2 = c^2 →
  b = 33 :=
by
  sorry

end right_triangle_with_reversed_digits_l3804_380434


namespace promotion_theorem_l3804_380464

/-- Calculates the maximum amount of goods that can be purchased given a promotion and initial spending. -/
def maxPurchaseAmount (promotionRate : Rat) (rewardRate : Rat) (initialSpend : ℕ) : ℕ :=
  sorry

/-- The promotion theorem -/
theorem promotion_theorem :
  let promotionRate : Rat := 100
  let rewardRate : Rat := 20
  let initialSpend : ℕ := 7020
  maxPurchaseAmount promotionRate rewardRate initialSpend = 8760 := by
  sorry

end promotion_theorem_l3804_380464


namespace circle_graph_proportion_l3804_380406

theorem circle_graph_proportion (angle : ℝ) (percentage : ℝ) :
  angle = 180 →
  angle / 360 = percentage / 100 →
  percentage = 50 := by
sorry

end circle_graph_proportion_l3804_380406


namespace max_point_of_f_l3804_380435

def f (x : ℝ) := 3 * x - x^3

theorem max_point_of_f :
  ∃ (x₀ : ℝ), ∀ (x : ℝ), f x ≤ f x₀ ∧ x₀ = 1 := by
  sorry

end max_point_of_f_l3804_380435


namespace one_story_height_l3804_380472

-- Define the parameters
def stories : ℕ := 6
def rope_length : ℝ := 20
def loss_percentage : ℝ := 0.25
def num_ropes : ℕ := 4

-- Define the theorem
theorem one_story_height :
  let total_usable_length := (1 - loss_percentage) * rope_length * num_ropes
  let story_height := total_usable_length / stories
  story_height = 10 := by sorry

end one_story_height_l3804_380472


namespace sum_of_c_values_l3804_380463

theorem sum_of_c_values : ∃ (S : Finset ℤ),
  (∀ c ∈ S, c ≤ 30 ∧ 
    ∃ (x y : ℚ), x^2 - 9*x - c = 0 ∧ y^2 - 9*y - c = 0 ∧ x ≠ y) ∧
  (∀ c : ℤ, c ≤ 30 → 
    (∃ (x y : ℚ), x^2 - 9*x - c = 0 ∧ y^2 - 9*y - c = 0 ∧ x ≠ y) → 
    c ∈ S) ∧
  (S.sum id = 32) := by
sorry

end sum_of_c_values_l3804_380463


namespace total_hotdogs_sold_l3804_380476

/-- Represents the number of hotdogs sold in each size category -/
structure HotdogSales where
  small : Float
  medium : Float
  large : Float
  extra_large : Float

/-- Calculates the total number of hotdogs sold -/
def total_hotdogs (sales : HotdogSales) : Float :=
  sales.small + sales.medium + sales.large + sales.extra_large

/-- Theorem: The total number of hotdogs sold is 131.3 -/
theorem total_hotdogs_sold (sales : HotdogSales)
  (h1 : sales.small = 58.3)
  (h2 : sales.medium = 21.7)
  (h3 : sales.large = 35.9)
  (h4 : sales.extra_large = 15.4) :
  total_hotdogs sales = 131.3 := by
  sorry

#eval total_hotdogs { small := 58.3, medium := 21.7, large := 35.9, extra_large := 15.4 }

end total_hotdogs_sold_l3804_380476


namespace specific_rhombus_area_l3804_380432

/-- Represents a rhombus with given properties -/
structure Rhombus where
  side_length : ℝ
  diagonal_difference : ℝ
  diagonals_perpendicular_bisectors : Bool

/-- Calculates the area of a rhombus given its properties -/
def rhombus_area (r : Rhombus) : ℝ :=
  sorry

/-- Theorem stating the area of a specific rhombus -/
theorem specific_rhombus_area :
  let r : Rhombus := {
    side_length := Real.sqrt 113,
    diagonal_difference := 8,
    diagonals_perpendicular_bisectors := true
  }
  rhombus_area r = 97 := by sorry

end specific_rhombus_area_l3804_380432


namespace min_troupe_size_l3804_380419

def is_valid_troupe_size (n : ℕ) : Prop :=
  n % 4 = 0 ∧ n % 9 = 0 ∧ n % 10 = 0 ∧ n > 50

theorem min_troupe_size :
  ∃ (n : ℕ), is_valid_troupe_size n ∧ ∀ (m : ℕ), is_valid_troupe_size m → n ≤ m :=
by
  sorry

end min_troupe_size_l3804_380419


namespace geometric_progression_ratio_l3804_380478

theorem geometric_progression_ratio (a b c : ℝ) (x : ℝ) (r : ℝ) : 
  a = 30 → b = 80 → c = 160 →
  (b + x)^2 = (a + x) * (c + x) →
  x = 160 / 3 →
  r = (b + x) / (a + x) →
  r = (c + x) / (b + x) →
  r = 8 / 5 := by
  sorry

end geometric_progression_ratio_l3804_380478


namespace rahul_salary_l3804_380409

def salary_calculation (salary : ℝ) : ℝ :=
  let after_rent := salary * 0.8
  let after_education := after_rent * 0.9
  let after_clothes := after_education * 0.9
  after_clothes

theorem rahul_salary : ∃ (salary : ℝ), salary_calculation salary = 1377 ∧ salary = 2125 := by
  sorry

end rahul_salary_l3804_380409


namespace final_balance_l3804_380496

def account_balance (initial : ℕ) (coffee_beans : ℕ) (tumbler : ℕ) (coffee_filter : ℕ) (refund : ℕ) : ℕ :=
  initial - (coffee_beans + tumbler + coffee_filter) + refund

theorem final_balance :
  account_balance 50 10 30 5 20 = 25 := by
  sorry

end final_balance_l3804_380496


namespace max_reflections_l3804_380461

theorem max_reflections (angle : ℝ) (h : angle = 8) : 
  ∃ (n : ℕ), n ≤ 10 ∧ n * angle < 90 ∧ ∀ m : ℕ, m > n → m * angle ≥ 90 :=
by sorry

end max_reflections_l3804_380461


namespace prime_odd_sum_l3804_380407

theorem prime_odd_sum (x y : ℕ) 
  (hx : Nat.Prime x) 
  (hy : Odd y) 
  (heq : x^2 + y = 2005) : 
  x + y = 2003 := by
sorry

end prime_odd_sum_l3804_380407


namespace discount_difference_l3804_380415

def initial_amount : ℝ := 15000

def apply_discount (amount : ℝ) (discount : ℝ) : ℝ :=
  amount * (1 - discount)

def option1_price : ℝ :=
  apply_discount (apply_discount (apply_discount initial_amount 0.25) 0.1) 0.05

def option2_price : ℝ :=
  apply_discount (apply_discount (apply_discount initial_amount 0.3) 0.1) 0.1

theorem discount_difference :
  option1_price - option2_price = 1113.75 := by sorry

end discount_difference_l3804_380415


namespace no_trapezoid_solution_l3804_380403

theorem no_trapezoid_solution : ¬ ∃ (b₁ b₂ : ℕ), 
  (b₁ + b₂) * 40 / 2 = 1800 ∧ 
  ∃ (k : ℕ), b₁ = 2 * k + 1 ∧ 
  ∃ (m : ℕ), b₁ = 5 * m ∧
  ∃ (n : ℕ), b₂ = 2 * n :=
by sorry

end no_trapezoid_solution_l3804_380403


namespace limits_of_f_l3804_380475

noncomputable def f (x : ℝ) : ℝ := 2^(1/x)

theorem limits_of_f :
  (∀ ε > 0, ∃ δ > 0, ∀ x < 0, |x| < δ → |f x| < ε) ∧
  (∀ M > 0, ∃ δ > 0, ∀ x > 0, x < δ → f x > M) ∧
  ¬ (∃ L : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ → |f x - L| < ε) :=
by sorry

end limits_of_f_l3804_380475


namespace adjacent_knights_probability_l3804_380468

/-- The number of knights seated at the round table -/
def num_knights : ℕ := 25

/-- The number of knights chosen -/
def chosen_knights : ℕ := 3

/-- The probability of choosing at least two adjacent knights -/
def P : ℚ := 21/92

/-- Theorem stating the probability of choosing at least two adjacent knights -/
theorem adjacent_knights_probability :
  (
    let total_choices := Nat.choose num_knights chosen_knights
    let adjacent_triples := num_knights
    let adjacent_pairs := num_knights * (num_knights - 2 * chosen_knights + 1)
    let favorable_outcomes := adjacent_triples + adjacent_pairs
    (favorable_outcomes : ℚ) / total_choices
  ) = P := by sorry

end adjacent_knights_probability_l3804_380468


namespace total_wall_area_to_paint_l3804_380447

def living_room_width : ℝ := 40
def living_room_length : ℝ := 40
def bedroom_width : ℝ := 10
def bedroom_length : ℝ := 12
def wall_height : ℝ := 10
def living_room_walls_to_paint : ℕ := 3
def bedroom_walls_to_paint : ℕ := 4

theorem total_wall_area_to_paint :
  (living_room_walls_to_paint * living_room_width * wall_height) +
  (bedroom_walls_to_paint * bedroom_width * wall_height) +
  (bedroom_walls_to_paint * bedroom_length * wall_height) -
  (2 * bedroom_width * wall_height) = 1640 := by sorry

end total_wall_area_to_paint_l3804_380447


namespace trig_sum_equals_two_l3804_380413

theorem trig_sum_equals_two : Real.cos (π / 4) ^ 2 + Real.tan (π / 3) * Real.cos (π / 6) = 2 := by
  sorry

end trig_sum_equals_two_l3804_380413


namespace inner_tangent_circle_radius_l3804_380412

/-- Given a right triangle with legs 3 and 4 units, the radius of the circle
    tangent to both legs and the circumcircle internally is 2 units. -/
theorem inner_tangent_circle_radius (a b c r : ℝ) : 
  a = 3 → b = 4 → c^2 = a^2 + b^2 → r = a + b - c → r = 2 := by
  sorry

end inner_tangent_circle_radius_l3804_380412


namespace smallest_positive_integer_2002m_44444n_l3804_380400

theorem smallest_positive_integer_2002m_44444n : 
  (∃ (k : ℕ+), ∀ (a : ℕ+), (∃ (m n : ℤ), a.val = 2002 * m + 44444 * n) → k ≤ a) ∧ 
  (∃ (m n : ℤ), (2 : ℕ+).val = 2002 * m + 44444 * n) :=
sorry

end smallest_positive_integer_2002m_44444n_l3804_380400


namespace max_profit_is_45_6_l3804_380417

/-- Profit function for location A -/
def L₁ (x : ℝ) : ℝ := 5.06 * x - 0.15 * x^2

/-- Profit function for location B -/
def L₂ (x : ℝ) : ℝ := 2 * x

/-- Total profit function -/
def S (x : ℝ) : ℝ := L₁ x + L₂ (15 - x)

/-- The maximum total profit is 45.6 when selling 15 cars across both locations -/
theorem max_profit_is_45_6 :
  ∃ x : ℕ, x ≤ 15 ∧ S x = 45.6 ∧ ∀ y : ℕ, y ≤ 15 → S y ≤ S x := by
  sorry

end max_profit_is_45_6_l3804_380417


namespace boulder_splash_width_l3804_380456

/-- The width of a boulder's splash given the number of pebbles, rocks, and boulders thrown,
    and the total width of all splashes. -/
theorem boulder_splash_width
  (num_pebbles : ℕ)
  (num_rocks : ℕ)
  (num_boulders : ℕ)
  (total_width : ℝ)
  (pebble_splash : ℝ)
  (rock_splash : ℝ)
  (h1 : num_pebbles = 6)
  (h2 : num_rocks = 3)
  (h3 : num_boulders = 2)
  (h4 : total_width = 7)
  (h5 : pebble_splash = 1/4)
  (h6 : rock_splash = 1/2)
  : (total_width - (num_pebbles * pebble_splash + num_rocks * rock_splash)) / num_boulders = 2 :=
sorry

end boulder_splash_width_l3804_380456


namespace circle_rational_points_infinite_l3804_380453

theorem circle_rational_points_infinite :
  ∃ (S : Set (ℚ × ℚ)), Set.Infinite S ∧ ∀ (p : ℚ × ℚ), p ∈ S → (p.1^2 + p.2^2 = 1) :=
by sorry

end circle_rational_points_infinite_l3804_380453


namespace similar_triangles_side_length_l3804_380470

/-- Given two similar right triangles, where the first triangle has a side of 18 units
    and a hypotenuse of 30 units, and the second triangle has a hypotenuse of 60 units,
    the side in the second triangle corresponding to the 18-unit side in the first triangle
    is 36 units long. -/
theorem similar_triangles_side_length (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  a^2 + 18^2 = 30^2 →
  c^2 + d^2 = 60^2 →
  30 / 60 = 18 / d →
  d = 36 := by
sorry

end similar_triangles_side_length_l3804_380470


namespace binomial_15_4_l3804_380446

theorem binomial_15_4 : Nat.choose 15 4 = 1365 := by
  sorry

end binomial_15_4_l3804_380446


namespace odd_numbers_property_l3804_380465

theorem odd_numbers_property (a b c d k m : ℕ) : 
  Odd a → Odd b → Odd c → Odd d →
  0 < a → a < b → b < c → c < d →
  a * d = b * c →
  a + d = 2^k →
  b + c = 2^m →
  a = 1 := by sorry

end odd_numbers_property_l3804_380465


namespace elephants_viewing_time_l3804_380469

def zoo_visit (total_time seals_time penguins_multiplier : ℕ) : ℕ :=
  total_time - (seals_time + seals_time * penguins_multiplier)

theorem elephants_viewing_time :
  zoo_visit 130 13 8 = 13 := by
  sorry

end elephants_viewing_time_l3804_380469


namespace prob_sum_greater_than_ten_is_three_sixteenths_l3804_380466

/-- A fair 6-sided die -/
def six_sided_die : Finset ℕ := Finset.range 6

/-- A fair 8-sided die -/
def eight_sided_die : Finset ℕ := Finset.range 8

/-- The product space of rolling both dice -/
def dice_product : Finset (ℕ × ℕ) := six_sided_die.product eight_sided_die

/-- The subset of outcomes where the sum is greater than 10 -/
def sum_greater_than_ten : Finset (ℕ × ℕ) :=
  dice_product.filter (fun p => p.1 + p.2 + 2 > 10)

/-- The probability of the sum being greater than 10 -/
def prob_sum_greater_than_ten : ℚ :=
  (sum_greater_than_ten.card : ℚ) / (dice_product.card : ℚ)

theorem prob_sum_greater_than_ten_is_three_sixteenths :
  prob_sum_greater_than_ten = 3 / 16 := by
  sorry

end prob_sum_greater_than_ten_is_three_sixteenths_l3804_380466


namespace sqrt_20_less_than_5_l3804_380481

theorem sqrt_20_less_than_5 : Real.sqrt 20 < 5 := by
  sorry

end sqrt_20_less_than_5_l3804_380481


namespace cubic_equation_properties_l3804_380449

theorem cubic_equation_properties :
  (∀ x y : ℕ, x^3 + y = y^3 + x → x = y) ∧
  (∃ x y : ℚ, x > 0 ∧ y > 0 ∧ x ≠ y ∧ x^3 + y = y^3 + x) :=
by sorry

end cubic_equation_properties_l3804_380449


namespace rectangular_field_dimensions_l3804_380497

theorem rectangular_field_dimensions (m : ℝ) : 
  (3 * m + 8) * (m - 3) = 120 → m = 7 := by
sorry

end rectangular_field_dimensions_l3804_380497


namespace tax_rate_is_ten_percent_l3804_380487

/-- Calculates the tax rate given the total amount spent, sales tax, and cost of tax-free items -/
def calculate_tax_rate (total_amount : ℚ) (sales_tax : ℚ) (tax_free_cost : ℚ) : ℚ :=
  let taxable_cost := total_amount - tax_free_cost - sales_tax
  (sales_tax / taxable_cost) * 100

/-- Theorem stating that the tax rate is 10% given the problem conditions -/
theorem tax_rate_is_ten_percent 
  (total_amount : ℚ) 
  (sales_tax : ℚ) 
  (tax_free_cost : ℚ)
  (h1 : total_amount = 25)
  (h2 : sales_tax = 3/10)
  (h3 : tax_free_cost = 217/10) :
  calculate_tax_rate total_amount sales_tax tax_free_cost = 10 := by
  sorry

#eval calculate_tax_rate 25 (3/10) (217/10)

end tax_rate_is_ten_percent_l3804_380487


namespace constant_term_expansion_l3804_380495

/-- The constant term in the expansion of (x + 1/x + 1)^4 -/
def constant_term : ℕ := 19

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

theorem constant_term_expansion :
  constant_term = 1 + binomial 4 2 * binomial 2 1 + binomial 4 4 * binomial 4 2 :=
sorry

end constant_term_expansion_l3804_380495


namespace parabola_point_distance_l3804_380489

/-- Parabola type representing y = -ax²/6 + ax + c -/
structure Parabola where
  a : ℝ
  c : ℝ
  h_a : a < 0

/-- Point on a parabola -/
structure ParabolaPoint (p : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y = -p.a * x^2 / 6 + p.a * x + p.c

/-- Theorem statement -/
theorem parabola_point_distance (p : Parabola) 
  (A B C : ParabolaPoint p) 
  (h_B_vertex : B.y = 3 * p.a / 2 + p.c) 
  (h_y_order : A.y > C.y ∧ C.y > B.y) :
  |A.x - B.x| ≥ |B.x - C.x| := by sorry

end parabola_point_distance_l3804_380489


namespace floor_ceiling_sum_seven_l3804_380414

theorem floor_ceiling_sum_seven (x : ℝ) : 
  (⌊x⌋ : ℤ) + (⌈x⌉ : ℤ) = 7 ↔ 3 < x ∧ x < 4 := by sorry

end floor_ceiling_sum_seven_l3804_380414


namespace total_pages_read_l3804_380444

-- Define the reading rates (pages per 60 minutes)
def rene_rate : ℕ := 30
def lulu_rate : ℕ := 27
def cherry_rate : ℕ := 25

-- Define the total reading time in minutes
def total_time : ℕ := 240

-- Define the function to calculate pages read given rate and time
def pages_read (rate : ℕ) (time : ℕ) : ℕ :=
  rate * (time / 60)

-- Theorem statement
theorem total_pages_read :
  pages_read rene_rate total_time +
  pages_read lulu_rate total_time +
  pages_read cherry_rate total_time = 328 := by
  sorry


end total_pages_read_l3804_380444


namespace markus_final_candies_l3804_380477

theorem markus_final_candies 
  (markus_initial : ℕ) 
  (katharina_initial : ℕ) 
  (sanjiv_distribution : ℕ) 
  (h1 : markus_initial = 9)
  (h2 : katharina_initial = 5)
  (h3 : sanjiv_distribution = 10)
  (h4 : ∃ (x : ℕ), x + markus_initial + x + katharina_initial = markus_initial + katharina_initial + sanjiv_distribution) :
  ∃ (markus_final : ℕ), markus_final = 12 ∧ 2 * markus_final = markus_initial + katharina_initial + sanjiv_distribution :=
sorry

end markus_final_candies_l3804_380477


namespace solution_composition_l3804_380492

theorem solution_composition (x : ℝ) : 
  -- First solution composition
  let solution1_A := 0.20
  let solution1_B := 0.80
  -- Second solution composition
  let solution2_A := x
  let solution2_B := 0.70
  -- Mixture composition
  let mixture_solution1 := 0.80
  let mixture_solution2 := 0.20
  -- Final mixture composition of material A
  let final_mixture_A := 0.22
  -- Equation for material A in the final mixture
  solution1_A * mixture_solution1 + solution2_A * mixture_solution2 = final_mixture_A
  →
  x = 0.30 := by
sorry

end solution_composition_l3804_380492


namespace hard_drive_cost_l3804_380454

/-- The cost of seven hard drives with a bulk discount -/
theorem hard_drive_cost : 
  -- Two hard drives cost $50
  (∃ (single_cost : ℝ), 2 * single_cost = 50) →
  -- There's a 10% discount for buying more than 4
  (∀ (n : ℕ), n > 4 → ∃ (discount_factor : ℝ), discount_factor = 0.9) →
  -- The cost of 7 hard drives with the discount is $157.5
  ∃ (total_cost : ℝ), total_cost = 157.5 := by
  sorry

end hard_drive_cost_l3804_380454


namespace cone_volume_with_special_surface_area_l3804_380474

/-- 
Given a cone with base radius R, if its lateral surface area is equal to the sum of 
the areas of its base and axial section, then its volume is (2π²R³) / (3(π² - 1)).
-/
theorem cone_volume_with_special_surface_area (R : ℝ) (h : R > 0) : 
  let lateral_surface_area := π * R * (R^2 + (2 * π * R / (π^2 - 1))^2).sqrt
  let base_area := π * R^2
  let axial_section_area := R * (2 * π * R / (π^2 - 1))
  lateral_surface_area = base_area + axial_section_area →
  (1/3) * π * R^2 * (2 * π * R / (π^2 - 1)) = 2 * π^2 * R^3 / (3 * (π^2 - 1)) := by
sorry

end cone_volume_with_special_surface_area_l3804_380474


namespace mean_temperature_is_negative_point_six_l3804_380422

def temperatures : List ℝ := [-8, -5, -5, -2, 0, 4, 5, 3, 6, 1]

theorem mean_temperature_is_negative_point_six :
  (temperatures.sum / temperatures.length : ℝ) = -0.6 := by
  sorry

end mean_temperature_is_negative_point_six_l3804_380422


namespace quadratic_trinomial_with_integral_roots_l3804_380423

theorem quadratic_trinomial_with_integral_roots :
  ∃ (a b c : ℕ+),
    (∃ (x : ℤ), a * x^2 + b * x + c = 0) ∧
    (∃ (y : ℤ), (a + 1) * y^2 + (b + 1) * y + (c + 1) = 0) ∧
    (∃ (z : ℤ), (a + 2) * z^2 + (b + 2) * z + (c + 2) = 0) :=
by sorry

end quadratic_trinomial_with_integral_roots_l3804_380423


namespace area_outside_squares_inside_triangle_l3804_380438

/-- Represents a square with a given side length -/
structure Square where
  sideLength : ℝ

/-- Represents the problem setup -/
structure SquareProblem where
  bigSquare : Square
  smallSquare1 : Square
  smallSquare2 : Square

/-- The main theorem stating the area of the region -/
theorem area_outside_squares_inside_triangle (p : SquareProblem) : 
  p.bigSquare.sideLength = 6 ∧ 
  p.smallSquare1.sideLength = 2 ∧ 
  p.smallSquare2.sideLength = 3 →
  let triangleArea := (p.bigSquare.sideLength ^ 2) / 2
  let smallSquaresArea := p.smallSquare1.sideLength ^ 2 + p.smallSquare2.sideLength ^ 2
  triangleArea - smallSquaresArea = 5 := by
  sorry

end area_outside_squares_inside_triangle_l3804_380438


namespace room_tiles_count_l3804_380436

/-- Calculates the number of tiles needed for a room with given specifications -/
def calculate_tiles (room_length room_width border_width tile_size column_size : ℕ) : ℕ :=
  let inner_length := room_length - 2 * border_width
  let inner_width := room_width - 2 * border_width
  let border_tiles := 2 * ((room_length / tile_size) + (room_width / tile_size) - 4)
  let inner_tiles := (inner_length * inner_width) / (tile_size * tile_size)
  let column_tiles := (column_size * column_size + tile_size * tile_size - 1) / (tile_size * tile_size)
  border_tiles + inner_tiles + column_tiles

/-- Theorem stating that the number of tiles for the given room specification is 78 -/
theorem room_tiles_count : calculate_tiles 15 20 2 2 1 = 78 := by
  sorry

end room_tiles_count_l3804_380436


namespace linear_function_inequality_l3804_380471

theorem linear_function_inequality (a b : ℝ) (h1 : a > 0) (h2 : -2 * a + b = 0) :
  ∀ x : ℝ, a * x > b ↔ x > 2 :=
by sorry

end linear_function_inequality_l3804_380471


namespace smallest_terminating_n_is_correct_l3804_380491

/-- A fraction a/b is a terminating decimal if b has only 2 and 5 as prime factors -/
def IsTerminatingDecimal (a b : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → p ∣ b → p = 2 ∨ p = 5

/-- The smallest positive integer n such that n/(n+150) is a terminating decimal -/
def SmallestTerminatingN : ℕ := 10

theorem smallest_terminating_n_is_correct :
  IsTerminatingDecimal SmallestTerminatingN (SmallestTerminatingN + 150) ∧
  ∀ m : ℕ, 0 < m → m < SmallestTerminatingN →
    ¬IsTerminatingDecimal m (m + 150) := by
  sorry

end smallest_terminating_n_is_correct_l3804_380491


namespace rubber_duck_race_l3804_380418

theorem rubber_duck_race (regular_price : ℚ) (large_price : ℚ) (regular_sold : ℕ) (total_raised : ℚ) :
  regular_price = 3 →
  large_price = 5 →
  regular_sold = 221 →
  total_raised = 1588 →
  ∃ (large_sold : ℕ), large_sold = 185 ∧ 
    regular_price * regular_sold + large_price * large_sold = total_raised :=
by sorry

end rubber_duck_race_l3804_380418


namespace multiplication_addition_equality_l3804_380452

theorem multiplication_addition_equality : 15 * 36 + 15 * 24 = 900 := by
  sorry

end multiplication_addition_equality_l3804_380452


namespace triangle_area_l3804_380441

/-- Given a triangle with perimeter 32 and inradius 2.5, prove its area is 40 -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) 
  (h1 : perimeter = 32) 
  (h2 : inradius = 2.5) 
  (h3 : area = inradius * (perimeter / 2)) : 
  area = 40 := by
  sorry

end triangle_area_l3804_380441


namespace smallest_odd_minimizer_l3804_380450

/-- The number of positive integer divisors of n, including 1 and n -/
def d (n : ℕ) : ℕ := (Nat.divisors n).card

/-- The function g(n) = d(n) / n^(1/4) -/
noncomputable def g (n : ℕ) : ℝ := (d n : ℝ) / n^(1/4 : ℝ)

/-- n is an odd positive integer -/
def isOddPositive (n : ℕ) : Prop := n > 0 ∧ n % 2 = 1

/-- 9 is the smallest odd positive integer N such that g(N) < g(n) for all odd positive integers n ≠ N -/
theorem smallest_odd_minimizer :
  isOddPositive 9 ∧
  (∀ n : ℕ, isOddPositive n → n ≠ 9 → g 9 < g n) ∧
  (∀ N : ℕ, isOddPositive N → N < 9 → ∃ n : ℕ, isOddPositive n ∧ n ≠ N ∧ g N ≥ g n) :=
sorry

end smallest_odd_minimizer_l3804_380450


namespace lasagna_profit_proof_l3804_380488

/-- Calculates the profit after expenses for selling lasagna pans -/
def profit_after_expenses (num_pans : ℕ) (cost_per_pan : ℚ) (price_per_pan : ℚ) : ℚ :=
  num_pans * (price_per_pan - cost_per_pan)

/-- Proves that the profit after expenses for selling 20 pans of lasagna is $300.00 -/
theorem lasagna_profit_proof :
  profit_after_expenses 20 10 25 = 300 := by
  sorry

end lasagna_profit_proof_l3804_380488


namespace not_parabola_l3804_380451

theorem not_parabola (α x y : ℝ) : 
  ∃ (a b c : ℝ), ∀ (x y : ℝ), x^2 * Real.sin α + y^2 * Real.cos α = 1 → y ≠ a*x^2 + b*x + c :=
sorry

end not_parabola_l3804_380451


namespace fraction_evaluation_l3804_380430

theorem fraction_evaluation : (1 - 1/4) / (1 - 1/5) = 15/16 := by
  sorry

end fraction_evaluation_l3804_380430


namespace inequality_implication_l3804_380498

theorem inequality_implication (x y : ℝ) (h : x < y) : -x + 3 > -y + 3 := by
  sorry

end inequality_implication_l3804_380498


namespace course_selection_plans_l3804_380458

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of course selection plans -/
def coursePlans (totalCourses restrictedCourses coursesToChoose : ℕ) : ℕ :=
  choose (totalCourses - restrictedCourses) coursesToChoose + 
  restrictedCourses * choose (totalCourses - restrictedCourses) (coursesToChoose - 1)

theorem course_selection_plans :
  coursePlans 8 2 5 = 36 := by sorry

end course_selection_plans_l3804_380458


namespace heights_academy_music_problem_l3804_380411

/-- The Heights Academy music problem -/
theorem heights_academy_music_problem
  (total_students : ℕ)
  (females_band : ℕ)
  (males_band : ℕ)
  (females_orchestra : ℕ)
  (males_orchestra : ℕ)
  (females_both : ℕ)
  (h1 : total_students = 260)
  (h2 : females_band = 120)
  (h3 : males_band = 90)
  (h4 : females_orchestra = 100)
  (h5 : males_orchestra = 130)
  (h6 : females_both = 80) :
  males_band - (males_band + males_orchestra - (total_students - (females_band + females_orchestra - females_both))) = 30 := by
  sorry


end heights_academy_music_problem_l3804_380411


namespace quadratic_shared_root_property_l3804_380494

/-- A quadratic polynomial P(x) = x^2 + bx + c -/
def P (b c x : ℝ) : ℝ := x^2 + b*x + c

/-- The theorem stating that if P(x) and P(P(P(x))) share a root, then P(0) * P(1) = 0 -/
theorem quadratic_shared_root_property (b c : ℝ) :
  (∃ r : ℝ, P b c r = 0 ∧ P b c (P b c (P b c r)) = 0) →
  P b c 0 * P b c 1 = 0 := by
sorry

end quadratic_shared_root_property_l3804_380494


namespace sum_of_reciprocal_G_powers_of_two_l3804_380408

def G : ℕ → ℚ
  | 0 => 0
  | 1 => 5/2
  | (n + 2) => 7/2 * G (n + 1) - G n

theorem sum_of_reciprocal_G_powers_of_two : ∑' n, 1 / G (2^n) = 1 := by sorry

end sum_of_reciprocal_G_powers_of_two_l3804_380408


namespace triangle_side_length_l3804_380459

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  b = Real.sqrt 7 →
  B = π / 3 →
  b^2 = a^2 + c^2 - 2*a*c*Real.cos B →
  c = 3 := by
  sorry

end triangle_side_length_l3804_380459


namespace nonagon_diagonals_l3804_380490

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular nonagon has 27 diagonals -/
theorem nonagon_diagonals : num_diagonals 9 = 27 := by
  sorry

end nonagon_diagonals_l3804_380490


namespace weight_of_replaced_person_l3804_380439

/-- Given 5 persons, if replacing one person with a new person weighing 95.5 kg
    increases the average weight by 5.5 kg, then the weight of the replaced person was 68 kg. -/
theorem weight_of_replaced_person (initial_count : ℕ) (new_person_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 5 →
  new_person_weight = 95.5 →
  avg_increase = 5.5 →
  (new_person_weight - initial_count * avg_increase : ℝ) = 68 := by
sorry

end weight_of_replaced_person_l3804_380439


namespace probability_roots_different_signs_l3804_380437

def S : Set ℕ := {1, 2, 3, 4, 5, 6}

def quadratic_equation (a b x : ℝ) : Prop :=
  x^2 - 2*(a-3)*x + 9 - b^2 = 0

def roots_different_signs (a b : ℝ) : Prop :=
  (9 - b^2 < 0) ∧ (4*(a-3)^2 - 4*(9-b^2) > 0)

def count_valid_pairs : ℕ := 18

def total_pairs : ℕ := 36

theorem probability_roots_different_signs :
  (count_valid_pairs : ℚ) / (total_pairs : ℚ) = 1/2 :=
sorry

end probability_roots_different_signs_l3804_380437


namespace window_area_l3804_380424

-- Define the number of panes
def num_panes : ℕ := 8

-- Define the length of each pane in inches
def pane_length : ℕ := 12

-- Define the width of each pane in inches
def pane_width : ℕ := 8

-- Theorem statement
theorem window_area :
  (num_panes * pane_length * pane_width) = 768 := by
  sorry

end window_area_l3804_380424


namespace f_2_equals_100_l3804_380429

-- Define the function f
def f (x y : ℝ) : ℝ := 2 * x^2 + y

-- State the theorem
theorem f_2_equals_100 :
  ∃ y : ℝ, f 5 y = 142 ∧ f 2 y = 100 :=
by sorry

end f_2_equals_100_l3804_380429


namespace solution_satisfies_system_solution_is_unique_l3804_380426

/-- Represents a system of two linear equations with two unknowns,
    where the coefficients form an arithmetic progression. -/
structure ArithmeticProgressionSystem where
  a : ℝ
  d : ℝ

/-- The solution to the system of linear equations. -/
def solution : ℝ × ℝ := (-1, 2)

/-- Checks if the given pair (x, y) satisfies the first equation of the system. -/
def satisfies_equation1 (sys : ArithmeticProgressionSystem) (sol : ℝ × ℝ) : Prop :=
  sys.a * sol.1 + (sys.a + sys.d) * sol.2 = sys.a + 2 * sys.d

/-- Checks if the given pair (x, y) satisfies the second equation of the system. -/
def satisfies_equation2 (sys : ArithmeticProgressionSystem) (sol : ℝ × ℝ) : Prop :=
  (sys.a + 3 * sys.d) * sol.1 + (sys.a + 4 * sys.d) * sol.2 = sys.a + 5 * sys.d

/-- Theorem stating that the solution satisfies both equations of the system. -/
theorem solution_satisfies_system (sys : ArithmeticProgressionSystem) :
  satisfies_equation1 sys solution ∧ satisfies_equation2 sys solution :=
sorry

/-- Theorem stating that the solution is unique. -/
theorem solution_is_unique (sys : ArithmeticProgressionSystem) (sol : ℝ × ℝ) :
  satisfies_equation1 sys sol ∧ satisfies_equation2 sys sol → sol = solution :=
sorry

end solution_satisfies_system_solution_is_unique_l3804_380426


namespace parabola_y_axis_intersection_l3804_380486

/-- A parabola passing through the points (-1, -6) and (1, 0) -/
def Parabola (x y : ℝ) : Prop :=
  ∃ (m n : ℝ), y = x^2 + m*x + n ∧ -6 = 1 - m + n ∧ 0 = 1 + m + n

/-- The intersection point of the parabola with the y-axis -/
def YAxisIntersection (x y : ℝ) : Prop :=
  Parabola x y ∧ x = 0

theorem parabola_y_axis_intersection :
  ∀ x y, YAxisIntersection x y → x = 0 ∧ y = -4 := by sorry

end parabola_y_axis_intersection_l3804_380486


namespace intersection_in_second_quadrant_l3804_380410

/-- 
If the intersection point of the lines y = 2x + 4 and y = -2x + m 
is in the second quadrant, then -4 < m < 4.
-/
theorem intersection_in_second_quadrant (m : ℝ) : 
  (∃ x y : ℝ, y = 2*x + 4 ∧ y = -2*x + m ∧ x < 0 ∧ y > 0) → 
  -4 < m ∧ m < 4 :=
by sorry

end intersection_in_second_quadrant_l3804_380410


namespace unique_quadratic_solution_l3804_380480

theorem unique_quadratic_solution (b : ℝ) (h1 : b ≠ 0) 
  (h2 : ∃! x, b * x^2 + 30 * x + 10 = 0) : 
  ∃ x, b * x^2 + 30 * x + 10 = 0 ∧ x = -2/3 := by
  sorry

end unique_quadratic_solution_l3804_380480


namespace largest_of_three_consecutive_even_numbers_l3804_380473

theorem largest_of_three_consecutive_even_numbers (x : ℤ) : 
  (∃ (a b c : ℤ), 
    (a + b + c = 312) ∧ 
    (b = a + 2) ∧ 
    (c = b + 2) ∧ 
    (Even a) ∧ (Even b) ∧ (Even c)) →
  (max a (max b c) = 106) :=
sorry

end largest_of_three_consecutive_even_numbers_l3804_380473


namespace pepper_plants_died_l3804_380483

/-- Represents the garden with its plants and vegetables --/
structure Garden where
  tomato_plants : ℕ
  eggplant_plants : ℕ
  pepper_plants : ℕ
  dead_tomato_plants : ℕ
  dead_pepper_plants : ℕ
  vegetables_per_plant : ℕ
  total_vegetables : ℕ

/-- Theorem representing the problem and its solution --/
theorem pepper_plants_died (g : Garden) : g.dead_pepper_plants = 1 :=
  by
  have h1 : g.tomato_plants = 6 := by sorry
  have h2 : g.eggplant_plants = 2 := by sorry
  have h3 : g.pepper_plants = 4 := by sorry
  have h4 : g.dead_tomato_plants = g.tomato_plants / 2 := by sorry
  have h5 : g.vegetables_per_plant = 7 := by sorry
  have h6 : g.total_vegetables = 56 := by sorry
  
  sorry

end pepper_plants_died_l3804_380483


namespace total_roses_theorem_l3804_380448

/-- The number of bouquets to be made -/
def num_bouquets : ℕ := 5

/-- The number of table decorations to be made -/
def num_table_decorations : ℕ := 7

/-- The number of white roses used in each bouquet -/
def roses_per_bouquet : ℕ := 5

/-- The number of white roses used in each table decoration -/
def roses_per_table_decoration : ℕ := 12

/-- The total number of white roses needed for all bouquets and table decorations -/
def total_roses_needed : ℕ := num_bouquets * roses_per_bouquet + num_table_decorations * roses_per_table_decoration

theorem total_roses_theorem : total_roses_needed = 109 := by
  sorry

end total_roses_theorem_l3804_380448


namespace rectangle_length_problem_l3804_380421

theorem rectangle_length_problem (b : ℝ) (h1 : b > 0) : 
  (2 * b - 5) * (b + 5) - 2 * b * b = 75 → 2 * b = 40 := by
  sorry

end rectangle_length_problem_l3804_380421


namespace fourth_root_equation_l3804_380401

theorem fourth_root_equation (x : ℝ) (h : x > 0) :
  (x^3 * x^(1/2))^(1/4) = 4 → x = 4^(8/7) := by
  sorry

end fourth_root_equation_l3804_380401


namespace sqrt_a_plus_b_equals_four_l3804_380462

theorem sqrt_a_plus_b_equals_four :
  ∀ a b : ℕ,
  (a = ⌊Real.sqrt 17⌋) →
  (b - 1 = Real.sqrt 121) →
  Real.sqrt (a + b) = 4 :=
by
  sorry

end sqrt_a_plus_b_equals_four_l3804_380462


namespace a0_value_sum_of_all_coefficients_sum_of_odd_coefficients_l3804_380445

-- Define the polynomial coefficients
variable (a : Fin 8 → ℤ)

-- Define the equality condition
axiom expansion_equality : ∀ x : ℝ, (2*x - 1)^7 = (Finset.range 8).sum (λ i => a i * x^i)

-- Theorem statements
theorem a0_value : a 0 = -1 := by sorry

theorem sum_of_all_coefficients : (Finset.range 8).sum (λ i => a i) - a 0 = 2 := by sorry

theorem sum_of_odd_coefficients : a 1 + a 3 + a 5 + a 7 = -126 := by sorry

end a0_value_sum_of_all_coefficients_sum_of_odd_coefficients_l3804_380445


namespace cyclic_quadrilateral_theorem_l3804_380427

-- Define the points
variable (A B C D E F : EuclideanSpace ℝ (Fin 2))

-- Define the quadrilateral ABCD
def is_convex_quadrilateral (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define E as the intersection of angle bisectors of ∠B and ∠C
def is_angle_bisector_intersection (E B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define F as the intersection of AB and CD
def is_line_intersection (F A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the condition AB + CD = BC
def sum_equals_side (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := 
  dist A B + dist C D = dist B C

-- Define cyclic quadrilateral
def is_cyclic (A D E F : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Theorem statement
theorem cyclic_quadrilateral_theorem 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : is_angle_bisector_intersection E B C)
  (h3 : is_line_intersection F A B C D)
  (h4 : sum_equals_side A B C D) :
  is_cyclic A D E F := by sorry

end cyclic_quadrilateral_theorem_l3804_380427


namespace means_inequality_l3804_380455

theorem means_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  Real.sqrt ((a^2 + b^2) / 2) > (a + b) / 2 ∧
  (a + b) / 2 > Real.sqrt (a * b) ∧
  Real.sqrt (a * b) > 2 * a * b / (a + b) := by
  sorry

end means_inequality_l3804_380455


namespace sqrt_expression_equality_l3804_380482

theorem sqrt_expression_equality : 
  (Real.sqrt 2 - Real.sqrt 3) ^ 2020 * (Real.sqrt 2 + Real.sqrt 3) ^ 2021 = Real.sqrt 2 + Real.sqrt 3 := by
  sorry

end sqrt_expression_equality_l3804_380482


namespace max_area_rectangle_l3804_380460

/-- The perimeter of a rectangle -/
def perimeter (x y : ℝ) : ℝ := 2 * (x + y)

/-- The area of a rectangle -/
def area (x y : ℝ) : ℝ := x * y

/-- Theorem: For a rectangle with a fixed perimeter, the area is maximized when length equals width -/
theorem max_area_rectangle (p : ℝ) (hp : p > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ perimeter x y = p ∧
  (∀ (a b : ℝ), a > 0 → b > 0 → perimeter a b = p → area a b ≤ area x y) ∧
  x = p / 4 ∧ y = p / 4 := by
  sorry

#check max_area_rectangle

end max_area_rectangle_l3804_380460


namespace hyperbola_focus_l3804_380405

/-- Definition of the hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  -x^2 + 2*y^2 - 10*x - 16*y + 1 = 0

/-- Theorem stating that one of the foci of the hyperbola is at (-5, 7) or (-5, 1) -/
theorem hyperbola_focus :
  ∃ (x y : ℝ), hyperbola_equation x y ∧ ((x = -5 ∧ y = 7) ∨ (x = -5 ∧ y = 1)) :=
by sorry

end hyperbola_focus_l3804_380405


namespace tree_height_after_two_years_l3804_380442

/-- A tree that triples its height each year --/
def tree_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (3 ^ years)

/-- Theorem: A tree that triples its height each year and reaches 81 feet after 4 years
    will have a height of 9 feet after 2 years --/
theorem tree_height_after_two_years
  (h : tree_height (tree_height h₀ 2) 2 = 81)
  (h₀ : ℝ) :
  tree_height h₀ 2 = 9 :=
sorry

end tree_height_after_two_years_l3804_380442


namespace bakery_flour_usage_l3804_380457

theorem bakery_flour_usage (wheat_flour : Real) (white_flour : Real)
  (h1 : wheat_flour = 0.2)
  (h2 : white_flour = 0.1) :
  wheat_flour + white_flour = 0.3 := by
  sorry

end bakery_flour_usage_l3804_380457


namespace entertainment_unit_theorem_l3804_380420

/-- A structure representing the entertainment unit -/
structure EntertainmentUnit where
  singers : ℕ
  dancers : ℕ
  total : ℕ
  both : ℕ
  h_singers : singers = 4
  h_dancers : dancers = 5
  h_total : total = singers + dancers - both
  h_all_can : total ≤ singers + dancers

/-- The probability of selecting at least one person who can both sing and dance -/
def prob_at_least_one (u : EntertainmentUnit) : ℚ :=
  1 - (Nat.choose (u.total - u.both) 2 : ℚ) / (Nat.choose u.total 2 : ℚ)

/-- The probability distribution of ξ -/
def prob_dist (u : EntertainmentUnit) : ℕ → ℚ
| 0 => (Nat.choose (u.total - u.both) 2 : ℚ) / (Nat.choose u.total 2 : ℚ)
| 1 => (u.both * (u.total - u.both) : ℚ) / (Nat.choose u.total 2 : ℚ)
| 2 => (Nat.choose u.both 2 : ℚ) / (Nat.choose u.total 2 : ℚ)
| _ => 0

/-- The expected value of ξ -/
def expected_value (u : EntertainmentUnit) : ℚ :=
  0 * prob_dist u 0 + 1 * prob_dist u 1 + 2 * prob_dist u 2

/-- The main theorem -/
theorem entertainment_unit_theorem (u : EntertainmentUnit) 
  (h_prob : prob_at_least_one u = 11/21) : 
  u.total = 7 ∧ 
  prob_dist u 0 = 10/21 ∧ 
  prob_dist u 1 = 10/21 ∧ 
  prob_dist u 2 = 1/21 ∧
  expected_value u = 4/7 := by
  sorry

end entertainment_unit_theorem_l3804_380420


namespace remaining_cooking_time_l3804_380416

def total_potatoes : ℕ := 13
def cooked_potatoes : ℕ := 5
def cooking_time_per_potato : ℕ := 6

theorem remaining_cooking_time : (total_potatoes - cooked_potatoes) * cooking_time_per_potato = 48 := by
  sorry

end remaining_cooking_time_l3804_380416


namespace shelves_count_l3804_380440

/-- The number of shelves in a library --/
def number_of_shelves (books_per_shelf : ℕ) (total_round_trip_distance : ℕ) : ℕ :=
  (total_round_trip_distance / 2) / books_per_shelf

/-- Theorem: The number of shelves is 4 --/
theorem shelves_count :
  number_of_shelves 400 3200 = 4 := by
  sorry

end shelves_count_l3804_380440


namespace gcf_lcm_sum_8_12_l3804_380433

theorem gcf_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end gcf_lcm_sum_8_12_l3804_380433


namespace min_bottles_to_fill_container_l3804_380485

def container_capacity : ℕ := 1125
def bottle_type1_capacity : ℕ := 45
def bottle_type2_capacity : ℕ := 75

theorem min_bottles_to_fill_container :
  ∃ (n1 n2 : ℕ),
    n1 * bottle_type1_capacity + n2 * bottle_type2_capacity = container_capacity ∧
    ∀ (m1 m2 : ℕ), 
      m1 * bottle_type1_capacity + m2 * bottle_type2_capacity = container_capacity →
      n1 + n2 ≤ m1 + m2 ∧
    n1 + n2 = 15 := by
  sorry

end min_bottles_to_fill_container_l3804_380485


namespace fraction_addition_l3804_380425

theorem fraction_addition (c : ℝ) : (6 + 5 * c) / 9 + 3 = (33 + 5 * c) / 9 := by
  sorry

end fraction_addition_l3804_380425


namespace q_polynomial_form_l3804_380479

/-- Given a function q(x) satisfying the equation
    q(x) + (2x^6 + 5x^4 + 10x^2) = (9x^4 + 30x^3 + 50x^2 + 4),
    prove that q(x) = -2x^6 + 4x^4 + 30x^3 + 40x^2 + 4 -/
theorem q_polynomial_form (q : ℝ → ℝ) 
    (h : ∀ x, q x + (2 * x^6 + 5 * x^4 + 10 * x^2) = 9 * x^4 + 30 * x^3 + 50 * x^2 + 4) :
  ∀ x, q x = -2 * x^6 + 4 * x^4 + 30 * x^3 + 40 * x^2 + 4 := by
  sorry

end q_polynomial_form_l3804_380479
