import Mathlib

namespace optimal_cutting_l3557_355794

/-- Represents a rectangular piece of cardboard -/
structure Rectangle :=
  (length : ℕ)
  (width : ℕ)

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℕ :=
  r.length * r.width

/-- Represents the problem of cutting small rectangles from a large rectangle -/
structure CuttingProblem :=
  (large : Rectangle)
  (small : Rectangle)

/-- Calculates the maximum number of small rectangles that can be cut from a large rectangle -/
def maxPieces (p : CuttingProblem) : ℕ :=
  sorry

theorem optimal_cutting (p : CuttingProblem) 
  (h1 : p.large = ⟨17, 22⟩) 
  (h2 : p.small = ⟨3, 5⟩) : 
  maxPieces p = 24 :=
sorry

end optimal_cutting_l3557_355794


namespace investment_ratio_l3557_355726

/-- Represents an investor in a shop -/
structure Investor where
  name : String
  investment : ℕ
  profit_ratio : ℕ

/-- Represents a shop with two investors -/
structure Shop where
  investor1 : Investor
  investor2 : Investor

/-- Theorem stating the relationship between investments and profit ratios -/
theorem investment_ratio (shop : Shop) 
  (h1 : shop.investor1.profit_ratio = 2)
  (h2 : shop.investor2.profit_ratio = 4)
  (h3 : shop.investor2.investment = 1000000) :
  shop.investor1.investment = 500000 := by
  sorry

#check investment_ratio

end investment_ratio_l3557_355726


namespace linear_coefficient_of_example_quadratic_l3557_355731

/-- The coefficient of the linear term in a quadratic equation ax^2 + bx + c = 0 -/
def linearCoefficient (a b c : ℝ) : ℝ := b

theorem linear_coefficient_of_example_quadratic :
  linearCoefficient 1 (-5) (-2) = -5 := by sorry

end linear_coefficient_of_example_quadratic_l3557_355731


namespace only_vegetarian_count_l3557_355762

/-- Represents the number of people in a family with different eating habits -/
structure FamilyEatingHabits where
  only_non_veg : ℕ
  both_veg_and_non_veg : ℕ
  total_veg : ℕ

/-- Theorem stating the number of people who eat only vegetarian -/
theorem only_vegetarian_count (f : FamilyEatingHabits) 
  (h1 : f.only_non_veg = 6)
  (h2 : f.both_veg_and_non_veg = 9)
  (h3 : f.total_veg = 20) :
  f.total_veg - f.both_veg_and_non_veg = 11 := by
  sorry

end only_vegetarian_count_l3557_355762


namespace remainder_problem_l3557_355764

theorem remainder_problem (x y : ℤ) 
  (hx : x % 72 = 65) 
  (hy : y % 54 = 22) : 
  (x - y) % 18 = 7 := by sorry

end remainder_problem_l3557_355764


namespace expression_simplification_and_evaluation_l3557_355727

theorem expression_simplification_and_evaluation :
  ∀ x : ℤ, -3 < x → x < 3 → x ≠ -1 → x ≠ 1 → x ≠ 0 →
  (((x^2 - 2*x + 1) / (x^2 - 1)) / ((x - 1) / (x + 1) - x + 1) = -1 / x) ∧
  (x = -2 → -1 / x = 1 / 2) :=
by sorry

end expression_simplification_and_evaluation_l3557_355727


namespace sunflower_height_feet_l3557_355715

def feet_to_inches (feet : ℕ) : ℕ := feet * 12

def sister_height_inches : ℕ := feet_to_inches 4 + 3

def sunflower_height_inches : ℕ := sister_height_inches + 21

def inches_to_feet (inches : ℕ) : ℕ := inches / 12

theorem sunflower_height_feet :
  inches_to_feet sunflower_height_inches = 6 :=
sorry

end sunflower_height_feet_l3557_355715


namespace line_through_d_divides_equally_l3557_355754

-- Define the shape
structure Shape :=
  (area : ℝ)
  (is_unit_squares : Bool)

-- Define points
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a line
structure Line :=
  (point1 : Point)
  (point2 : Point)

-- Define the problem setup
def problem_setup (s : Shape) (p a b c d e : Point) : Prop :=
  s.is_unit_squares ∧
  s.area = 9 ∧
  b.x = (a.x + c.x) / 2 ∧
  b.y = (a.y + c.y) / 2 ∧
  d.x = (c.x + e.x) / 2 ∧
  d.y = (c.y + e.y) / 2

-- Define the division of area by a line
def divides_area_equally (l : Line) (s : Shape) : Prop :=
  ∃ (area1 area2 : ℝ), 
    area1 = area2 ∧
    area1 + area2 = s.area

-- Theorem statement
theorem line_through_d_divides_equally 
  (s : Shape) (p a b c d e : Point) (l : Line) :
  problem_setup s p a b c d e →
  l.point1 = p →
  l.point2 = d →
  divides_area_equally l s :=
sorry

end line_through_d_divides_equally_l3557_355754


namespace F_less_than_G_l3557_355750

theorem F_less_than_G : ∀ x : ℝ, (2 * x^2 - 3 * x - 2) < (3 * x^2 - 7 * x + 5) := by
  sorry

end F_less_than_G_l3557_355750


namespace trapezoid_vector_range_l3557_355714

/-- Right trapezoid ABCD with moving point P -/
structure Trapezoid where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  P : ℝ × ℝ
  h : A.2 = D.2  -- AB ⟂ AD
  i : D.1 - A.1 = 1  -- AD = 1
  j : C.1 - D.1 = 1  -- DC = 1
  k : B.2 - A.2 = 3  -- AB = 3
  l : (P.1 - C.1)^2 + (P.2 - C.2)^2 ≤ 1  -- P is within or on the circle centered at C with radius 1

def vector_decomposition (t : Trapezoid) (α β : ℝ) : Prop :=
  t.P.1 - t.A.1 = α * (t.D.1 - t.A.1) + β * (t.B.1 - t.A.1) ∧
  t.P.2 - t.A.2 = α * (t.D.2 - t.A.2) + β * (t.B.2 - t.A.2)

theorem trapezoid_vector_range (t : Trapezoid) :
  ∃ (α β : ℝ), vector_decomposition t α β ∧ 
  (∀ (γ δ : ℝ), vector_decomposition t γ δ → 1 < γ + δ ∧ γ + δ < 5/3) :=
sorry

end trapezoid_vector_range_l3557_355714


namespace alice_sequence_characterization_l3557_355724

/-- Represents the sequence of numbers generated by Alice's operations -/
def AliceSequence (a₀ : ℕ+) : ℕ → ℚ
| 0 => a₀
| (n+1) => if AliceSequence a₀ n > 8763 then 1 / (AliceSequence a₀ n)
           else if AliceSequence a₀ n ≤ 8763 ∧ AliceSequence a₀ (n-1) = 1 / (AliceSequence a₀ (n-2))
                then 2 * (AliceSequence a₀ n) + 1
           else if (AliceSequence a₀ n).den = 1 then 1 / (AliceSequence a₀ n)
           else 2 * (AliceSequence a₀ n) + 1

/-- The set of indices where the sequence value is a natural number -/
def NaturalIndices (a₀ : ℕ+) : Set ℕ :=
  {i | (AliceSequence a₀ i).den = 1}

/-- The theorem stating the characterization of initial values -/
theorem alice_sequence_characterization :
  {a₀ : ℕ+ | Set.Infinite (NaturalIndices a₀)} =
  {a₀ : ℕ+ | a₀ ≤ 17526 ∧ Even a₀} :=
sorry

end alice_sequence_characterization_l3557_355724


namespace cos_plus_sin_range_l3557_355765

/-- 
Given a point P(x,1) where x ≥ 1 on the terminal side of angle θ in the Cartesian coordinate system,
the sum of cosine and sine of θ is strictly greater than 1 and less than or equal to √2.
-/
theorem cos_plus_sin_range (x : ℝ) (θ : ℝ) (h1 : x ≥ 1) 
  (h2 : x = Real.cos θ * Real.sqrt (x^2 + 1)) 
  (h3 : 1 = Real.sin θ * Real.sqrt (x^2 + 1)) : 
  1 < Real.cos θ + Real.sin θ ∧ Real.cos θ + Real.sin θ ≤ Real.sqrt 2 := by
sorry

end cos_plus_sin_range_l3557_355765


namespace toys_in_box_time_l3557_355784

/-- The time required to put all toys in the box -/
def time_to_put_toys_in_box (total_toys : ℕ) (net_increase_per_minute : ℕ) : ℕ :=
  ((total_toys - net_increase_per_minute) / net_increase_per_minute) + 1

/-- Theorem: It takes 15 minutes to put 45 toys in the box with a net increase of 3 toys per minute -/
theorem toys_in_box_time : time_to_put_toys_in_box 45 3 = 15 := by
  sorry

end toys_in_box_time_l3557_355784


namespace smallest_angle_theorem_l3557_355719

/-- The smallest positive angle y in degrees that satisfies the equation 
    9 sin(y) cos³(y) - 9 sin³(y) cos(y) = 3√2 is 22.5° -/
theorem smallest_angle_theorem : 
  ∃ y : ℝ, y > 0 ∧ y < 360 ∧ 
  (9 * Real.sin y * (Real.cos y)^3 - 9 * (Real.sin y)^3 * Real.cos y = 3 * Real.sqrt 2) ∧
  (∀ z : ℝ, z > 0 ∧ z < y → 
    9 * Real.sin z * (Real.cos z)^3 - 9 * (Real.sin z)^3 * Real.cos z ≠ 3 * Real.sqrt 2) ∧
  y = 22.5 := by
  sorry

end smallest_angle_theorem_l3557_355719


namespace two_distinct_solutions_l3557_355773

/-- The cubic equation in x with parameter a -/
def cubic_equation (a x : ℝ) : ℝ := x^3 - 2*a*x^2 - 3*a*x + a^2 - 2

/-- Theorem stating the condition for the cubic equation to have exactly two distinct real solutions -/
theorem two_distinct_solutions (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    cubic_equation a x = 0 ∧ 
    cubic_equation a y = 0 ∧ 
    ∀ z : ℝ, cubic_equation a z = 0 → z = x ∨ z = y) ↔ 
  a > 15/8 :=
sorry

end two_distinct_solutions_l3557_355773


namespace arithmetic_computation_l3557_355733

theorem arithmetic_computation : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 := by
  sorry

end arithmetic_computation_l3557_355733


namespace all_symmetry_statements_correct_l3557_355782

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define symmetry with respect to y-axis
def symmetric_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define symmetry with respect to x-axis
def symmetric_x_axis (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f x

-- Define symmetry with respect to origin
def symmetric_origin (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

-- Define symmetry with respect to vertical line x = a
def symmetric_vertical_line (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

theorem all_symmetry_statements_correct (f : ℝ → ℝ) : 
  (symmetric_y_axis f) ∧ 
  (symmetric_x_axis f) ∧ 
  (symmetric_origin f) ∧ 
  (∀ a : ℝ, symmetric_vertical_line f a → 
    ∃ g : ℝ → ℝ, ∀ x, g (x - a) = f x) :=
by sorry

end all_symmetry_statements_correct_l3557_355782


namespace team_selection_ways_eq_8400_l3557_355799

/-- The number of ways to select a team of 4 boys from 8 boys and 3 girls from 10 girls -/
def team_selection_ways : ℕ :=
  (Nat.choose 8 4) * (Nat.choose 10 3)

/-- Theorem stating that the number of ways to select the team is 8400 -/
theorem team_selection_ways_eq_8400 : team_selection_ways = 8400 := by
  sorry

end team_selection_ways_eq_8400_l3557_355799


namespace tulip_count_after_addition_tulip_count_is_24_l3557_355774

/-- Given a garden with tulips and sunflowers, prove the number of tulips after an addition of sunflowers. -/
theorem tulip_count_after_addition 
  (initial_ratio : Rat) 
  (initial_sunflowers : Nat) 
  (added_sunflowers : Nat) : Nat :=
  let final_sunflowers := initial_sunflowers + added_sunflowers
  let tulip_ratio := 3
  let sunflower_ratio := 7
  (tulip_ratio * final_sunflowers) / sunflower_ratio

#check tulip_count_after_addition (3/7) 42 14 = 24

/-- Prove that the result is indeed 24 -/
theorem tulip_count_is_24 : 
  tulip_count_after_addition (3/7) 42 14 = 24 := by
  sorry


end tulip_count_after_addition_tulip_count_is_24_l3557_355774


namespace seminar_scheduling_l3557_355778

theorem seminar_scheduling (n : ℕ) (h : n = 5) : 
  (n! / 2 : ℕ) = 60 :=
sorry

end seminar_scheduling_l3557_355778


namespace jensens_inequality_l3557_355745

/-- Jensen's inequality for convex functions -/
theorem jensens_inequality (f : ℝ → ℝ) (hf : ConvexOn ℝ Set.univ f) 
  (x₁ x₂ q₁ q₂ : ℝ) (hq₁ : q₁ > 0) (hq₂ : q₂ > 0) (hsum : q₁ + q₂ = 1) :
  f (q₁ * x₁ + q₂ * x₂) ≤ q₁ * f x₁ + q₂ * f x₂ := by
  sorry

end jensens_inequality_l3557_355745


namespace stratified_sample_male_count_l3557_355703

theorem stratified_sample_male_count :
  let total_male : ℕ := 560
  let total_female : ℕ := 420
  let sample_size : ℕ := 280
  let total_students : ℕ := total_male + total_female
  let male_ratio : ℚ := total_male / total_students
  male_ratio * sample_size = 160 := by
  sorry

end stratified_sample_male_count_l3557_355703


namespace expense_increase_percentage_l3557_355798

def monthly_salary : ℚ := 4166.67
def initial_savings_rate : ℚ := 0.20
def new_savings : ℚ := 500

def initial_savings : ℚ := monthly_salary * initial_savings_rate
def original_expenses : ℚ := monthly_salary - initial_savings
def increase_in_expenses : ℚ := initial_savings - new_savings
def percentage_increase : ℚ := (increase_in_expenses / original_expenses) * 100

theorem expense_increase_percentage :
  percentage_increase = 10 := by sorry

end expense_increase_percentage_l3557_355798


namespace benny_bought_two_cards_l3557_355787

/-- The number of Pokemon cards Benny bought -/
def cards_bought (initial_cards final_cards : ℕ) : ℕ :=
  initial_cards - final_cards

/-- Proof that Benny bought 2 Pokemon cards -/
theorem benny_bought_two_cards :
  let initial_cards := 3
  let final_cards := 1
  cards_bought initial_cards final_cards = 2 := by
sorry

end benny_bought_two_cards_l3557_355787


namespace no_two_digit_special_number_l3557_355702

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def tens_digit (n : ℕ) : ℕ := n / 10
def ones_digit (n : ℕ) : ℕ := n % 10

def sum_of_digits (n : ℕ) : ℕ := tens_digit n + ones_digit n
def product_of_digits (n : ℕ) : ℕ := tens_digit n * ones_digit n

theorem no_two_digit_special_number :
  ¬∃ n : ℕ, is_two_digit n ∧ 
    (sum_of_digits n + 1) ∣ n ∧ 
    (product_of_digits n + 1) ∣ n :=
by sorry

end no_two_digit_special_number_l3557_355702


namespace parabola_vertex_coordinates_l3557_355743

/-- The vertex coordinates of a parabola in the form y = -(x + h)^2 + k are (h, k) -/
theorem parabola_vertex_coordinates (h k : ℝ) :
  let f : ℝ → ℝ := λ x => -(x + h)^2 + k
  (∀ x, f x = -(x + h)^2 + k) →
  (h, k) = Prod.mk (- h) k :=
sorry

end parabola_vertex_coordinates_l3557_355743


namespace flu_infection_rate_l3557_355746

theorem flu_infection_rate : 
  ∀ (x : ℝ), 
  (1 : ℝ) + x + x * ((1 : ℝ) + x) = 144 → 
  x = 11 := by
sorry

end flu_infection_rate_l3557_355746


namespace mom_bought_71_packages_l3557_355777

/-- The number of t-shirts in each package -/
def shirts_per_package : ℕ := 6

/-- The total number of t-shirts Mom has -/
def total_shirts : ℕ := 426

/-- The number of packages Mom bought -/
def packages_bought : ℕ := total_shirts / shirts_per_package

theorem mom_bought_71_packages : packages_bought = 71 := by
  sorry

end mom_bought_71_packages_l3557_355777


namespace smallest_positive_multiple_of_45_l3557_355732

theorem smallest_positive_multiple_of_45 : 
  ∀ n : ℕ, n > 0 → 45 ∣ n → n ≥ 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l3557_355732


namespace total_deduction_in_cents_l3557_355749

/-- Elena's hourly wage in dollars -/
def hourly_wage : ℝ := 25

/-- Local tax rate as a decimal -/
def tax_rate : ℝ := 0.02

/-- Health benefit rate as a decimal -/
def health_rate : ℝ := 0.015

/-- Conversion rate from dollars to cents -/
def dollars_to_cents : ℝ := 100

/-- Theorem stating the total deduction in cents -/
theorem total_deduction_in_cents : 
  hourly_wage * dollars_to_cents * (tax_rate + health_rate) = 87.5 := by
  sorry

end total_deduction_in_cents_l3557_355749


namespace books_division_l3557_355766

theorem books_division (total_books : ℕ) (divisions : ℕ) (final_category_size : ℕ) : 
  total_books = 400 → divisions = 4 → final_category_size = total_books / (2^divisions) → 
  final_category_size = 25 := by
sorry

end books_division_l3557_355766


namespace cone_surface_area_l3557_355772

/-- The surface area of a cone, given its lateral surface properties -/
theorem cone_surface_area (r l θ : Real) (h1 : l = 1) (h2 : θ = π / 2) :
  let lateral_area := π * r * l
  let base_area := π * r^2
  lateral_area = l^2 * θ / 2 →
  lateral_area + base_area = 5 * π / 16 := by
  sorry

end cone_surface_area_l3557_355772


namespace fraction_equality_l3557_355768

/-- Given that (Bx-13)/(x^2-7x+10) = A/(x-2) + 5/(x-5) for all x ≠ 2 and x ≠ 5,
    prove that A = 3/5, B = 28/5, and A + B = 31/5 -/
theorem fraction_equality (A B : ℚ) : 
  (∀ x : ℚ, x ≠ 2 → x ≠ 5 → (B * x - 13) / (x^2 - 7*x + 10) = A / (x - 2) + 5 / (x - 5)) →
  A = 3/5 ∧ B = 28/5 ∧ A + B = 31/5 := by
  sorry

end fraction_equality_l3557_355768


namespace four_item_match_probability_correct_match_probability_theorem_l3557_355752

/-- The probability of correctly matching n distinct items to n distinct positions when guessing randomly. -/
def correct_match_probability (n : ℕ) : ℚ :=
  1 / n.factorial

/-- Theorem: For 4 items, the probability of a correct random match is 1/24. -/
theorem four_item_match_probability :
  correct_match_probability 4 = 1 / 24 := by
  sorry

/-- Theorem: The probability of correctly matching n distinct items to n distinct positions
    when guessing randomly is 1/n!. -/
theorem correct_match_probability_theorem (n : ℕ) :
  correct_match_probability n = 1 / n.factorial := by
  sorry

end four_item_match_probability_correct_match_probability_theorem_l3557_355752


namespace cucumber_salad_problem_l3557_355781

theorem cucumber_salad_problem (total : ℕ) (ratio : ℕ) : 
  total = 280 → ratio = 3 → ∃ (cucumbers : ℕ), cucumbers * (ratio + 1) = total ∧ cucumbers = 70 :=
by
  sorry

end cucumber_salad_problem_l3557_355781


namespace mark_radiator_cost_l3557_355722

/-- The total cost Mark paid for replacing his car radiator -/
def total_cost (labor_hours : ℕ) (hourly_rate : ℕ) (part_cost : ℕ) : ℕ :=
  labor_hours * hourly_rate + part_cost

/-- Theorem stating that Mark paid $300 for replacing his car radiator -/
theorem mark_radiator_cost :
  total_cost 2 75 150 = 300 := by
  sorry

end mark_radiator_cost_l3557_355722


namespace M_less_than_N_l3557_355740

theorem M_less_than_N (x y : ℝ) (α : ℝ) (hx : x > 0) (hy : y > 0) :
  x^(Real.sin α)^2 * y^(Real.cos α)^2 < x + y := by
  sorry

end M_less_than_N_l3557_355740


namespace f_increasing_l3557_355728

-- Define the function f(x) = x^3 + x
def f (x : ℝ) : ℝ := x^3 + x

-- State the theorem
theorem f_increasing : ∀ (a b : ℝ), a < b → f a < f b := by
  sorry

end f_increasing_l3557_355728


namespace curve_k_range_l3557_355759

theorem curve_k_range (a : ℝ) (k : ℝ) : 
  ((-a)^2 - a*(-a) + 2*a + k = 0) → k ≤ (1/2 : ℝ) ∧ ∀ (ε : ℝ), ε > 0 → ∃ (k' : ℝ), k' < -ε ∧ ∃ (a' : ℝ), ((-a')^2 - a'*(-a') + 2*a' + k' = 0) :=
by sorry

end curve_k_range_l3557_355759


namespace profit_percentage_l3557_355729

theorem profit_percentage (C S : ℝ) (h : 72 * C = 60 * S) : 
  (S - C) / C * 100 = 20 := by
  sorry

end profit_percentage_l3557_355729


namespace beatrice_auction_tvs_l3557_355734

/-- The number of TVs Beatrice looked at on the auction site -/
def auction_tvs (in_person : ℕ) (online_multiplier : ℕ) (total : ℕ) : ℕ :=
  total - (in_person + online_multiplier * in_person)

/-- Proof that Beatrice looked at 10 TVs on the auction site -/
theorem beatrice_auction_tvs :
  auction_tvs 8 3 42 = 10 := by
  sorry

end beatrice_auction_tvs_l3557_355734


namespace lucas_future_age_l3557_355790

def age_problem (gladys_age billy_age lucas_age : ℕ) : Prop :=
  (gladys_age = 30) ∧
  (gladys_age = 3 * billy_age) ∧
  (gladys_age = 2 * (billy_age + lucas_age))

theorem lucas_future_age 
  (gladys_age billy_age lucas_age : ℕ) 
  (h : age_problem gladys_age billy_age lucas_age) : 
  lucas_age + 3 = 8 := by
  sorry

end lucas_future_age_l3557_355790


namespace complex_sum_theorem_l3557_355712

theorem complex_sum_theorem (x : ℂ) (h1 : x^7 = 1) (h2 : x ≠ 1) :
  (x^2)/(x-1) + (x^4)/(x^2-1) + (x^6)/(x^3-1) + (x^8)/(x^4-1) + (x^10)/(x^5-1) + (x^12)/(x^6-1) = 2 := by
  sorry

end complex_sum_theorem_l3557_355712


namespace functional_equation_problem_l3557_355753

theorem functional_equation_problem (f : ℝ → ℝ) 
  (h1 : ∀ a b : ℝ, f (a + b) = f a * f b) 
  (h2 : f 1 = 2) : 
  f 1^2 + f 2 / f 1 + f 2^2 + f 4 / f 3 + f 3^2 + f 6 / f 5 + f 4^2 + f 8 / f 7 = 16 := by
  sorry

end functional_equation_problem_l3557_355753


namespace range_of_a_l3557_355705

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 4}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*a*x + a + 2 = 0}

-- State the theorem
theorem range_of_a (a : ℝ) : B a ⊆ A → a ∈ Set.Ioo (-1 : ℝ) (18/7 : ℝ) := by
  sorry

end range_of_a_l3557_355705


namespace initial_number_count_l3557_355788

theorem initial_number_count (n : ℕ) (S : ℝ) : 
  n > 0 ∧ 
  S / n = 12 ∧ 
  (S - 20) / (n - 1) = 10 →
  n = 5 := by
sorry

end initial_number_count_l3557_355788


namespace product_digit_permutation_l3557_355713

theorem product_digit_permutation :
  ∃ (x : ℕ) (A B C D : ℕ),
    x * (x + 1) = 1000 * A + 100 * B + 10 * C + D ∧
    (x - 3) * (x - 2) = 1000 * C + 100 * A + 10 * B + D ∧
    (x - 30) * (x - 29) = 1000 * B + 100 * C + 10 * A + D ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧
    x = 91 ∧ A = 8 ∧ B = 3 ∧ C = 7 ∧ D = 2 :=
by sorry

end product_digit_permutation_l3557_355713


namespace correct_conic_propositions_l3557_355775

/-- Represents a proposition about conic sections -/
inductive ConicProposition
| Prop1
| Prop2
| Prop3
| Prop4
| Prop5

/-- Determines if a given proposition is correct -/
def is_correct (prop : ConicProposition) : Bool :=
  match prop with
  | ConicProposition.Prop1 => true
  | ConicProposition.Prop2 => false
  | ConicProposition.Prop3 => false
  | ConicProposition.Prop4 => true
  | ConicProposition.Prop5 => false

/-- The theorem to be proved -/
theorem correct_conic_propositions :
  (List.filter is_correct [ConicProposition.Prop1, ConicProposition.Prop2, 
                           ConicProposition.Prop3, ConicProposition.Prop4, 
                           ConicProposition.Prop5]).length = 2 := by
  sorry

end correct_conic_propositions_l3557_355775


namespace integer_pair_solution_l3557_355771

theorem integer_pair_solution (a b : ℤ) : 
  (a + b) / (a - b) = 3 ∧ (a + b) * (a - b) = 300 →
  (a = 20 ∧ b = 10) ∨ (a = -20 ∧ b = -10) := by
sorry

end integer_pair_solution_l3557_355771


namespace bakery_revenue_l3557_355785

/-- Calculates the total revenue from selling pumpkin and custard pies --/
def total_revenue (pumpkin_slices_per_pie : ℕ) (custard_slices_per_pie : ℕ) 
                  (pumpkin_price_per_slice : ℕ) (custard_price_per_slice : ℕ) 
                  (pumpkin_pies_sold : ℕ) (custard_pies_sold : ℕ) : ℕ :=
  (pumpkin_slices_per_pie * pumpkin_pies_sold * pumpkin_price_per_slice) +
  (custard_slices_per_pie * custard_pies_sold * custard_price_per_slice)

theorem bakery_revenue : 
  total_revenue 8 6 5 6 4 5 = 340 := by
  sorry

end bakery_revenue_l3557_355785


namespace parallelogram_means_input_output_l3557_355748

/-- Represents the different symbols used in a program flowchart --/
inductive FlowchartSymbol
  | Parallelogram
  | Rectangle
  | Diamond
  | Oval

/-- Represents the different operations in a program flowchart --/
inductive FlowchartOperation
  | InputOutput
  | Process
  | Decision
  | Start_End

/-- Associates a FlowchartSymbol with its corresponding FlowchartOperation --/
def symbolMeaning : FlowchartSymbol → FlowchartOperation
  | FlowchartSymbol.Parallelogram => FlowchartOperation.InputOutput
  | FlowchartSymbol.Rectangle => FlowchartOperation.Process
  | FlowchartSymbol.Diamond => FlowchartOperation.Decision
  | FlowchartSymbol.Oval => FlowchartOperation.Start_End

theorem parallelogram_means_input_output :
  symbolMeaning FlowchartSymbol.Parallelogram = FlowchartOperation.InputOutput :=
by sorry

end parallelogram_means_input_output_l3557_355748


namespace quadruplet_babies_l3557_355780

theorem quadruplet_babies (total_babies : ℕ) 
  (h_total : total_babies = 1250)
  (h_twins_quintuplets : ∃ t p : ℕ, t = 4 * p)
  (h_triplets_quadruplets : ∃ r q : ℕ, r = 2 * q)
  (h_quadruplets_quintuplets : ∃ q p : ℕ, q = 2 * p)
  (h_sum : ∃ t r q p : ℕ, 2 * t + 3 * r + 4 * q + 5 * p = total_babies) :
  ∃ q : ℕ, 4 * q = 303 :=
by sorry

end quadruplet_babies_l3557_355780


namespace land_plot_side_length_l3557_355742

/-- For a square-shaped land plot with an area of 100 square units, 
    the length of one side is 10 units. -/
theorem land_plot_side_length (area : ℝ) (side : ℝ) : 
  area = 100 → side * side = area → side = 10 := by
  sorry

end land_plot_side_length_l3557_355742


namespace basis_transformation_l3557_355738

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem basis_transformation (OA OB OC : V) 
  (h : LinearIndependent ℝ ![OA, OB, OC]) 
  (h_span : Submodule.span ℝ {OA, OB, OC} = ⊤) :
  LinearIndependent ℝ ![OA + OB, OA - OB, OC] ∧ 
  Submodule.span ℝ {OA + OB, OA - OB, OC} = ⊤ := by
  sorry

end basis_transformation_l3557_355738


namespace sector_area_l3557_355789

/-- Given a sector with central angle 60° and arc length π, its area is 3π/2 -/
theorem sector_area (angle : Real) (arc_length : Real) (area : Real) :
  angle = 60 * (π / 180) →
  arc_length = π →
  area = (angle / (2 * π)) * arc_length * arc_length / angle →
  area = 3 * π / 2 := by
  sorry

end sector_area_l3557_355789


namespace factorial_236_trailing_zeros_l3557_355770

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: 236! has 57 trailing zeros -/
theorem factorial_236_trailing_zeros :
  trailingZeros 236 = 57 := by sorry

end factorial_236_trailing_zeros_l3557_355770


namespace sum_of_distinct_prime_divisors_of_2520_l3557_355797

theorem sum_of_distinct_prime_divisors_of_2520 : 
  (Finset.sum (Finset.filter Nat.Prime (Nat.divisors 2520)) id) = 17 := by
  sorry

end sum_of_distinct_prime_divisors_of_2520_l3557_355797


namespace pages_multiple_l3557_355704

theorem pages_multiple (beatrix_pages cristobal_extra_pages : ℕ) 
  (h1 : beatrix_pages = 704)
  (h2 : cristobal_extra_pages = 1423)
  (h3 : ∃ x : ℕ, x * beatrix_pages + 15 = cristobal_extra_pages) :
  ∃ x : ℕ, x * beatrix_pages + 15 = cristobal_extra_pages ∧ x = 2 := by
  sorry

end pages_multiple_l3557_355704


namespace angle_triple_complement_l3557_355791

theorem angle_triple_complement (x : ℝ) : x = 3 * (90 - x) → x = 67.5 := by
  sorry

end angle_triple_complement_l3557_355791


namespace museum_visit_orders_l3557_355735

-- Define the number of museums
def n : ℕ := 5

-- Define the factorial function
def factorial (m : ℕ) : ℕ :=
  match m with
  | 0 => 1
  | k + 1 => (k + 1) * factorial k

-- Theorem: The number of permutations of n distinct objects is n!
theorem museum_visit_orders : factorial n = 120 := by
  sorry

end museum_visit_orders_l3557_355735


namespace orange_distribution_l3557_355736

theorem orange_distribution (x : ℚ) : 
  (x/2 + 1/2) + (1/2 * (x/2 - 1/2) + 1/2) + (1/2 * (x/4 - 3/4) + 1/2) = x → x = 7 := by
  sorry

end orange_distribution_l3557_355736


namespace equation_proof_l3557_355776

theorem equation_proof : (100 - 6) * 7 - 52 + 8 + 9 = 623 := by
  sorry

end equation_proof_l3557_355776


namespace parabola_max_value_l3557_355761

theorem parabola_max_value (x : ℝ) : 
  ∃ (max : ℝ), max = 6 ∧ ∀ y : ℝ, y = -3 * x^2 + 6 → y ≤ max :=
sorry

end parabola_max_value_l3557_355761


namespace carla_teaches_23_students_l3557_355717

/-- The number of students Carla teaches -/
def total_students : ℕ :=
  let students_in_restroom : ℕ := 2
  let absent_students : ℕ := 3 * students_in_restroom - 1
  let total_desks : ℕ := 4 * 6
  let occupied_desks : ℕ := (2 * total_desks) / 3
  occupied_desks + students_in_restroom + absent_students

/-- Theorem stating that Carla teaches 23 students -/
theorem carla_teaches_23_students : total_students = 23 := by
  sorry

end carla_teaches_23_students_l3557_355717


namespace rectangle_area_l3557_355718

/-- The area of a rectangle with dimensions 0.5 meters and 0.36 meters is 1800 square centimeters. -/
theorem rectangle_area : 
  let length_m : ℝ := 0.5
  let width_m : ℝ := 0.36
  let cm_per_m : ℝ := 100
  let length_cm := length_m * cm_per_m
  let width_cm := width_m * cm_per_m
  length_cm * width_cm = 1800 := by
  sorry

end rectangle_area_l3557_355718


namespace thirty_percent_less_problem_l3557_355711

theorem thirty_percent_less_problem (x : ℝ) : 
  (63 = 90 - 0.3 * 90) → (x + 0.25 * x = 63) → x = 50 := by
  sorry

end thirty_percent_less_problem_l3557_355711


namespace sum_of_max_min_is_negative_one_l3557_355769

-- Define the function f(x)
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- Define the interval
def interval : Set ℝ := Set.Icc (-2) 3

-- State the theorem
theorem sum_of_max_min_is_negative_one :
  ∃ (max min : ℝ), 
    (∀ x ∈ interval, f x ≤ max) ∧
    (∃ x ∈ interval, f x = max) ∧
    (∀ x ∈ interval, min ≤ f x) ∧
    (∃ x ∈ interval, f x = min) ∧
    max + min = -1 := by sorry

end sum_of_max_min_is_negative_one_l3557_355769


namespace circle_passes_through_points_l3557_355756

/-- The circle equation passing through points A(4, 1), B(6, -3), and C(-3, 0) -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 6*y - 15 = 0

/-- Point A coordinates -/
def point_A : ℝ × ℝ := (4, 1)

/-- Point B coordinates -/
def point_B : ℝ × ℝ := (6, -3)

/-- Point C coordinates -/
def point_C : ℝ × ℝ := (-3, 0)

theorem circle_passes_through_points :
  circle_equation point_A.1 point_A.2 ∧
  circle_equation point_B.1 point_B.2 ∧
  circle_equation point_C.1 point_C.2 :=
by sorry

end circle_passes_through_points_l3557_355756


namespace triple_composition_fixed_point_implies_fixed_point_l3557_355710

theorem triple_composition_fixed_point_implies_fixed_point
  (f : ℝ → ℝ) (hf : Continuous f)
  (h : ∃ x, f (f (f x)) = x) :
  ∃ x₀, f x₀ = x₀ := by
  sorry

end triple_composition_fixed_point_implies_fixed_point_l3557_355710


namespace expected_score_is_one_l3557_355701

/-- The number of black balls in the bag -/
def num_black : ℕ := 3

/-- The number of red balls in the bag -/
def num_red : ℕ := 1

/-- The total number of balls in the bag -/
def total_balls : ℕ := num_black + num_red

/-- The score for drawing a black ball -/
def black_score : ℝ := 0

/-- The score for drawing a red ball -/
def red_score : ℝ := 2

/-- The expected value of the score when drawing two balls -/
def expected_score : ℝ := 1

/-- Theorem stating that the expected score when drawing two balls is 1 -/
theorem expected_score_is_one :
  let prob_two_black : ℝ := (num_black / total_balls) * ((num_black - 1) / (total_balls - 1))
  let prob_one_each : ℝ := (num_black / total_balls) * (num_red / (total_balls - 1)) +
                           (num_red / total_balls) * (num_black / (total_balls - 1))
  prob_two_black * (2 * black_score) + prob_one_each * (black_score + red_score) = expected_score :=
by sorry

end expected_score_is_one_l3557_355701


namespace contrapositive_equivalence_l3557_355720

theorem contrapositive_equivalence (a : ℝ) : 
  (¬(a > 1) → ¬(a > 0)) ↔ (a ≤ 1 → a ≤ 0) :=
by sorry

end contrapositive_equivalence_l3557_355720


namespace equation_solution_l3557_355723

theorem equation_solution : ∃ x : ℕ, (8000 * 6000 : ℕ) = x * (10^5 : ℕ) ∧ x = 480 := by
  sorry

end equation_solution_l3557_355723


namespace star_value_zero_l3557_355795

-- Define the star operation
def star (a b c : ℤ) : ℤ := (a + b + c)^2

-- Theorem statement
theorem star_value_zero : star 3 (-5) 2 = 0 := by
  sorry

end star_value_zero_l3557_355795


namespace f_zero_range_l3557_355792

/-- The function f(x) = x^3 + 2x - a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 2*x - a

/-- The theorem stating that if f(x) has exactly one zero in (1, 2), then a is in (3, 12) -/
theorem f_zero_range (a : ℝ) : 
  (∃! x, x ∈ (Set.Ioo 1 2) ∧ f a x = 0) → a ∈ Set.Ioo 3 12 := by
  sorry

end f_zero_range_l3557_355792


namespace total_cost_is_246_l3557_355709

/-- Represents a person's balloon collection --/
structure BalloonCollection where
  yellowCount : Nat
  yellowPrice : Nat
  redCount : Nat
  redPrice : Nat

/-- Calculates the total cost of a balloon collection --/
def totalCost (bc : BalloonCollection) : Nat :=
  bc.yellowCount * bc.yellowPrice + bc.redCount * bc.redPrice

/-- The balloon collections for each person --/
def fred : BalloonCollection := ⟨5, 3, 3, 4⟩
def sam : BalloonCollection := ⟨6, 4, 4, 5⟩
def mary : BalloonCollection := ⟨7, 5, 5, 6⟩
def susan : BalloonCollection := ⟨4, 6, 6, 7⟩
def tom : BalloonCollection := ⟨10, 2, 8, 3⟩

/-- Theorem: The total cost of all balloon collections is $246 --/
theorem total_cost_is_246 :
  totalCost fred + totalCost sam + totalCost mary + totalCost susan + totalCost tom = 246 := by
  sorry

end total_cost_is_246_l3557_355709


namespace unknown_room_width_is_15_l3557_355721

-- Define the room dimensions
def room_length : ℝ := 25
def room_height : ℝ := 12
def door_area : ℝ := 6 * 3
def window_area : ℝ := 4 * 3
def num_windows : ℕ := 3
def cost_per_sqft : ℝ := 10
def total_cost : ℝ := 9060

-- Define the function to calculate the total area to be whitewashed
def area_to_whitewash (x : ℝ) : ℝ :=
  2 * (room_length + x) * room_height - (door_area + num_windows * window_area)

-- Define the theorem
theorem unknown_room_width_is_15 :
  ∃ x : ℝ, x > 0 ∧ cost_per_sqft * area_to_whitewash x = total_cost ∧ x = 15 := by
  sorry

end unknown_room_width_is_15_l3557_355721


namespace min_point_of_translated_abs_function_l3557_355763

-- Define the function
def f (x : ℝ) : ℝ := |x - 4| - 2

-- State the theorem
theorem min_point_of_translated_abs_function :
  ∃ (x₀ : ℝ), (∀ (x : ℝ), f x₀ ≤ f x) ∧ (x₀ = 4 ∧ f x₀ = -2) :=
sorry

end min_point_of_translated_abs_function_l3557_355763


namespace sum_of_mobile_keypad_numbers_l3557_355758

def mobile_keypad : List Nat := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

theorem sum_of_mobile_keypad_numbers : 
  mobile_keypad.sum = 45 := by
  sorry

end sum_of_mobile_keypad_numbers_l3557_355758


namespace opposite_sign_implies_y_power_x_25_l3557_355793

theorem opposite_sign_implies_y_power_x_25 (x y : ℝ) : 
  (((x - 2)^2 > 0 ∧ |5 + y| < 0) ∨ ((x - 2)^2 < 0 ∧ |5 + y| > 0)) → y^x = 25 := by
  sorry

end opposite_sign_implies_y_power_x_25_l3557_355793


namespace problem_solution_l3557_355783

theorem problem_solution (a b c d e : ℝ) 
  (h1 : a * b = 1)  -- a and b are reciprocals
  (h2 : c + d = 0)  -- c and d are opposites
  (h3 : e < 0)
  (h4 : |e| = 1) :
  (-a*b)^2009 - (c+d)^2010 - e^2011 = 0 := by
  sorry

end problem_solution_l3557_355783


namespace union_of_M_and_N_l3557_355706

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def N : Set ℝ := {x | 2*x + 1 < 5}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = {x : ℝ | x < 3} := by sorry

end union_of_M_and_N_l3557_355706


namespace set_intersection_theorem_l3557_355737

def A : Set ℝ := { x | x ≥ 1 }
def B : Set ℝ := { x | x^2 ≤ 4 }

theorem set_intersection_theorem :
  A ∩ B = { x : ℝ | 1 ≤ x ∧ x ≤ 2 } := by sorry

end set_intersection_theorem_l3557_355737


namespace max_brownies_l3557_355707

theorem max_brownies (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) : 
  (m - 2) * (n - 2) = 2 * m + 2 * n - 4 → m * n ≤ 60 := by
sorry

end max_brownies_l3557_355707


namespace intersection_complement_theorem_l3557_355796

universe u

def M : Set ℤ := {-1, 0, 1, 2, 3, 4}
def A : Set ℤ := {2, 3}
def AUnionB : Set ℤ := {1, 2, 3, 4}

theorem intersection_complement_theorem :
  ∃ B : Set ℤ, A ∪ B = AUnionB ∧ B ∩ (M \ A) = {1, 4} := by
  sorry

end intersection_complement_theorem_l3557_355796


namespace hyperbola_sum_l3557_355741

/-- Represents a hyperbola with center (h, k), focus (h + c, k), and vertex (h - a, k) -/
structure Hyperbola where
  h : ℝ
  k : ℝ
  a : ℝ
  c : ℝ
  vertex_x : ℝ
  focus_x : ℝ
  h_pos : 0 < a
  h_c_gt_a : c > a
  h_vertex : vertex_x = h - a
  h_focus : focus_x = h + c

/-- The theorem stating the sum of h, k, a, and b for the given hyperbola -/
theorem hyperbola_sum (H : Hyperbola) (h_center : H.h = 1 ∧ H.k = -1)
    (h_vertex : H.vertex_x = -2) (h_focus : H.focus_x = 1 + Real.sqrt 41) :
    H.h + H.k + H.a + Real.sqrt (H.c^2 - H.a^2) = 3 + 4 * Real.sqrt 2 := by
  sorry

end hyperbola_sum_l3557_355741


namespace smaller_circle_radius_l3557_355708

theorem smaller_circle_radius (R : ℝ) (h : R = 12) : 
  ∃ (r : ℝ), r = 3 * Real.sqrt 3 ∧
  r > 0 ∧
  r < R ∧
  (∃ (A B C D E F G : ℝ × ℝ),
    -- A is the center of the left circle
    -- B is on the right circle
    -- C is the center of the right circle
    -- D is the center of the smaller circle
    -- E, F, G are points of tangency

    -- The centers are R apart
    Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = R ∧

    -- AB is a diameter of the right circle
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2*R ∧

    -- D is r away from E, F, and G
    Real.sqrt ((D.1 - E.1)^2 + (D.2 - E.2)^2) = r ∧
    Real.sqrt ((D.1 - F.1)^2 + (D.2 - F.2)^2) = r ∧
    Real.sqrt ((D.1 - G.1)^2 + (D.2 - G.2)^2) = r ∧

    -- A is R+r away from F
    Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = R + r ∧

    -- C is R-r away from G
    Real.sqrt ((C.1 - G.1)^2 + (C.2 - G.2)^2) = R - r ∧

    -- E is on AB
    (E.2 - A.2) / (E.1 - A.1) = (B.2 - A.2) / (B.1 - A.1)
  ) := by
sorry

end smaller_circle_radius_l3557_355708


namespace min_value_fraction_l3557_355767

theorem min_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z)^3 / ((x + y)^3 * (y + z)^3) ≥ 27/8 :=
by sorry

end min_value_fraction_l3557_355767


namespace translation_theorem_l3557_355751

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point horizontally -/
def translate_right (p : Point) (units : ℝ) : Point :=
  { x := p.x + units, y := p.y }

theorem translation_theorem :
  ∀ (A : Point),
    translate_right A 2 = Point.mk 3 2 →
    A = Point.mk 1 2 := by
  sorry

end translation_theorem_l3557_355751


namespace supply_duration_l3557_355725

/-- Represents the number of pills in one supply -/
def supply_size : ℕ := 90

/-- Represents the number of days between taking each pill -/
def days_per_pill : ℕ := 3

/-- Represents the approximate number of days in a month -/
def days_per_month : ℕ := 30

/-- Proves that a supply of pills lasts approximately 9 months -/
theorem supply_duration : 
  (supply_size * days_per_pill) / days_per_month = 9 := by
  sorry

end supply_duration_l3557_355725


namespace craigs_commission_problem_l3557_355786

/-- Craig's appliance sales commission problem -/
theorem craigs_commission_problem 
  (fixed_amount : ℝ) 
  (num_appliances : ℕ) 
  (total_selling_price : ℝ) 
  (total_commission : ℝ) 
  (h1 : num_appliances = 6)
  (h2 : total_selling_price = 3620)
  (h3 : total_commission = 662)
  (h4 : total_commission = num_appliances * fixed_amount + 0.1 * total_selling_price) :
  fixed_amount = 50 := by
sorry

end craigs_commission_problem_l3557_355786


namespace dish_initial_temp_l3557_355747

/-- The initial temperature of a dish given its heating rate and time to reach a final temperature --/
def initial_temperature (final_temp : ℝ) (heating_rate : ℝ) (heating_time : ℝ) : ℝ :=
  final_temp - heating_rate * heating_time

/-- Theorem stating that the initial temperature of the dish is 20 degrees --/
theorem dish_initial_temp : initial_temperature 100 5 16 = 20 := by
  sorry

end dish_initial_temp_l3557_355747


namespace melany_money_theorem_l3557_355779

/-- The amount of money Melany initially had to fence a square field --/
def melany_initial_money (field_size : ℕ) (wire_cost_per_foot : ℕ) (unfenced_length : ℕ) : ℕ :=
  (field_size - unfenced_length) * wire_cost_per_foot

/-- Theorem stating that Melany's initial money was $120,000 --/
theorem melany_money_theorem (field_size : ℕ) (wire_cost_per_foot : ℕ) (unfenced_length : ℕ) 
  (h1 : field_size = 5000)
  (h2 : wire_cost_per_foot = 30)
  (h3 : unfenced_length = 1000) :
  melany_initial_money field_size wire_cost_per_foot unfenced_length = 120000 := by
  sorry

end melany_money_theorem_l3557_355779


namespace ring_width_equals_disk_radius_l3557_355760

/-- A flat ring formed by two concentric circles with seven equal touching disks inserted -/
structure FlatRing where
  R₁ : ℝ  -- Radius of the outer circle
  R₂ : ℝ  -- Radius of the inner circle
  r : ℝ   -- Radius of each disk
  h₁ : R₁ > R₂  -- Outer radius is greater than inner radius
  h₂ : R₂ = 3 * r  -- Inner radius is 3 times the disk radius
  h₃ : 7 * π * r^2 = π * (R₁^2 - R₂^2)  -- Area of ring equals sum of disk areas

/-- The width of the ring is equal to the radius of one disk -/
theorem ring_width_equals_disk_radius (ring : FlatRing) : ring.R₁ - ring.R₂ = ring.r := by
  sorry


end ring_width_equals_disk_radius_l3557_355760


namespace find_k_l3557_355757

theorem find_k : ∃ k : ℝ, (64 / k = 4) ∧ (k = 16) := by
  sorry

end find_k_l3557_355757


namespace total_jellybeans_l3557_355716

def dozen : ℕ := 12

def caleb_jellybeans : ℕ := 3 * dozen

def sophie_jellybeans : ℕ := caleb_jellybeans / 2

theorem total_jellybeans : caleb_jellybeans + sophie_jellybeans = 54 := by
  sorry

end total_jellybeans_l3557_355716


namespace line_circle_relationship_l3557_355700

theorem line_circle_relationship (m : ℝ) :
  ∃ (x y : ℝ), (m * x + y - m - 1 = 0 ∧ x^2 + y^2 = 2) ∨
  ∃ (x y : ℝ), (m * x + y - m - 1 = 0 ∧ x^2 + y^2 = 2 ∧
    ∀ (x' y' : ℝ), m * x' + y' - m - 1 = 0 → x'^2 + y'^2 ≥ 2) :=
by sorry

end line_circle_relationship_l3557_355700


namespace evaluate_power_l3557_355755

theorem evaluate_power : (81 : ℝ) ^ (5/4 : ℝ) = 243 := by sorry

end evaluate_power_l3557_355755


namespace quadratic_always_has_two_roots_find_m_value_l3557_355744

/-- Given quadratic equation x^2 - (2m+1)x + m - 2 = 0 -/
def quadratic_equation (m x : ℝ) : Prop :=
  x^2 - (2*m+1)*x + m - 2 = 0

theorem quadratic_always_has_two_roots (m : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m x₁ ∧ quadratic_equation m x₂ :=
sorry

theorem find_m_value :
  ∃ m : ℝ, m = 6/5 ∧
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    quadratic_equation m x₁ ∧ 
    quadratic_equation m x₂ ∧
    x₁ + x₂ + 3*x₁*x₂ = 1) :=
sorry

end quadratic_always_has_two_roots_find_m_value_l3557_355744


namespace turnover_equation_l3557_355730

-- Define the monthly average growth rate
variable (x : ℝ)

-- Define the initial turnover in January (in units of 10,000 yuan)
def initial_turnover : ℝ := 200

-- Define the total turnover in the first quarter (in units of 10,000 yuan)
def total_turnover : ℝ := 1000

-- Theorem statement
theorem turnover_equation :
  initial_turnover + initial_turnover * (1 + x) + initial_turnover * (1 + x)^2 = total_turnover := by
  sorry

end turnover_equation_l3557_355730


namespace carpenter_square_problem_l3557_355739

theorem carpenter_square_problem (s : ℝ) :
  (s^2 - 4 * (0.09 * s^2) = 256) → s = 20 := by
  sorry

end carpenter_square_problem_l3557_355739
