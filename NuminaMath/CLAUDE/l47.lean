import Mathlib

namespace max_value_theorem_l47_4730

def feasible_region (x y : ℝ) : Prop :=
  2 * x + y ≤ 4 ∧ x ≤ y ∧ x ≥ 1/2

def objective_function (x y : ℝ) : ℝ :=
  2 * x - y

theorem max_value_theorem :
  ∃ (x y : ℝ), feasible_region x y ∧
  ∀ (x' y' : ℝ), feasible_region x' y' →
  objective_function x y ≥ objective_function x' y' ∧
  objective_function x y = 4/3 :=
sorry

end max_value_theorem_l47_4730


namespace odd_function_value_at_negative_one_l47_4779

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_value_at_negative_one
  (f : ℝ → ℝ)
  (h_odd : IsOdd f)
  (h_nonneg : ∀ x ≥ 0, f x = 2^x + 2*x + f 0)
  : f (-1) = -3 :=
by sorry

end odd_function_value_at_negative_one_l47_4779


namespace jack_morning_emails_l47_4764

theorem jack_morning_emails (afternoon_emails : ℕ) (morning_afternoon_difference : ℕ) : 
  afternoon_emails = 2 → 
  morning_afternoon_difference = 4 →
  afternoon_emails + morning_afternoon_difference = 6 :=
by
  sorry

end jack_morning_emails_l47_4764


namespace mary_remaining_money_l47_4737

def remaining_money (p : ℝ) : ℝ :=
  let drink_cost := p
  let medium_pizza_cost := 3 * p
  let large_pizza_cost := 4 * p
  let total_cost := 5 * drink_cost + medium_pizza_cost + large_pizza_cost
  30 - total_cost

theorem mary_remaining_money (p : ℝ) :
  remaining_money p = 30 - 12 * p :=
by sorry

end mary_remaining_money_l47_4737


namespace remaining_cards_l47_4767

def initial_cards : ℕ := 87
def sam_cards : ℕ := 8
def alex_cards : ℕ := 13

theorem remaining_cards : initial_cards - (sam_cards + alex_cards) = 66 := by
  sorry

end remaining_cards_l47_4767


namespace x₀_value_l47_4726

noncomputable section

variables (a b : ℝ) (x₀ : ℝ)

def f (x : ℝ) := a * x^2 + b

theorem x₀_value (ha : a ≠ 0) (hx₀ : x₀ > 0) 
  (h_integral : ∫ x in (0)..(2), f a b x = 2 * f a b x₀) : 
  x₀ = 2 * Real.sqrt 3 / 3 := by
  sorry

end

end x₀_value_l47_4726


namespace paco_cookies_l47_4777

/-- Calculates the number of cookies Paco bought given the initial, eaten, and final cookie counts. -/
def cookies_bought (initial : ℕ) (eaten : ℕ) (final : ℕ) : ℕ :=
  final - (initial - eaten)

/-- Proves that Paco bought 37 cookies given the problem conditions. -/
theorem paco_cookies : cookies_bought 40 2 75 = 37 := by
  sorry

end paco_cookies_l47_4777


namespace smallest_three_digit_multiple_plus_one_l47_4714

theorem smallest_three_digit_multiple_plus_one : ∃ (n : ℕ), 
  (n = 421) ∧ 
  (100 ≤ n) ∧ 
  (n < 1000) ∧ 
  (∃ (k : ℕ), n = k * 3 + 1) ∧
  (∃ (k : ℕ), n = k * 4 + 1) ∧
  (∃ (k : ℕ), n = k * 5 + 1) ∧
  (∃ (k : ℕ), n = k * 6 + 1) ∧
  (∃ (k : ℕ), n = k * 7 + 1) ∧
  (∀ (m : ℕ), 
    (100 ≤ m) ∧ 
    (m < n) → 
    ¬((∃ (k : ℕ), m = k * 3 + 1) ∧
      (∃ (k : ℕ), m = k * 4 + 1) ∧
      (∃ (k : ℕ), m = k * 5 + 1) ∧
      (∃ (k : ℕ), m = k * 6 + 1) ∧
      (∃ (k : ℕ), m = k * 7 + 1))) :=
by sorry

end smallest_three_digit_multiple_plus_one_l47_4714


namespace bruce_age_multiple_l47_4768

/-- The number of years it takes for a person to become a multiple of another person's age -/
def years_to_multiple (initial_age_older : ℕ) (initial_age_younger : ℕ) (multiple : ℕ) : ℕ :=
  let x := (multiple * initial_age_younger - initial_age_older) / (multiple - 1)
  x

/-- Theorem stating that it takes 6 years for a 36-year-old to become 3 times as old as an 8-year-old -/
theorem bruce_age_multiple : years_to_multiple 36 8 3 = 6 := by
  sorry

end bruce_age_multiple_l47_4768


namespace women_science_majors_percentage_l47_4770

theorem women_science_majors_percentage
  (non_science_percentage : Real)
  (men_percentage : Real)
  (men_science_percentage : Real)
  (h1 : non_science_percentage = 0.6)
  (h2 : men_percentage = 0.4)
  (h3 : men_science_percentage = 0.5500000000000001) :
  let women_percentage := 1 - men_percentage
  let total_science_percentage := 1 - non_science_percentage
  let men_science_total_percentage := men_percentage * men_science_percentage
  let women_science_total_percentage := total_science_percentage - men_science_total_percentage
  women_science_total_percentage / women_percentage = 0.29999999999999993 :=
by sorry

end women_science_majors_percentage_l47_4770


namespace triangle_classification_l47_4720

/-- Checks if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem triangle_classification :
  ¬(is_right_triangle 1.5 2 3) ∧
  (is_right_triangle 7 24 25) ∧
  (is_right_triangle 3 4 5) ∧
  (is_right_triangle 9 12 15) :=
by sorry

end triangle_classification_l47_4720


namespace bananas_per_box_l47_4778

/-- Given 40 bananas and 8 boxes, prove that 5 bananas must go in each box. -/
theorem bananas_per_box (total_bananas : ℕ) (num_boxes : ℕ) 
  (h1 : total_bananas = 40) (h2 : num_boxes = 8) : 
  total_bananas / num_boxes = 5 := by
  sorry

#check bananas_per_box

end bananas_per_box_l47_4778


namespace product_mod_five_l47_4789

theorem product_mod_five : (2023 * 2024 * 2025 * 2026) % 5 = 0 := by
  sorry

end product_mod_five_l47_4789


namespace product_of_fractions_l47_4751

theorem product_of_fractions : 
  (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * (1 + 1/5) * (1 + 1/6) * (1 + 1/7) = 8 := by
  sorry

end product_of_fractions_l47_4751


namespace solution_set_f_geq_2_range_of_a_for_full_solution_set_l47_4791

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Theorem for part I
theorem solution_set_f_geq_2 :
  {x : ℝ | f x ≥ 2} = {x : ℝ | x ≥ 3/2} :=
sorry

-- Theorem for part II
theorem range_of_a_for_full_solution_set :
  {a : ℝ | ∀ x, f x ≤ |a - 2|} = {a : ℝ | a ≤ -1 ∨ a ≥ 5} :=
sorry

end solution_set_f_geq_2_range_of_a_for_full_solution_set_l47_4791


namespace geometric_arithmetic_geometric_sequence_l47_4746

theorem geometric_arithmetic_geometric_sequence 
  (a b c : ℝ) 
  (h1 : b ^ 2 = a * c)  -- geometric progression condition
  (h2 : b + 2 = (a + c) / 2)  -- arithmetic progression condition
  (h3 : (b + 2) ^ 2 = a * (c + 16))  -- second geometric progression condition
  : (a = 1 ∧ b = 3 ∧ c = 9) ∨ (a = 1/9 ∧ b = -5/9 ∧ c = 25/9) := by
  sorry

end geometric_arithmetic_geometric_sequence_l47_4746


namespace m_range_l47_4766

/-- A function f satisfying the given conditions -/
def f_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x, x ∈ Set.Icc (-1) 1 → f (-x) = -f x) ∧
  (∀ a b, a ∈ Set.Ioo (-1) 0 → b ∈ Set.Ioo (-1) 0 → a ≠ b → (f a - f b) / (a - b) > 0)

theorem m_range (f : ℝ → ℝ) (m : ℝ) (hf : f_conditions f) (h : f (m + 1) > f (2 * m)) :
  -1/2 ≤ m ∧ m ≤ 0 := by
  sorry

end m_range_l47_4766


namespace triangle_reconstruction_theorem_l47_4761

-- Define the basic structures
structure Point := (x y : ℝ)

structure Triangle :=
(A B C : Point)

-- Define the given points
variable (D E F : Point)

-- Define the properties of the given points
def is_altitude_median_intersection (D : Point) (T : Triangle) : Prop := sorry

def is_altitude_bisector_intersection (E : Point) (T : Triangle) : Prop := sorry

def is_median_bisector_intersection (F : Point) (T : Triangle) : Prop := sorry

-- State the theorem
theorem triangle_reconstruction_theorem 
  (hD : ∃ T : Triangle, is_altitude_median_intersection D T)
  (hE : ∃ T : Triangle, is_altitude_bisector_intersection E T)
  (hF : ∃ T : Triangle, is_median_bisector_intersection F T) :
  ∃! T : Triangle, 
    is_altitude_median_intersection D T ∧ 
    is_altitude_bisector_intersection E T ∧ 
    is_median_bisector_intersection F T :=
sorry

end triangle_reconstruction_theorem_l47_4761


namespace lcm_factor_not_unique_l47_4701

/-- Given two positive integers with HCF 52 and larger number 624, 
    the other factor of their LCM cannot be uniquely determined. -/
theorem lcm_factor_not_unique (A B : ℕ+) : 
  (Nat.gcd A B = 52) → 
  (max A B = 624) → 
  ∃ (y : ℕ+), y ≠ 1 ∧ 
    ∃ (lcm : ℕ+), lcm = Nat.lcm A B ∧ lcm = 624 * y :=
by sorry

end lcm_factor_not_unique_l47_4701


namespace pond_length_l47_4775

/-- Given a rectangular field and a square pond, prove the length of the pond -/
theorem pond_length (field_length : ℝ) (field_width : ℝ) (pond_length : ℝ) : 
  field_length = 16 →
  field_length = 2 * field_width →
  pond_length ^ 2 = (field_length * field_width) / 2 →
  pond_length = 8 := by
sorry

end pond_length_l47_4775


namespace rachel_toys_l47_4702

theorem rachel_toys (jason_toys : ℕ) (john_toys : ℕ) (rachel_toys : ℕ)
  (h1 : jason_toys = 21)
  (h2 : jason_toys = 3 * john_toys)
  (h3 : john_toys = rachel_toys + 6) :
  rachel_toys = 1 := by
  sorry

end rachel_toys_l47_4702


namespace elena_car_rental_cost_l47_4782

/-- Calculates the total cost of a car rental given the daily rate, mileage rate, number of days, and miles driven. -/
def car_rental_cost (daily_rate : ℚ) (mileage_rate : ℚ) (days : ℕ) (miles : ℕ) : ℚ :=
  daily_rate * days + mileage_rate * miles

/-- Proves that Elena's car rental cost is $215 given the specified conditions. -/
theorem elena_car_rental_cost :
  car_rental_cost 30 0.25 3 500 = 215 := by
  sorry

end elena_car_rental_cost_l47_4782


namespace equilateral_triangles_area_sum_l47_4723

/-- Given an isosceles right triangle with leg length 36 units, the sum of the areas
    of an infinite series of equilateral triangles drawn on one leg (with their third
    vertices on the hypotenuse) is equal to half the area of the original right triangle. -/
theorem equilateral_triangles_area_sum (leg_length : ℝ) (h : leg_length = 36) :
  let right_triangle_area := (1 / 2) * leg_length * leg_length
  let equilateral_triangles_area_sum := right_triangle_area / 2
  equilateral_triangles_area_sum = 324 := by
  sorry

end equilateral_triangles_area_sum_l47_4723


namespace alternative_basis_l47_4706

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Given that {a, b, c} is a basis in space, prove that {c, a+b, a-b} is also a basis. -/
theorem alternative_basis
  (a b c : V)
  (h : LinearIndependent ℝ ![a, b, c])
  (hspan : Submodule.span ℝ {a, b, c} = ⊤) :
  LinearIndependent ℝ ![c, a+b, a-b] ∧
  Submodule.span ℝ {c, a+b, a-b} = ⊤ :=
sorry

end alternative_basis_l47_4706


namespace angle_relationship_l47_4785

theorem angle_relationship (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.sin α = (1/2) * Real.sin (α + β)) : α < β := by
  sorry

end angle_relationship_l47_4785


namespace color_film_fraction_l47_4740

theorem color_film_fraction (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  let total_bw := 20 * x
  let total_color := 8 * y
  let selected_bw := y / 5
  let selected_color := total_color
  let total_selected := selected_bw + selected_color
  selected_color / total_selected = 40 / 41 := by
sorry

end color_film_fraction_l47_4740


namespace acid_dilution_l47_4773

/-- Proves that adding 80/3 ounces of pure water to 40 ounces of a 25% acid solution
    results in a 15% acid solution. -/
theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
    (added_water : ℝ) (final_concentration : ℝ) : 
    initial_volume = 40 →
    initial_concentration = 0.25 →
    added_water = 80 / 3 →
    final_concentration = 0.15 →
    (initial_volume * initial_concentration) / (initial_volume + added_water) = final_concentration :=
by
  sorry


end acid_dilution_l47_4773


namespace number_of_children_l47_4792

theorem number_of_children (pencils_per_child : ℕ) (total_pencils : ℕ) (h1 : pencils_per_child = 4) (h2 : total_pencils = 32) :
  total_pencils / pencils_per_child = 8 := by
  sorry

end number_of_children_l47_4792


namespace min_quadratic_function_l47_4787

theorem min_quadratic_function :
  (∀ x : ℝ, x^2 - 2*x ≥ -1) ∧ (∃ x : ℝ, x^2 - 2*x = -1) := by
  sorry

end min_quadratic_function_l47_4787


namespace two_distinct_negative_roots_l47_4716

/-- The polynomial function for which we're finding roots -/
def f (p : ℝ) (x : ℝ) : ℝ := x^4 + 4*p*x^3 + 2*x^2 + 4*p*x + 1

/-- A root of the polynomial is a real number x such that f p x = 0 -/
def is_root (p : ℝ) (x : ℝ) : Prop := f p x = 0

/-- A function to represent that a real number is negative -/
def is_negative (x : ℝ) : Prop := x < 0

/-- The main theorem stating that for p > 1, there are at least two distinct negative real roots -/
theorem two_distinct_negative_roots (p : ℝ) (hp : p > 1) :
  ∃ (x y : ℝ), x ≠ y ∧ is_negative x ∧ is_negative y ∧ is_root p x ∧ is_root p y :=
sorry

end two_distinct_negative_roots_l47_4716


namespace sniper_B_wins_l47_4727

/-- Represents a sniper with probabilities of scoring 1, 2, and 3 points -/
structure Sniper where
  prob1 : ℝ
  prob2 : ℝ
  prob3 : ℝ

/-- Calculate the expected score for a sniper -/
def expectedScore (s : Sniper) : ℝ := 1 * s.prob1 + 2 * s.prob2 + 3 * s.prob3

/-- Sniper A with given probabilities -/
def sniperA : Sniper := { prob1 := 0.4, prob2 := 0.1, prob3 := 0.5 }

/-- Sniper B with given probabilities -/
def sniperB : Sniper := { prob1 := 0.1, prob2 := 0.6, prob3 := 0.3 }

/-- Theorem stating that Sniper B has a higher expected score than Sniper A -/
theorem sniper_B_wins : expectedScore sniperB > expectedScore sniperA := by
  sorry

end sniper_B_wins_l47_4727


namespace subset_implies_membership_l47_4721

theorem subset_implies_membership (P Q : Set α) 
  (h_nonempty_P : P.Nonempty) (h_nonempty_Q : Q.Nonempty) (h_subset : P ⊆ Q) : 
  ∀ x ∈ P, x ∈ Q := by
  sorry

end subset_implies_membership_l47_4721


namespace egypt_tour_promotion_l47_4724

theorem egypt_tour_promotion (total_tourists : ℕ) (free_tourists : ℕ) : 
  (13 : ℕ) + 4 * free_tourists = total_tourists ∧ 
  free_tourists + (100 : ℕ) = total_tourists →
  free_tourists = 29 := by
sorry

end egypt_tour_promotion_l47_4724


namespace cards_traded_is_35_l47_4769

/-- The total number of cards traded between Padma and Robert -/
def total_cards_traded (padma_initial : ℕ) (robert_initial : ℕ) 
  (padma_first_trade : ℕ) (robert_first_trade : ℕ) 
  (padma_second_trade : ℕ) (robert_second_trade : ℕ) : ℕ :=
  padma_first_trade + robert_first_trade + padma_second_trade + robert_second_trade

/-- Theorem stating the total number of cards traded is 35 -/
theorem cards_traded_is_35 : 
  total_cards_traded 75 88 2 10 15 8 = 35 := by
  sorry


end cards_traded_is_35_l47_4769


namespace largest_2010_digit_prime_squared_minus_one_div_24_l47_4719

/-- The largest prime number with 2010 digits -/
def p : ℕ := sorry

/-- p is prime -/
axiom p_prime : Nat.Prime p

/-- p has 2010 digits -/
axiom p_digits : 10^2009 ≤ p ∧ p < 10^2010

/-- p is the largest prime with 2010 digits -/
axiom p_largest : ∀ q, Nat.Prime q → 10^2009 ≤ q → q < 10^2010 → q ≤ p

theorem largest_2010_digit_prime_squared_minus_one_div_24 : 
  24 ∣ (p^2 - 1) := by sorry

end largest_2010_digit_prime_squared_minus_one_div_24_l47_4719


namespace farmer_extra_days_l47_4700

/-- The number of extra days a farmer needs to work given initial and actual ploughing rates, total area, and remaining area. -/
theorem farmer_extra_days (initial_rate actual_rate total_area remaining_area : ℕ) : 
  initial_rate = 90 →
  actual_rate = 85 →
  total_area = 3780 →
  remaining_area = 40 →
  (total_area - remaining_area) % actual_rate = 0 →
  (remaining_area + actual_rate - 1) / actual_rate = 1 := by
  sorry

end farmer_extra_days_l47_4700


namespace five_to_five_sum_equals_five_to_six_l47_4780

theorem five_to_five_sum_equals_five_to_six : 
  5^5 + 5^5 + 5^5 + 5^5 + 5^5 = 5^6 := by
  sorry

end five_to_five_sum_equals_five_to_six_l47_4780


namespace hamiltonian_cycle_with_at_most_one_color_change_l47_4760

/-- A complete graph with n vertices where each edge is colored either red or blue -/
structure ColoredCompleteGraph (n : ℕ) where
  vertices : Fin n → Type
  edge_color : ∀ (i j : Fin n), i ≠ j → Bool

/-- A Hamiltonian cycle in the graph -/
def HamiltonianCycle (n : ℕ) (G : ColoredCompleteGraph n) :=
  { cycle : List (Fin n) // cycle.length = n ∧ cycle.Nodup }

/-- The number of color changes in a Hamiltonian cycle -/
def ColorChanges (n : ℕ) (G : ColoredCompleteGraph n) (cycle : HamiltonianCycle n G) : ℕ :=
  sorry

/-- Theorem: There exists a Hamiltonian cycle with at most one color change -/
theorem hamiltonian_cycle_with_at_most_one_color_change (n : ℕ) (G : ColoredCompleteGraph n) :
  ∃ (cycle : HamiltonianCycle n G), ColorChanges n G cycle ≤ 1 :=
sorry

end hamiltonian_cycle_with_at_most_one_color_change_l47_4760


namespace prob_four_friends_same_group_l47_4705

/-- Represents the total number of students -/
def total_students : ℕ := 800

/-- Represents the number of lunch groups -/
def num_groups : ℕ := 4

/-- Represents the size of each lunch group -/
def group_size : ℕ := total_students / num_groups

/-- Represents the probability of a student being assigned to a specific group -/
def prob_assigned_to_group : ℚ := 1 / num_groups

/-- Represents the four friends -/
inductive Friend : Type
  | Al | Bob | Carol | Dan

/-- 
Theorem: The probability that four specific students (friends) are assigned 
to the same lunch group in a random assignment is 1/64.
-/
theorem prob_four_friends_same_group : 
  (prob_assigned_to_group ^ 3 : ℚ) = 1 / 64 := by sorry

end prob_four_friends_same_group_l47_4705


namespace jesus_squares_count_l47_4763

/-- The number of squares Pedro has -/
def pedro_squares : ℕ := 200

/-- The number of squares Linden has -/
def linden_squares : ℕ := 75

/-- The number of extra squares Pedro has compared to Jesus and Linden combined -/
def extra_squares : ℕ := 65

/-- The number of squares Jesus has -/
def jesus_squares : ℕ := pedro_squares - linden_squares - extra_squares

theorem jesus_squares_count : jesus_squares = 60 := by sorry

end jesus_squares_count_l47_4763


namespace quadratic_solution_square_l47_4734

theorem quadratic_solution_square (y : ℝ) (h : 3 * y^2 + 2 = 7 * y + 15) : (6 * y - 5)^2 = 269 := by
  sorry

end quadratic_solution_square_l47_4734


namespace multiples_properties_l47_4783

theorem multiples_properties (c d : ℤ) 
  (hc : ∃ k : ℤ, c = 4 * k) 
  (hd : ∃ m : ℤ, d = 8 * m) : 
  (∃ n : ℤ, d = 4 * n) ∧ 
  (∃ p : ℤ, c - d = 4 * p) ∧ 
  (∃ q : ℤ, c - d = 2 * q) :=
by sorry

end multiples_properties_l47_4783


namespace equal_circles_in_quadrilateral_l47_4762

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents a convex quadrilateral with four circles inside -/
structure ConvexQuadrilateral where
  circle_a : Circle
  circle_b : Circle
  circle_c : Circle
  circle_d : Circle
  is_convex : Bool
  circles_touch_sides : Bool
  circles_touch_each_other : Bool
  has_inscribed_circle : Bool

/-- 
Given a convex quadrilateral with four circles inside, each touching two adjacent sides 
and two other circles externally, and given that a circle can be inscribed in the quadrilateral, 
at least two of the four circles have equal radii.
-/
theorem equal_circles_in_quadrilateral (q : ConvexQuadrilateral) 
  (h1 : q.is_convex = true) 
  (h2 : q.circles_touch_sides = true)
  (h3 : q.circles_touch_each_other = true)
  (h4 : q.has_inscribed_circle = true) : 
  (q.circle_a.radius = q.circle_b.radius) ∨ 
  (q.circle_a.radius = q.circle_c.radius) ∨ 
  (q.circle_a.radius = q.circle_d.radius) ∨ 
  (q.circle_b.radius = q.circle_c.radius) ∨ 
  (q.circle_b.radius = q.circle_d.radius) ∨ 
  (q.circle_c.radius = q.circle_d.radius) :=
by
  sorry

end equal_circles_in_quadrilateral_l47_4762


namespace product_remainder_zero_l47_4794

theorem product_remainder_zero : (2005 * 2006 * 2007 * 2008 * 2009) % 17 = 0 := by
  sorry

end product_remainder_zero_l47_4794


namespace geometric_sequence_common_ratio_l47_4756

/-- A geometric sequence with common ratio r satisfying a_n * a_(n+1) = 16^n has r = 4 --/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (r : ℝ) 
  (h_geom : ∀ n, a (n + 1) = r * a n) 
  (h_prod : ∀ n, a n * a (n + 1) = 16^n) : 
  r = 4 := by
sorry

end geometric_sequence_common_ratio_l47_4756


namespace perimeter_of_specific_quadrilateral_l47_4776

structure Quadrilateral where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  h_positive : 0 < AB ∧ 0 < BC ∧ 0 < CD ∧ 0 < DA

def perimeter (q : Quadrilateral) : ℝ :=
  q.AB + q.BC + q.CD + q.DA

theorem perimeter_of_specific_quadrilateral :
  ∃ (q : Quadrilateral), 
    q.DA < q.BC ∧
    q.DA = 4 ∧
    q.AB = 5 ∧
    q.BC = 10 ∧
    q.CD = 7 ∧
    perimeter q = 26 := by
  sorry

end perimeter_of_specific_quadrilateral_l47_4776


namespace greatest_root_of_f_l47_4742

def f (x : ℝ) : ℝ := 16 * x^6 - 15 * x^4 + 4 * x^2 - 1

theorem greatest_root_of_f :
  ∃ (r : ℝ), f r = 0 ∧ r = 1 ∧ ∀ (x : ℝ), f x = 0 → x ≤ r :=
sorry

end greatest_root_of_f_l47_4742


namespace invalid_transformation_l47_4786

theorem invalid_transformation (x y m : ℝ) : 
  ¬(∀ (x y m : ℝ), x = y → x / m = y / m) :=
sorry

end invalid_transformation_l47_4786


namespace hundredth_figure_count_l47_4771

/-- Represents the number of nonoverlapping unit triangles in the nth figure of the pattern. -/
def triangle_count (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

/-- The first four terms of the sequence match the given pattern. -/
axiom first_four_correct : 
  triangle_count 0 = 1 ∧ 
  triangle_count 1 = 7 ∧ 
  triangle_count 2 = 19 ∧ 
  triangle_count 3 = 37

/-- The 100th figure contains 30301 nonoverlapping unit triangles. -/
theorem hundredth_figure_count : triangle_count 100 = 30301 := by
  sorry

end hundredth_figure_count_l47_4771


namespace ray_nickels_left_l47_4758

-- Define the value of a nickel in cents
def nickel_value : ℕ := 5

-- Define Ray's initial amount in cents
def initial_amount : ℕ := 95

-- Define the amount given to Peter in cents
def amount_to_peter : ℕ := 25

-- Theorem stating that Ray will have 4 nickels left
theorem ray_nickels_left : 
  let amount_to_randi := 2 * amount_to_peter
  let total_given := amount_to_peter + amount_to_randi
  let remaining_cents := initial_amount - total_given
  remaining_cents / nickel_value = 4 := by
sorry

end ray_nickels_left_l47_4758


namespace frank_final_position_l47_4793

/-- Calculates Frank's final position after a series of dance moves --/
def frankPosition (initialBackSteps : ℤ) (firstForwardSteps : ℤ) (secondBackSteps : ℤ) : ℤ :=
  -initialBackSteps + firstForwardSteps - secondBackSteps + 2 * secondBackSteps

/-- Proves that Frank ends up 7 steps forward from his original starting point --/
theorem frank_final_position :
  frankPosition 5 10 2 = 7 :=
by sorry

end frank_final_position_l47_4793


namespace apple_profit_calculation_l47_4738

/-- Calculates the total percentage profit for a shopkeeper selling apples -/
theorem apple_profit_calculation (total_apples : ℝ) (first_portion : ℝ) (second_portion : ℝ)
  (first_profit_rate : ℝ) (second_profit_rate : ℝ) 
  (h1 : total_apples = 100)
  (h2 : first_portion = 0.5 * total_apples)
  (h3 : second_portion = 0.5 * total_apples)
  (h4 : first_profit_rate = 0.25)
  (h5 : second_profit_rate = 0.3) :
  (((first_portion * (1 + first_profit_rate) + second_portion * (1 + second_profit_rate)) - total_apples) / total_apples) * 100 = 27.5 := by
  sorry

end apple_profit_calculation_l47_4738


namespace min_value_problem_l47_4733

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, a * x + b * y = 1 → x^2 + y^2 - 2*x - 2*y - 2 = 0 → x = 1 ∧ y = 1) →
  (∀ c d : ℝ, c > 0 → d > 0 → c * 1 + d * 1 = 1 → 1/c + 2/d ≥ 1/a + 2/b) →
  1/a + 2/b = 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_problem_l47_4733


namespace gcd_765432_654321_l47_4732

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 3 := by sorry

end gcd_765432_654321_l47_4732


namespace trajectory_of_P_l47_4784

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the point M on the ellipse C
def M : ℝ × ℝ → Prop := λ p => C p.1 p.2

-- Define the point N on the x-axis
def N (m : ℝ × ℝ) : ℝ × ℝ := (m.1, 0)

-- Define the vector NP
def NP (n p : ℝ × ℝ) : ℝ × ℝ := (p.1 - n.1, p.2 - n.2)

-- Define the vector NM
def NM (n m : ℝ × ℝ) : ℝ × ℝ := (m.1 - n.1, m.2 - n.2)

-- State the theorem
theorem trajectory_of_P (x y : ℝ) :
  (∃ m : ℝ × ℝ, M m ∧ 
   let n := N m
   NP n (x, y) = Real.sqrt 2 • NM n m) →
  x^2 + y^2 = 2 :=
sorry

end trajectory_of_P_l47_4784


namespace constant_order_magnitude_l47_4748

theorem constant_order_magnitude (k : ℕ) (h : k > 4) :
  k + 2 < 2 * k ∧ 2 * k < k^2 ∧ k^2 < 2^k := by
  sorry

end constant_order_magnitude_l47_4748


namespace cistern_fill_time_theorem_l47_4750

/-- Represents the rate at which a pipe can fill or empty a cistern -/
structure PipeRate where
  fill : ℚ  -- Fraction of cistern filled or emptied
  time : ℚ  -- Time taken in minutes
  deriving Repr

/-- Calculates the rate of filling or emptying per minute -/
def rate_per_minute (p : PipeRate) : ℚ := p.fill / p.time

/-- Represents the problem of filling a cistern with multiple pipes -/
structure CisternProblem where
  pipe_a : PipeRate
  pipe_b : PipeRate
  pipe_c : PipeRate
  target_fill : ℚ
  deriving Repr

/-- Calculates the time required to fill the target amount of the cistern -/
def fill_time (problem : CisternProblem) : ℚ :=
  let combined_rate := rate_per_minute problem.pipe_a + rate_per_minute problem.pipe_b - rate_per_minute problem.pipe_c
  problem.target_fill / combined_rate

/-- The main theorem stating the time required to fill half the cistern -/
theorem cistern_fill_time_theorem (problem : CisternProblem) 
  (h1 : problem.pipe_a = ⟨1/2, 10⟩)
  (h2 : problem.pipe_b = ⟨2/3, 15⟩)
  (h3 : problem.pipe_c = ⟨1/4, 20⟩)
  (h4 : problem.target_fill = 1/2) :
  fill_time problem = 720/118 := by
  sorry

end cistern_fill_time_theorem_l47_4750


namespace tangent_line_at_point_l47_4790

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 3

-- Define the point of tangency
def point : ℝ × ℝ := (2, 2)

-- Define the two possible tangent lines
def line1 (x : ℝ) : ℝ := 2
def line2 (x : ℝ) : ℝ := 9*x - 16

theorem tangent_line_at_point :
  (∀ x, line1 x = f x → x = 2) ∧
  (∀ x, line2 x = f x → x = 2) ∧
  line1 (point.1) = point.2 ∧
  line2 (point.1) = point.2 ∧
  f' (point.1) = 9 :=
sorry

end tangent_line_at_point_l47_4790


namespace july_birth_percentage_l47_4709

/-- The percentage of people born in July, given 15 out of 120 famous Americans were born in July -/
theorem july_birth_percentage :
  let total_people : ℕ := 120
  let july_births : ℕ := 15
  (july_births : ℚ) / total_people * 100 = 25 / 2 := by
  sorry

end july_birth_percentage_l47_4709


namespace intersection_M_N_l47_4712

def M : Set ℝ := {1, 2, 3, 4, 5}
def N : Set ℝ := {x | Real.log x / Real.log 4 ≥ 1}

theorem intersection_M_N : M ∩ N = {4, 5} := by
  sorry

end intersection_M_N_l47_4712


namespace power_of_two_greater_than_square_minus_two_l47_4755

theorem power_of_two_greater_than_square_minus_two (n : ℕ) (h : n > 0) : 
  2^n > n^2 - 2 :=
by
  -- Assume the proposition holds for n = 1, n = 2, and n = 3
  have base_case_1 : 2^1 > 1^2 - 2 := by sorry
  have base_case_2 : 2^2 > 2^2 - 2 := by sorry
  have base_case_3 : 2^3 > 3^2 - 2 := by sorry

  -- Proof by induction
  induction n with
  | zero => contradiction
  | succ n ih =>
    -- Inductive step
    sorry

end power_of_two_greater_than_square_minus_two_l47_4755


namespace xiao_hong_books_l47_4749

/-- Given that Xiao Hong originally had 5 books and bought 'a' more books,
    prove that her total number of books now is 5 + a. -/
theorem xiao_hong_books (a : ℕ) : 5 + a = 5 + a := by sorry

end xiao_hong_books_l47_4749


namespace danny_jane_age_difference_l47_4788

theorem danny_jane_age_difference : ∃ (x : ℝ), 
  (40 : ℝ) - x = (4.5 : ℝ) * ((26 : ℝ) - x) ∧ x = 22 := by
  sorry

end danny_jane_age_difference_l47_4788


namespace inequality_expression_l47_4781

theorem inequality_expression (x : ℝ) : (x + 4 ≥ -1) ↔ (x + 4 ≥ -1) := by sorry

end inequality_expression_l47_4781


namespace whale_first_hour_consumption_l47_4757

def whale_feeding (first_hour : ℕ) : Prop :=
  let second_hour := first_hour + 3
  let third_hour := first_hour + 6
  let fourth_hour := first_hour + 9
  let fifth_hour := first_hour + 12
  (third_hour = 93) ∧ 
  (first_hour + second_hour + third_hour + fourth_hour + fifth_hour = 450)

theorem whale_first_hour_consumption : 
  ∃ (x : ℕ), whale_feeding x ∧ x = 87 :=
sorry

end whale_first_hour_consumption_l47_4757


namespace P_in_first_quadrant_l47_4711

/-- A point in the Cartesian coordinate system -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- Definition of the first quadrant -/
def isInFirstQuadrant (p : CartesianPoint) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- The given point P(3,2) -/
def P : CartesianPoint :=
  { x := 3, y := 2 }

/-- Theorem: Point P(3,2) lies in the first quadrant -/
theorem P_in_first_quadrant : isInFirstQuadrant P := by
  sorry


end P_in_first_quadrant_l47_4711


namespace existence_of_special_numbers_l47_4739

theorem existence_of_special_numbers :
  ∃ (a b c : ℕ),
    a > 10^10 ∧ b > 10^10 ∧ c > 10^10 ∧
    (a * b * c) % (a + 2012) = 0 ∧
    (a * b * c) % (b + 2012) = 0 ∧
    (a * b * c) % (c + 2012) = 0 :=
by sorry

end existence_of_special_numbers_l47_4739


namespace not_right_triangle_l47_4774

theorem not_right_triangle (a b c : ℝ) (ha : a = Real.sqrt 3) (hb : b = Real.sqrt 4) (hc : c = Real.sqrt 5) :
  ¬(a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) :=
by sorry

end not_right_triangle_l47_4774


namespace no_integer_solution_l47_4708

theorem no_integer_solution : ¬∃ (x y : ℤ), (x + 2020) * (x + 2021) + (x + 2021) * (x + 2022) + (x + 2020) * (x + 2022) = y^2 := by
  sorry

end no_integer_solution_l47_4708


namespace students_taking_neither_music_nor_art_l47_4722

theorem students_taking_neither_music_nor_art 
  (total : ℕ) (music : ℕ) (art : ℕ) (both : ℕ) : 
  total = 500 → music = 30 → art = 20 → both = 10 →
  total - (music + art - both) = 460 := by
sorry

end students_taking_neither_music_nor_art_l47_4722


namespace discount_from_profit_l47_4703

/-- Represents a car sale transaction -/
structure CarSale where
  originalPrice : ℝ
  discountRate : ℝ
  profitRate : ℝ
  sellIncrease : ℝ

/-- Theorem stating the relationship between discount and profit in a specific car sale scenario -/
theorem discount_from_profit (sale : CarSale) 
  (h1 : sale.profitRate = 0.28000000000000004)
  (h2 : sale.sellIncrease = 0.60) : 
  sale.discountRate = 0.5333333333333333 := by
  sorry

end discount_from_profit_l47_4703


namespace cube_root_equation_solution_l47_4735

theorem cube_root_equation_solution :
  ∃ y : ℝ, (5 + 2 / y)^(1/3) = -3 ↔ y = -1/16 := by sorry

end cube_root_equation_solution_l47_4735


namespace quadrilateral_diagonal_theorem_l47_4707

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Checks if a quadrilateral is convex -/
def is_convex (q : Quadrilateral) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Finds the intersection point of two line segments -/
def intersection (p1 p2 p3 p4 : Point) : Point := sorry

/-- Calculates the area of a triangle given three points -/
def triangle_area (p1 p2 p3 : Point) : ℝ := sorry

/-- Main theorem -/
theorem quadrilateral_diagonal_theorem (ABCD : Quadrilateral) (O : Point) :
  is_convex ABCD →
  distance ABCD.A ABCD.B = 10 →
  distance ABCD.C ABCD.D = 15 →
  distance ABCD.A ABCD.C = 20 →
  O = intersection ABCD.A ABCD.C ABCD.B ABCD.D →
  triangle_area ABCD.A O ABCD.D = triangle_area ABCD.B O ABCD.C →
  distance ABCD.A O = 8 := by
  sorry

end quadrilateral_diagonal_theorem_l47_4707


namespace simplify_expression_1_simplify_expression_2_l47_4704

-- Problem 1
theorem simplify_expression_1 (x : ℝ) :
  (x + 3) * (3 * x - 2) + x * (1 - 3 * x) = 8 * x - 6 := by sorry

-- Problem 2
theorem simplify_expression_2 (m : ℝ) (h1 : m ≠ 2) (h2 : m ≠ -2) :
  (1 - m / (m + 2)) / ((m^2 - 4*m + 4) / (m^2 - 4)) = 2 / (m - 2) := by sorry

end simplify_expression_1_simplify_expression_2_l47_4704


namespace remainder_sum_l47_4728

theorem remainder_sum (n : ℤ) (h : n % 24 = 10) : (n % 4 + n % 6 = 6) := by
  sorry

end remainder_sum_l47_4728


namespace distance_to_point_l47_4796

/-- The distance from the origin to the point (8, -3, 6) in 3D space is √109. -/
theorem distance_to_point : Real.sqrt 109 = Real.sqrt (8^2 + (-3)^2 + 6^2) := by
  sorry

end distance_to_point_l47_4796


namespace pencil_notebook_cost_l47_4718

/-- The cost of pencils and notebooks given specific conditions --/
theorem pencil_notebook_cost :
  ∀ (p n : ℝ),
  -- Condition 1: 9 pencils and 10 notebooks cost $5.35
  9 * p + 10 * n = 5.35 →
  -- Condition 2: 6 pencils and 4 notebooks cost $2.50
  6 * p + 4 * n = 2.50 →
  -- The cost of 24 pencils (with 10% discount on packs of 4) and 15 notebooks is $9.24
  24 * (0.9 * p) + 15 * n = 9.24 :=
by
  sorry


end pencil_notebook_cost_l47_4718


namespace least_with_twelve_factors_l47_4754

/-- A function that counts the number of positive factors of a natural number -/
def count_factors (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number has exactly 12 positive factors -/
def has_twelve_factors (n : ℕ) : Prop := count_factors n = 12

/-- Theorem stating that 108 is the least positive integer with exactly 12 positive factors -/
theorem least_with_twelve_factors :
  (∀ m : ℕ, m > 0 → m < 108 → ¬(has_twelve_factors m)) ∧ has_twelve_factors 108 := by
  sorry

end least_with_twelve_factors_l47_4754


namespace sum_of_x_values_l47_4729

open Real

theorem sum_of_x_values (x : ℝ) : 
  (0 < x) → 
  (x < 180) → 
  (sin (2 * x * π / 180))^3 + (sin (6 * x * π / 180))^3 = 
    8 * (sin (3 * x * π / 180))^3 * (sin (x * π / 180))^3 → 
  ∃ (x₁ x₂ x₃ : ℝ), 
    (0 < x₁) ∧ (x₁ < 180) ∧
    (0 < x₂) ∧ (x₂ < 180) ∧
    (0 < x₃) ∧ (x₃ < 180) ∧
    (sin (2 * x₁ * π / 180))^3 + (sin (6 * x₁ * π / 180))^3 = 
      8 * (sin (3 * x₁ * π / 180))^3 * (sin (x₁ * π / 180))^3 ∧
    (sin (2 * x₂ * π / 180))^3 + (sin (6 * x₂ * π / 180))^3 = 
      8 * (sin (3 * x₂ * π / 180))^3 * (sin (x₂ * π / 180))^3 ∧
    (sin (2 * x₃ * π / 180))^3 + (sin (6 * x₃ * π / 180))^3 = 
      8 * (sin (3 * x₃ * π / 180))^3 * (sin (x₃ * π / 180))^3 ∧
    x₁ + x₂ + x₃ = 270 :=
by
  sorry

end sum_of_x_values_l47_4729


namespace line_segment_ratios_l47_4765

/-- Given four points X, Y, Z, W on a straight line in that order,
    with XY = 3, YZ = 4, and XW = 20, prove that
    the ratio of XZ to YW is 7/16 and the ratio of YZ to XW is 1/5. -/
theorem line_segment_ratios
  (X Y Z W : ℝ)  -- Points represented as real numbers
  (h_order : X < Y ∧ Y < Z ∧ Z < W)  -- Order of points
  (h_xy : Y - X = 3)  -- XY = 3
  (h_yz : Z - Y = 4)  -- YZ = 4
  (h_xw : W - X = 20)  -- XW = 20
  : (Z - X) / (W - Y) = 7 / 16 ∧ (Z - Y) / (W - X) = 1 / 5 := by
  sorry


end line_segment_ratios_l47_4765


namespace slope_of_line_l47_4799

theorem slope_of_line (x₁ x₂ y₁ y₂ : ℝ) (h₁ : x₁ ≠ x₂) 
  (h₂ : (4 / x₁) + (5 / y₁) = 0) (h₃ : (4 / x₂) + (5 / y₂) = 0) :
  (y₂ - y₁) / (x₂ - x₁) = -5/4 := by
  sorry

end slope_of_line_l47_4799


namespace quadratic_form_sum_l47_4743

theorem quadratic_form_sum (x : ℝ) :
  ∃ (b c : ℝ), x^2 - 18*x + 45 = (x + b)^2 + c ∧ b + c = -45 := by
  sorry

end quadratic_form_sum_l47_4743


namespace steaks_needed_l47_4710

def family_members : ℕ := 5
def pounds_per_person : ℕ := 1
def ounces_per_steak : ℕ := 20
def ounces_per_pound : ℕ := 16

def total_ounces : ℕ := family_members * pounds_per_person * ounces_per_pound

theorem steaks_needed : (total_ounces / ounces_per_steak : ℕ) = 4 := by
  sorry

end steaks_needed_l47_4710


namespace equation_solution_iff_common_root_l47_4753

theorem equation_solution_iff_common_root
  (a : ℝ) (f g h : ℝ → ℝ) 
  (ha : a > 1)
  (h_sum_nonneg : ∀ x, f x + g x + h x ≥ 0) :
  (∃ x, a^(f x) + a^(g x) + a^(h x) = 3) ↔ 
  (∃ x, f x = 0 ∧ g x = 0 ∧ h x = 0) :=
by sorry

end equation_solution_iff_common_root_l47_4753


namespace least_integer_greater_than_sqrt_500_l47_4713

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℕ, (n : ℝ) > Real.sqrt 500 ∧ ∀ m : ℕ, (m : ℝ) > Real.sqrt 500 → m ≥ n :=
sorry

end least_integer_greater_than_sqrt_500_l47_4713


namespace greatest_common_factor_of_150_225_300_l47_4745

theorem greatest_common_factor_of_150_225_300 : 
  Nat.gcd 150 (Nat.gcd 225 300) = 75 := by
  sorry

end greatest_common_factor_of_150_225_300_l47_4745


namespace sequence_term_proof_l47_4752

def sequence_sum (n : ℕ) := 3^n + 2

def sequence_term (n : ℕ) : ℝ :=
  if n = 1 then 5 else 2 * 3^(n-1)

theorem sequence_term_proof (n : ℕ) :
  sequence_term n = 
    if n = 1 
    then sequence_sum 1
    else sequence_sum n - sequence_sum (n-1) :=
by sorry

end sequence_term_proof_l47_4752


namespace modulo_equivalence_solution_l47_4725

theorem modulo_equivalence_solution :
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 15 ∧ n ≡ 12345 [ZMOD 11] ∧ n = 3 := by
  sorry

end modulo_equivalence_solution_l47_4725


namespace polynomial_sum_l47_4759

theorem polynomial_sum (x : ℝ) : (x^2 + 3*x - 4) + (-3*x + 1) = x^2 - 3 := by
  sorry

end polynomial_sum_l47_4759


namespace divisibility_condition_l47_4736

theorem divisibility_condition (n : ℤ) : (n + 1) ∣ (n^2 + 1) ↔ n = -3 ∨ n = -2 ∨ n = 0 ∨ n = 1 := by
  sorry

end divisibility_condition_l47_4736


namespace largest_whole_number_nine_times_less_than_150_l47_4744

theorem largest_whole_number_nine_times_less_than_150 :
  ∀ x : ℕ, x ≤ 16 ↔ 9 * x < 150 :=
by sorry

end largest_whole_number_nine_times_less_than_150_l47_4744


namespace unique_prime_pair_solution_l47_4731

theorem unique_prime_pair_solution : 
  ∃! p q : ℕ, Prime p ∧ Prime q ∧ p^3 - q^5 = (p + q)^2 := by
  sorry

end unique_prime_pair_solution_l47_4731


namespace minimum_male_students_l47_4798

theorem minimum_male_students (num_benches : ℕ) (students_per_bench : ℕ) :
  num_benches = 29 →
  students_per_bench = 5 →
  ∃ (male_students : ℕ),
    male_students ≥ 29 ∧
    male_students * 5 ≥ num_benches * students_per_bench ∧
    ∀ m : ℕ, m < 29 → m * 5 < num_benches * students_per_bench :=
by sorry

end minimum_male_students_l47_4798


namespace current_tariff_calculation_specific_case_calculation_l47_4741

/-- Calculates the current actual tariff after two successive reductions -/
def current_tariff (S : ℝ) : ℝ := (1 - 0.4) * (1 - 0.3) * S

/-- Theorem stating the current actual tariff calculation -/
theorem current_tariff_calculation (S : ℝ) : 
  current_tariff S = (1 - 0.4) * (1 - 0.3) * S := by sorry

/-- Theorem for the specific case when S = 1000 -/
theorem specific_case_calculation : 
  current_tariff 1000 = 420 := by sorry

end current_tariff_calculation_specific_case_calculation_l47_4741


namespace inequality_system_solution_l47_4747

theorem inequality_system_solution (m : ℝ) : 
  (∃ x : ℤ, (x : ℝ) ≥ 2 ∧ (x - m) / 2 ≥ 2 ∧ x - 4 ≤ 3 * (x - 2) ∧ 
   ∀ y : ℤ, y < x → (y : ℝ) - m < 4 ∨ y - 4 > 3 * (y - 2)) →
  -3 < m ∧ m ≤ -2 :=
by sorry

end inequality_system_solution_l47_4747


namespace solution_to_equation_l47_4717

theorem solution_to_equation (x : ℝ) : -200 * x = 1600 → x = -8 := by
  sorry

end solution_to_equation_l47_4717


namespace negation_equivalence_l47_4797

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 > 0) := by
  sorry

end negation_equivalence_l47_4797


namespace hose_flow_rate_l47_4715

/-- Given a pool that takes 50 hours to fill, water costs 1 cent for 10 gallons,
    and it costs 5 dollars to fill the pool, the hose runs at a rate of 100 gallons per hour. -/
theorem hose_flow_rate (fill_time : ℕ) (water_cost : ℚ) (fill_cost : ℕ) :
  fill_time = 50 →
  water_cost = 1 / 10 →
  fill_cost = 5 →
  (fill_cost * 100 : ℚ) / (water_cost * fill_time) = 100 := by
  sorry

end hose_flow_rate_l47_4715


namespace elias_bananas_l47_4795

def bananas_eaten (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

theorem elias_bananas : 
  let initial := 12
  let remaining := 11
  bananas_eaten initial remaining = 1 := by
sorry

end elias_bananas_l47_4795


namespace crazy_silly_school_remaining_books_l47_4772

/-- Given a book series with a total number of books and a number of books already read,
    calculate the number of books remaining to be read. -/
def books_remaining (total : ℕ) (read : ℕ) : ℕ := total - read

/-- Theorem stating that for the 'crazy silly school' series with 14 total books
    and 8 books already read, there are 6 books remaining to be read. -/
theorem crazy_silly_school_remaining_books :
  books_remaining 14 8 = 6 := by
  sorry

end crazy_silly_school_remaining_books_l47_4772
