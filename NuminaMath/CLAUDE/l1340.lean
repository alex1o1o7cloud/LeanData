import Mathlib

namespace quadratic_inequality_solution_l1340_134064

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, ax^2 - 2*x + b > 0 ↔ -3 < x ∧ x < 1) →
  (a = -1 ∧ b = 3 ∧ 
   ∀ x, 3*x^2 - x - 2 ≤ 0 ↔ -2/3 ≤ x ∧ x ≤ 1) :=
by sorry

end quadratic_inequality_solution_l1340_134064


namespace lowest_common_multiple_10_to_30_l1340_134008

theorem lowest_common_multiple_10_to_30 :
  ∃ (n : ℕ), n > 0 ∧
  (∀ k : ℕ, 10 ≤ k ∧ k ≤ 30 → k ∣ n) ∧
  (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, 10 ≤ k ∧ k ≤ 30 → k ∣ m) → n ≤ m) ∧
  n = 232792560 :=
by sorry

end lowest_common_multiple_10_to_30_l1340_134008


namespace cost_of_four_books_l1340_134023

/-- Given that two identical books cost $36, prove that four of these books cost $72. -/
theorem cost_of_four_books (cost_of_two : ℝ) (h : cost_of_two = 36) : 
  2 * cost_of_two = 72 := by
  sorry

end cost_of_four_books_l1340_134023


namespace volume_of_rotated_specific_cone_l1340_134063

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a triangle in 3D space -/
structure Triangle3D where
  O : Point3D
  A : Point3D
  B : Point3D

/-- Represents a cone in 3D space -/
structure Cone3D where
  base_center : Point3D
  apex : Point3D
  base_radius : ℝ

/-- Function to create a cone by rotating a triangle around the x-axis -/
def createConeFromTriangle (t : Triangle3D) : Cone3D :=
  { base_center := ⟨t.A.x, 0, 0⟩,
    apex := t.O,
    base_radius := t.B.y - t.A.y }

/-- Function to calculate the volume of a solid obtained by rotating a cone around the y-axis -/
noncomputable def volumeOfRotatedCone (c : Cone3D) : ℝ := sorry

/-- The main theorem to prove -/
theorem volume_of_rotated_specific_cone :
  let t : Triangle3D := { O := ⟨0, 0, 0⟩, A := ⟨1, 0, 0⟩, B := ⟨1, 1, 0⟩ }
  let c : Cone3D := createConeFromTriangle t
  volumeOfRotatedCone c = (8 * Real.pi) / 3 := by sorry

end volume_of_rotated_specific_cone_l1340_134063


namespace intersection_dot_product_l1340_134002

/-- An ellipse with equation x²/25 + y²/16 = 1 -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

/-- A hyperbola with equation x²/4 - y²/5 = 1 -/
def is_on_hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

/-- The common foci of the ellipse and hyperbola -/
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

/-- A point P that lies on both the ellipse and the hyperbola -/
def P : ℝ × ℝ := sorry

/-- Vector from P to F₁ -/
def PF₁ : ℝ × ℝ := (F₁.1 - P.1, F₁.2 - P.2)

/-- Vector from P to F₂ -/
def PF₂ : ℝ × ℝ := (F₂.1 - P.1, F₂.2 - P.2)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem intersection_dot_product :
  is_on_ellipse P.1 P.2 ∧ is_on_hyperbola P.1 P.2 →
  dot_product PF₁ PF₂ = 11 := by sorry

end intersection_dot_product_l1340_134002


namespace gravel_path_cost_l1340_134060

def plot_length : ℝ := 110
def plot_width : ℝ := 65
def path_width : ℝ := 2.5
def cost_per_sq_meter_paise : ℝ := 80

theorem gravel_path_cost :
  let larger_length := plot_length + 2 * path_width
  let larger_width := plot_width + 2 * path_width
  let larger_area := larger_length * larger_width
  let plot_area := plot_length * plot_width
  let path_area := larger_area - plot_area
  let cost_per_sq_meter_rupees := cost_per_sq_meter_paise / 100
  path_area * cost_per_sq_meter_rupees = 720 :=
by sorry

end gravel_path_cost_l1340_134060


namespace complex_expression_equals_one_l1340_134025

theorem complex_expression_equals_one : 
  Real.sqrt 6 / Real.sqrt 2 + |1 - Real.sqrt 3| - Real.sqrt 12 + (1/2)⁻¹ = 1 := by
  sorry

end complex_expression_equals_one_l1340_134025


namespace probability_heart_then_diamond_l1340_134047

/-- The probability of drawing a heart first and a diamond second from a standard deck of cards -/
theorem probability_heart_then_diamond (total_cards : ℕ) (suits : ℕ) (cards_per_suit : ℕ) 
  (h_total : total_cards = 52)
  (h_suits : suits = 4)
  (h_cards_per_suit : cards_per_suit = 13)
  (h_deck : total_cards = suits * cards_per_suit) :
  (cards_per_suit : ℚ) / total_cards * cards_per_suit / (total_cards - 1) = 13 / 204 := by
  sorry

end probability_heart_then_diamond_l1340_134047


namespace distance_to_origin_l1340_134034

/-- The distance from point P(1, 2, 2) to the origin (0, 0, 0) is 3. -/
theorem distance_to_origin : Real.sqrt (1^2 + 2^2 + 2^2) = 3 := by
  sorry

end distance_to_origin_l1340_134034


namespace geometric_sequence_product_l1340_134010

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 7 * a 13 = 2) →
  (a 7 + a 13 = 3) →
  a 2 * a 18 = 2 := by
  sorry

end geometric_sequence_product_l1340_134010


namespace balcony_seat_cost_l1340_134044

/-- Theorem: Cost of a balcony seat in a theater --/
theorem balcony_seat_cost
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (orchestra_price : ℕ)
  (balcony_orchestra_diff : ℕ)
  (h1 : total_tickets = 355)
  (h2 : total_revenue = 3320)
  (h3 : orchestra_price = 12)
  (h4 : balcony_orchestra_diff = 115) :
  ∃ (balcony_price : ℕ),
    balcony_price = 8 ∧
    balcony_price * (total_tickets / 2 + balcony_orchestra_diff / 2) +
    orchestra_price * (total_tickets / 2 - balcony_orchestra_diff / 2) =
    total_revenue :=
by sorry

end balcony_seat_cost_l1340_134044


namespace area_relation_l1340_134065

/-- A square with vertices O, P, Q, R where O is the origin and Q is at (3,3) -/
structure Square :=
  (O : ℝ × ℝ)
  (P : ℝ × ℝ)
  (Q : ℝ × ℝ)
  (R : ℝ × ℝ)
  (is_origin : O = (0, 0))
  (is_square : Q = (3, 3))

/-- The area of a square -/
def area_square (s : Square) : ℝ := sorry

/-- The area of a triangle given three points -/
def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

/-- The theorem stating that T(3, -12) makes the area of triangle PQT twice the area of square OPQR -/
theorem area_relation (s : Square) : 
  let T : ℝ × ℝ := (3, -12)
  area_triangle s.P s.Q T = 2 * area_square s := by sorry

end area_relation_l1340_134065


namespace cd_length_in_isosceles_triangles_l1340_134088

/-- An isosceles triangle with given side lengths -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := 2 * t.leg + t.base

theorem cd_length_in_isosceles_triangles 
  (abc : IsoscelesTriangle) 
  (cbd : IsoscelesTriangle) 
  (h1 : perimeter cbd = 25)
  (h2 : perimeter abc = 20)
  (h3 : cbd.base = 9) :
  cbd.leg = 8 := by
  sorry

end cd_length_in_isosceles_triangles_l1340_134088


namespace multiply_polynomials_l1340_134013

theorem multiply_polynomials (x : ℝ) : 
  (x^4 + 8*x^2 + 16) * (x^2 - 4) = x^4 + 8*x^2 + 12 := by
sorry

end multiply_polynomials_l1340_134013


namespace unique_number_in_intersection_l1340_134019

theorem unique_number_in_intersection : ∃! x : ℝ, 3 < x ∧ x < 8 ∧ 6 < x ∧ x < 10 := by
  sorry

end unique_number_in_intersection_l1340_134019


namespace parabola_equation_l1340_134070

/-- A parabola with vertex at the origin and focus at (0, 3) has the equation x^2 = 12y -/
theorem parabola_equation (p : ℝ × ℝ → Prop) :
  (∀ x y, p (x, y) ↔ x^2 = 12*y) →
  (p (0, 0)) →  -- vertex at origin
  (∀ x y, x^2 + (y - 3)^2 = 4 → p (0, y)) →  -- focus at center of circle
  ∀ x y, p (x, y) ↔ x^2 = 12*y :=
by sorry

end parabola_equation_l1340_134070


namespace circles_intersection_theorem_l1340_134029

-- Define the basic structures
structure Point : Type :=
  (x y : ℝ)

structure Circle : Type :=
  (center : Point)
  (radius : ℝ)

-- Define the given conditions
def O₁ : Point := sorry
def O₂ : Point := sorry
def A : Point := sorry
def B : Point := sorry
def P : Point := sorry
def Q : Point := sorry

def circle₁ : Circle := ⟨O₁, sorry⟩
def circle₂ : Circle := ⟨O₂, sorry⟩

-- Define the necessary predicates
def intersect (c₁ c₂ : Circle) (p : Point) : Prop := sorry
def on_circle (c : Circle) (p : Point) : Prop := sorry
def on_segment (p₁ p₂ p : Point) : Prop := sorry

-- State the theorem
theorem circles_intersection_theorem :
  intersect circle₁ circle₂ A ∧
  intersect circle₁ circle₂ B ∧
  on_circle circle₁ Q ∧
  on_circle circle₂ P ∧
  (∃ (c : Circle), on_circle c O₁ ∧ on_circle c A ∧ on_circle c O₂ ∧ on_circle c P ∧ on_circle c Q) →
  on_segment O₁ Q B ∧ on_segment O₂ P B :=
sorry

end circles_intersection_theorem_l1340_134029


namespace division_multiplication_problem_l1340_134057

theorem division_multiplication_problem : (180 / 6) / 3 * 2 = 20 := by
  sorry

end division_multiplication_problem_l1340_134057


namespace suzanna_ride_l1340_134052

/-- Calculates the distance traveled given a constant rate and time -/
def distanceTraveled (rate : ℚ) (time : ℚ) : ℚ :=
  rate * time

theorem suzanna_ride : 
  let rate : ℚ := 1.5 / 4  -- miles per minute
  let time : ℚ := 40       -- minutes
  distanceTraveled rate time = 15 := by
sorry

#eval (1.5 / 4) * 40  -- To verify the result

end suzanna_ride_l1340_134052


namespace prime_representation_l1340_134098

theorem prime_representation (N : ℕ) (hN : Nat.Prime N) :
  ∃ (n : ℤ) (p : ℕ), Nat.Prime p ∧ p < 30 ∧ N = 30 * n.natAbs + p :=
sorry

end prime_representation_l1340_134098


namespace total_cds_count_l1340_134030

/-- The number of CDs Dawn has -/
def dawn_cds : ℕ := 10

/-- The number of CDs Kristine has -/
def kristine_cds : ℕ := dawn_cds + 7

/-- The number of CDs Mark has -/
def mark_cds : ℕ := 2 * kristine_cds

/-- The total number of CDs owned by Dawn, Kristine, and Mark -/
def total_cds : ℕ := dawn_cds + kristine_cds + mark_cds

theorem total_cds_count : total_cds = 61 := by
  sorry

end total_cds_count_l1340_134030


namespace binomial_square_condition_l1340_134042

theorem binomial_square_condition (a : ℝ) : 
  (∃ (p q : ℝ), ∀ x, 4*x^2 + 16*x + a = (p*x + q)^2) → a = 16 := by
  sorry

end binomial_square_condition_l1340_134042


namespace systematic_sample_fourth_element_l1340_134049

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  first_element : ℕ
  h_pop_size : population_size > 0
  h_sample_size : sample_size > 0
  h_sample_size_le_pop : sample_size ≤ population_size
  h_first_element : first_element > 0 ∧ first_element ≤ population_size

/-- The interval between elements in a systematic sample -/
def SystematicSample.interval (s : SystematicSample) : ℕ :=
  s.population_size / s.sample_size

/-- Checks if a number is in the systematic sample -/
def SystematicSample.contains (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = s.first_element + k * s.interval ∧ n ≤ s.population_size

/-- The theorem to be proved -/
theorem systematic_sample_fourth_element
  (s : SystematicSample)
  (h_pop_size : s.population_size = 36)
  (h_sample_size : s.sample_size = 4)
  (h_contains_5 : s.contains 5)
  (h_contains_23 : s.contains 23)
  (h_contains_32 : s.contains 32) :
  s.contains 14 :=
sorry

end systematic_sample_fourth_element_l1340_134049


namespace existence_of_prime_and_cube_root_l1340_134036

theorem existence_of_prime_and_cube_root (n : ℕ+) :
  ∃ (p : ℕ) (m : ℤ), 
    Nat.Prime p ∧ 
    p % 6 = 5 ∧ 
    ¬(p ∣ n.val) ∧ 
    n.val % p = (m ^ 3) % p :=
by sorry

end existence_of_prime_and_cube_root_l1340_134036


namespace exists_integer_function_double_application_square_l1340_134004

theorem exists_integer_function_double_application_square :
  ∃ f : ℤ → ℤ, ∀ n : ℤ, f (f n) = n^2 := by
  sorry

end exists_integer_function_double_application_square_l1340_134004


namespace computer_cost_l1340_134035

theorem computer_cost (total_budget : ℕ) (tv_cost : ℕ) (fridge_computer_diff : ℕ) 
  (h1 : total_budget = 1600)
  (h2 : tv_cost = 600)
  (h3 : fridge_computer_diff = 500) : 
  ∃ (computer_cost : ℕ), 
    computer_cost + tv_cost + (computer_cost + fridge_computer_diff) = total_budget ∧ 
    computer_cost = 250 := by
  sorry

end computer_cost_l1340_134035


namespace machine_purchase_price_l1340_134003

def machine_value (purchase_price : ℝ) (years : ℕ) : ℝ :=
  purchase_price * (1 - 0.3) ^ years

theorem machine_purchase_price : 
  ∃ (purchase_price : ℝ), 
    purchase_price > 0 ∧ 
    machine_value purchase_price 2 = 3200 ∧
    purchase_price = 8000 := by
  sorry

end machine_purchase_price_l1340_134003


namespace algebraic_expression_theorem_l1340_134053

-- Define the algebraic expression
def algebraic_expression (a b x : ℝ) : ℝ :=
  (a*x - 3) * (2*x + 4) - x^2 - b

-- Define the condition for no x^2 term
def no_x_squared_term (a : ℝ) : Prop :=
  2*a - 1 = 0

-- Define the condition for no constant term
def no_constant_term (b : ℝ) : Prop :=
  -12 - b = 0

-- Define the final expression to be calculated
def final_expression (a b : ℝ) : ℝ :=
  (2*a + b)^2 - (2 - 2*b)*(2 + 2*b) - 3*a*(a - b)

-- Theorem statement
theorem algebraic_expression_theorem (a b : ℝ) :
  no_x_squared_term a ∧ no_constant_term b →
  a = 1/2 ∧ b = -12 ∧ final_expression a b = 678 := by
  sorry

end algebraic_expression_theorem_l1340_134053


namespace probability_diamond_or_ace_l1340_134092

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (target_cards : ℕ)
  (h_total : total_cards = 52)
  (h_target : target_cards = 16)

/-- The probability of drawing at least one target card in two draws with replacement -/
def probability_at_least_one (d : Deck) : ℚ :=
  1 - (((d.total_cards - d.target_cards : ℚ) / d.total_cards) ^ 2)

theorem probability_diamond_or_ace (d : Deck) :
  probability_at_least_one d = 88 / 169 := by
  sorry

end probability_diamond_or_ace_l1340_134092


namespace restaurant_bill_average_cost_l1340_134046

theorem restaurant_bill_average_cost
  (total_bill : ℝ)
  (gratuity_rate : ℝ)
  (num_people : ℕ)
  (h1 : total_bill = 720)
  (h2 : gratuity_rate = 0.2)
  (h3 : num_people = 6) :
  (total_bill / (1 + gratuity_rate)) / num_people = 100 :=
by sorry

end restaurant_bill_average_cost_l1340_134046


namespace library_shelves_l1340_134045

theorem library_shelves (books : ℕ) (additional_books : ℕ) (shelves : ℕ) : 
  books = 4305 →
  additional_books = 11 →
  (books + additional_books) % shelves = 0 →
  shelves = 11 :=
by sorry

end library_shelves_l1340_134045


namespace abs_diff_inequality_l1340_134027

theorem abs_diff_inequality (x : ℝ) : |x| - |x - 3| < 2 ↔ x < (5/2) := by sorry

end abs_diff_inequality_l1340_134027


namespace externally_tangent_circles_m_value_l1340_134075

/-- A circle in the 2D plane defined by its equation coefficients -/
structure Circle where
  a : ℝ -- coefficient of x^2
  b : ℝ -- coefficient of y^2
  c : ℝ -- coefficient of x
  d : ℝ -- coefficient of y
  e : ℝ -- constant term

/-- Check if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let center1 := (- c1.c / (2 * c1.a), - c1.d / (2 * c1.b))
  let center2 := (- c2.c / (2 * c2.a), - c2.d / (2 * c2.b))
  let radius1 := Real.sqrt ((c1.c^2 / (4 * c1.a^2) + c1.d^2 / (4 * c1.b^2) - c1.e / c1.a))
  let radius2 := Real.sqrt ((c2.c^2 / (4 * c2.a^2) + c2.d^2 / (4 * c2.b^2) - c2.e / c2.a))
  let distance := Real.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2)
  distance = radius1 + radius2

/-- The main theorem -/
theorem externally_tangent_circles_m_value :
  ∀ m : ℝ,
  let c1 : Circle := { a := 1, b := 1, c := -2, d := -4, e := m }
  let c2 : Circle := { a := 1, b := 1, c := -8, d := -12, e := 36 }
  are_externally_tangent c1 c2 → m = 4 := by
  sorry

end externally_tangent_circles_m_value_l1340_134075


namespace garden_to_land_ratio_l1340_134062

/-- A rectangle with width 3/5 of its length -/
structure Rectangle where
  length : ℝ
  width : ℝ
  width_prop : width = (3/5) * length

theorem garden_to_land_ratio (land garden : Rectangle) : 
  (garden.length * garden.width) / (land.length * land.width) = 9/25 := by
  sorry

end garden_to_land_ratio_l1340_134062


namespace least_subtraction_l1340_134090

theorem least_subtraction (n : ℕ) : ∃! x : ℕ, 
  (∀ d ∈ ({9, 11, 17} : Set ℕ), (3381 - x) % d = 8) ∧ 
  (∀ y : ℕ, y < x → ∃ d ∈ ({9, 11, 17} : Set ℕ), (3381 - y) % d ≠ 8) :=
by sorry

end least_subtraction_l1340_134090


namespace expression_equals_6500_l1340_134081

theorem expression_equals_6500 : (2015 / 1 + 2015 / 0.31) / (1 + 0.31) = 6500 := by
  sorry

end expression_equals_6500_l1340_134081


namespace no_positive_triples_sum_l1340_134096

theorem no_positive_triples_sum : 
  ¬∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a = b + c ∧ b = c + a ∧ c = a + b := by
  sorry

end no_positive_triples_sum_l1340_134096


namespace exactly_one_valid_number_l1340_134026

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧  -- 3-digit whole number
  (n / 100 + (n / 10) % 10 + n % 10 = 28) ∧  -- digit-sum is 28
  (n % 10 < 7) ∧  -- units digit is less than 7
  (n % 2 = 0)  -- units digit is an even number

theorem exactly_one_valid_number : 
  ∃! n : ℕ, is_valid_number n :=
sorry

end exactly_one_valid_number_l1340_134026


namespace round_robin_tournament_teams_l1340_134037

/-- The number of games in a round-robin tournament with n teams -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a round-robin tournament with 28 games, there are 8 teams -/
theorem round_robin_tournament_teams : ∃ (n : ℕ), n > 0 ∧ num_games n = 28 ∧ n = 8 := by
  sorry

end round_robin_tournament_teams_l1340_134037


namespace common_point_polar_coords_l1340_134041

-- Define the circle O in polar coordinates
def circle_O (ρ θ : ℝ) : Prop := ρ = Real.cos θ + Real.sin θ

-- Define the line l in polar coordinates
def line_l (ρ θ : ℝ) : Prop := ρ * Real.sin (θ - Real.pi / 4) = Real.sqrt 2 / 2

-- Theorem statement
theorem common_point_polar_coords :
  ∃ (ρ θ : ℝ), 
    circle_O ρ θ ∧ 
    line_l ρ θ ∧ 
    0 < θ ∧ 
    θ < Real.pi ∧ 
    ρ = 1 ∧ 
    θ = Real.pi / 2 :=
sorry

end common_point_polar_coords_l1340_134041


namespace point_on_same_side_l1340_134055

/-- A point (x, y) is on the same side of the line 2x - y + 1 = 0 as (1, 2) if both points satisfy 2x - y + 1 > 0 -/
def same_side (x y : ℝ) : Prop :=
  2*x - y + 1 > 0 ∧ 2*1 - 2 + 1 > 0

/-- The point (1, 0) is on the same side of the line 2x - y + 1 = 0 as the point (1, 2) -/
theorem point_on_same_side : same_side 1 0 := by
  sorry

end point_on_same_side_l1340_134055


namespace games_mike_can_buy_l1340_134058

theorem games_mike_can_buy (initial_amount : ℕ) (spent_amount : ℕ) (game_cost : ℕ) : 
  initial_amount = 69 → spent_amount = 24 → game_cost = 5 →
  (initial_amount - spent_amount) / game_cost = 9 :=
by sorry

end games_mike_can_buy_l1340_134058


namespace scout_cookies_unpacked_l1340_134069

/-- The number of boxes that cannot be fully packed into cases -/
def unpacked_boxes (total_boxes : ℕ) (boxes_per_case : ℕ) : ℕ :=
  total_boxes % boxes_per_case

/-- Proof that 7 boxes cannot be fully packed into cases -/
theorem scout_cookies_unpacked :
  unpacked_boxes 31 12 = 7 := by
  sorry

end scout_cookies_unpacked_l1340_134069


namespace population_ratio_theorem_l1340_134080

/-- Represents the population ratio in a town --/
structure PopulationRatio where
  men : ℝ
  women : ℝ
  children : ℝ
  elderly : ℝ

/-- The population ratio satisfies the given conditions --/
def satisfiesConditions (p : PopulationRatio) : Prop :=
  p.women = 0.9 * p.men ∧
  p.children = 0.6 * (p.men + p.women) ∧
  p.elderly = 0.25 * (p.women + p.children)

/-- The theorem stating the ratio of men to the combined population of others --/
theorem population_ratio_theorem (p : PopulationRatio) 
  (h : satisfiesConditions p) : 
  p.men / (p.women + p.children + p.elderly) = 1 / 2.55 := by
  sorry

#check population_ratio_theorem

end population_ratio_theorem_l1340_134080


namespace binomial_product_equals_6720_l1340_134051

theorem binomial_product_equals_6720 : Nat.choose 10 3 * Nat.choose 8 3 = 6720 := by
  sorry

end binomial_product_equals_6720_l1340_134051


namespace complement_of_A_l1340_134043

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}

theorem complement_of_A (x : ℝ) : x ∈ (Set.compl A) ↔ x ∈ Set.Icc (-1) 3 := by
  sorry

end complement_of_A_l1340_134043


namespace password_decryption_probability_l1340_134073

theorem password_decryption_probability :
  let p_a : ℝ := 1/5  -- Probability of A's success
  let p_b : ℝ := 1/3  -- Probability of B's success
  let p_c : ℝ := 1/4  -- Probability of C's success
  let p_success : ℝ := 1 - (1 - p_a) * (1 - p_b) * (1 - p_c)  -- Probability of successful decryption
  p_success = 3/5 := by
  sorry

end password_decryption_probability_l1340_134073


namespace divisibility_puzzle_l1340_134006

theorem divisibility_puzzle :
  ∃ N : ℕ, (N % 2 = 0) ∧ (N % 4 = 0) ∧ (N % 12 = 0) ∧ (N % 24 ≠ 0) :=
by
  sorry

end divisibility_puzzle_l1340_134006


namespace complex_number_in_third_quadrant_l1340_134066

/-- The complex number z = i(-2-i) is located in the third quadrant of the complex plane. -/
theorem complex_number_in_third_quadrant : 
  let z : ℂ := Complex.I * (-2 - Complex.I)
  (z.re < 0) ∧ (z.im < 0) := by
  sorry

end complex_number_in_third_quadrant_l1340_134066


namespace geometric_sequence_arithmetic_means_l1340_134095

theorem geometric_sequence_arithmetic_means (a b c m n : ℝ) 
  (h1 : b^2 = a*c)  -- geometric sequence condition
  (h2 : m = (a + b) / 2)  -- arithmetic mean of a and b
  (h3 : n = (b + c) / 2)  -- arithmetic mean of b and c
  : a / m + c / n = 2 := by
  sorry

end geometric_sequence_arithmetic_means_l1340_134095


namespace f_property_l1340_134038

/-- Represents a number with k digits, all being 1 -/
def rep_ones (k : ℕ) : ℕ :=
  (10^k - 1) / 9

/-- The function f(x) = 9x^2 + 2x -/
def f (x : ℕ) : ℕ :=
  9 * x^2 + 2 * x

/-- Theorem stating the property of f for numbers with all digits being 1 -/
theorem f_property (k : ℕ) :
  f (rep_ones k) = rep_ones (2 * k) :=
sorry

end f_property_l1340_134038


namespace right_triangle_hypotenuse_l1340_134009

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- sides are positive
  a^2 + b^2 = c^2 →  -- right-angled triangle (Pythagorean theorem)
  a^2 + b^2 + c^2 = 1800 →  -- sum of squares of all sides
  c = 30 := by sorry

end right_triangle_hypotenuse_l1340_134009


namespace triangle_inequality_l1340_134084

/-- Given that a, b, and c are the side lengths of a triangle, 
    prove that a^2(b+c-a) + b^2(c+a-b) + c^2(a+b-c) ≤ 3abc -/
theorem triangle_inequality (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end triangle_inequality_l1340_134084


namespace number_satisfying_equations_l1340_134059

theorem number_satisfying_equations (x : ℝ) : 
  16 * x = 3408 ∧ 1.6 * x = 340.8 → x = 213 := by
  sorry

end number_satisfying_equations_l1340_134059


namespace python_eating_theorem_l1340_134079

/-- Represents the eating rate of a python in terms of days per alligator --/
structure PythonEatingRate where
  days_per_alligator : ℕ

/-- Calculates the number of alligators a python can eat in a given number of days --/
def alligators_eaten (rate : PythonEatingRate) (days : ℕ) : ℕ :=
  days / rate.days_per_alligator

/-- The total number of alligators eaten by all pythons --/
def total_alligators_eaten (p1 p2 p3 : PythonEatingRate) (days : ℕ) : ℕ :=
  alligators_eaten p1 days + alligators_eaten p2 days + alligators_eaten p3 days

theorem python_eating_theorem (p1 p2 p3 : PythonEatingRate) 
  (h1 : p1.days_per_alligator = 7)  -- P1 eats one alligator per week
  (h2 : p2.days_per_alligator = 5)  -- P2 eats one alligator every 5 days
  (h3 : p3.days_per_alligator = 10) -- P3 eats one alligator every 10 days
  : total_alligators_eaten p1 p2 p3 21 = 9 := by
  sorry

#check python_eating_theorem

end python_eating_theorem_l1340_134079


namespace find_m_l1340_134039

theorem find_m (x₁ x₂ m : ℝ) 
  (h1 : x₁^2 - 3*x₁ + m = 0) 
  (h2 : x₂^2 - 3*x₂ + m = 0)
  (h3 : x₁ + x₂ - x₁*x₂ = 1) : 
  m = 2 := by
sorry

end find_m_l1340_134039


namespace fifteenth_term_of_specific_sequence_l1340_134077

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 15th term of the specific arithmetic sequence -/
theorem fifteenth_term_of_specific_sequence (a : ℕ → ℝ) 
    (h_arithmetic : ArithmeticSequence a)
    (h_first : a 1 = 3)
    (h_second : a 2 = 15)
    (h_third : a 3 = 27) :
  a 15 = 171 := by
  sorry

end fifteenth_term_of_specific_sequence_l1340_134077


namespace max_area_region_T_l1340_134056

/-- A configuration of four circles tangent to a line -/
structure CircleConfiguration where
  radii : Fin 4 → ℝ
  tangent_point : ℝ × ℝ
  line : Set (ℝ × ℝ)

/-- The region T formed by points inside exactly one circle -/
def region_T (config : CircleConfiguration) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The theorem stating the maximum area of region T -/
theorem max_area_region_T :
  ∃ (config : CircleConfiguration),
    (config.radii 0 = 2) ∧
    (config.radii 1 = 4) ∧
    (config.radii 2 = 6) ∧
    (config.radii 3 = 8) ∧
    (∀ (other_config : CircleConfiguration),
      (other_config.radii 0 = 2) →
      (other_config.radii 1 = 4) →
      (other_config.radii 2 = 6) →
      (other_config.radii 3 = 8) →
      area (region_T config) ≥ area (region_T other_config)) ∧
    area (region_T config) = 84 * Real.pi :=
  sorry

end max_area_region_T_l1340_134056


namespace cuboid_edge_sum_l1340_134094

/-- The sum of the lengths of the edges of a cuboid -/
def sumOfEdges (width length height : ℝ) : ℝ :=
  4 * (width + length + height)

/-- Theorem: The sum of the lengths of the edges of a cuboid with
    width 10 cm, length 8 cm, and height 5 cm is equal to 92 cm -/
theorem cuboid_edge_sum :
  sumOfEdges 10 8 5 = 92 := by
  sorry

end cuboid_edge_sum_l1340_134094


namespace solution_sets_equality_l1340_134020

-- Define the parameter a
def a : ℝ := 1

-- Define the solution set of ax - 1 > 0
def solution_set_1 : Set ℝ := {x | x > 1}

-- Define the solution set of (ax-1)(x+2) ≥ 0
def solution_set_2 : Set ℝ := {x | x ≤ -2 ∨ x ≥ 1}

-- State the theorem
theorem solution_sets_equality (h : solution_set_1 = {x | x > 1}) : 
  solution_set_2 = {x | x ≤ -2 ∨ x ≥ 1} := by
  sorry

end solution_sets_equality_l1340_134020


namespace simple_interest_problem_l1340_134089

/-- Represents a date with year, month, and day components. -/
structure Date where
  year : Nat
  month : Nat
  day : Nat

/-- Calculates the ending date given the start date and time period. -/
def calculateEndDate (startDate : Date) (timePeriod : Rat) : Date :=
  sorry

/-- Calculates the time period in years given principal, rate, and interest. -/
def calculateTimePeriod (principal : Rat) (rate : Rat) (interest : Rat) : Rat :=
  sorry

theorem simple_interest_problem (principal : Rat) (rate : Rat) (startDate : Date) (interest : Rat) :
  principal = 2000 →
  rate = 25 / (4 * 100) →
  startDate = ⟨2005, 2, 4⟩ →
  interest = 25 →
  let timePeriod := calculateTimePeriod principal rate interest
  let endDate := calculateEndDate startDate timePeriod
  endDate = ⟨2005, 4, 16⟩ :=
by
  sorry

end simple_interest_problem_l1340_134089


namespace candy_distribution_l1340_134018

theorem candy_distribution (total : Nat) (friends : Nat) (to_remove : Nat) : 
  total = 47 → friends = 5 → to_remove = 2 → 
  to_remove = (total % friends) ∧ 
  ∀ k : Nat, k < to_remove → (total - k) % friends ≠ 0 := by
sorry

end candy_distribution_l1340_134018


namespace smallest_number_l1340_134097

theorem smallest_number (a b c d : Int) (h1 : a = 2023) (h2 : b = 2022) (h3 : c = -2023) (h4 : d = -2022) :
  c ≤ a ∧ c ≤ b ∧ c ≤ d :=
by sorry

end smallest_number_l1340_134097


namespace right_triangle_median_property_l1340_134054

theorem right_triangle_median_property (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_median : (c/2)^2 = a*b) : c/2 = (c/2) :=
by sorry

end right_triangle_median_property_l1340_134054


namespace shaded_area_theorem_l1340_134012

def U : Set Nat := {1,2,3,4,5,6,7,8}
def M : Set Nat := {1,3,5,7}
def N : Set Nat := {5,6,7}

theorem shaded_area_theorem : U \ (M ∪ N) = {2,4,8} := by sorry

end shaded_area_theorem_l1340_134012


namespace circle_theorem_l1340_134005

-- Define the two given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 4 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*y - 28 = 0

-- Define the line on which the center lies
def centerLine (x y : ℝ) : Prop := x - y - 4 = 0

-- Define the resulting circle
def resultCircle (x y : ℝ) : Prop := (x - 1/2)^2 + (y + 7/2)^2 = 89/2

-- Theorem statement
theorem circle_theorem :
  ∃ (x1 y1 x2 y2 : ℝ),
    -- Intersection points of the two given circles
    circle1 x1 y1 ∧ circle2 x1 y1 ∧
    circle1 x2 y2 ∧ circle2 x2 y2 ∧
    -- The resulting circle passes through these intersection points
    resultCircle x1 y1 ∧ resultCircle x2 y2 ∧
    -- The center of the resulting circle lies on the given line
    centerLine (1/2) (-7/2) :=
by
  sorry

end circle_theorem_l1340_134005


namespace isosceles_triangle_perimeter_l1340_134014

def is_root (x : ℝ) : Prop := x^2 - 5*x + 6 = 0

theorem isosceles_triangle_perimeter : 
  ∀ (leg : ℝ), 
  is_root leg → 
  leg > 0 → 
  leg + leg > 4 → 
  leg + leg + 4 = 10 := by
sorry

end isosceles_triangle_perimeter_l1340_134014


namespace cubic_equation_coefficient_sum_of_squares_l1340_134017

theorem cubic_equation_coefficient_sum_of_squares :
  ∀ (p q r s t u : ℤ),
  (∀ x, 1728 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 23456 := by
sorry

end cubic_equation_coefficient_sum_of_squares_l1340_134017


namespace downstream_distance_l1340_134086

-- Define the given conditions
def boat_speed : ℝ := 16
def stream_speed : ℝ := 5
def time_downstream : ℝ := 7

-- Define the theorem
theorem downstream_distance :
  let effective_speed := boat_speed + stream_speed
  effective_speed * time_downstream = 147 :=
by sorry

end downstream_distance_l1340_134086


namespace ball_problem_l1340_134011

/-- Given the conditions of the ball problem, prove that the number of red, yellow, and white balls is (45, 40, 75). -/
theorem ball_problem (red yellow white : ℕ) : 
  (red + yellow + white = 160) →
  (2 * red / 3 + 3 * yellow / 4 + 4 * white / 5 = 120) →
  (4 * red / 5 + 3 * yellow / 4 + 2 * white / 3 = 116) →
  (red = 45 ∧ yellow = 40 ∧ white = 75) := by
sorry

end ball_problem_l1340_134011


namespace alternating_color_probability_l1340_134093

def total_balls : ℕ := 10
def white_balls : ℕ := 5
def black_balls : ℕ := 5

def alternating_sequences : ℕ := 2

def total_arrangements : ℕ := Nat.choose total_balls white_balls

theorem alternating_color_probability :
  (alternating_sequences : ℚ) / total_arrangements = 1 / 126 :=
sorry

end alternating_color_probability_l1340_134093


namespace jame_gold_bars_l1340_134024

/-- The number of gold bars Jame has left after tax and divorce -/
def gold_bars_left (initial : ℕ) (tax_rate : ℚ) (divorce_loss : ℚ) : ℕ :=
  let after_tax := initial - (initial * tax_rate).floor
  (after_tax - (after_tax * divorce_loss).floor).toNat

/-- Theorem stating that Jame has 27 gold bars left after tax and divorce -/
theorem jame_gold_bars :
  gold_bars_left 60 (1/10) (1/2) = 27 := by
  sorry

end jame_gold_bars_l1340_134024


namespace decreasing_quadratic_condition_l1340_134022

/-- A quadratic function f(x) = x^2 + ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 3

/-- The property of f being decreasing on (-∞, 2] -/
def is_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x < y → y ≤ 2 → f a x > f a y

/-- The main theorem: if f is decreasing on (-∞, 2], then a ≤ -4 -/
theorem decreasing_quadratic_condition (a : ℝ) :
  is_decreasing_on_interval a → a ≤ -4 := by sorry

end decreasing_quadratic_condition_l1340_134022


namespace sum_of_47_and_negative_27_l1340_134032

theorem sum_of_47_and_negative_27 : 47 + (-27) = 20 := by
  sorry

end sum_of_47_and_negative_27_l1340_134032


namespace smallest_integer_solution_smallest_integer_solution_exists_l1340_134007

theorem smallest_integer_solution (x : ℤ) : 
  (7 - 3 * x > 22) ∧ (x < 5) → x ≥ -6 :=
by
  sorry

theorem smallest_integer_solution_exists : 
  ∃ x : ℤ, (7 - 3 * x > 22) ∧ (x < 5) ∧ (x = -6) :=
by
  sorry

end smallest_integer_solution_smallest_integer_solution_exists_l1340_134007


namespace fraction_to_decimal_l1340_134085

theorem fraction_to_decimal : (59 : ℚ) / (2^2 * 5^7) = (1888 : ℚ) / 10^7 := by sorry

end fraction_to_decimal_l1340_134085


namespace square_difference_pattern_l1340_134087

theorem square_difference_pattern (n : ℕ) : (2*n + 1)^2 - (2*n - 1)^2 = 8*n := by
  sorry

end square_difference_pattern_l1340_134087


namespace simplify_expression_l1340_134033

theorem simplify_expression : 
  Real.sqrt 6 * 6^(1/2) + 18 / 3 * 4 - (2 + 2)^(5/2) = -2 := by
  sorry

end simplify_expression_l1340_134033


namespace max_diff_correct_l1340_134083

/-- A convex N-gon divided into triangles by non-intersecting diagonals -/
structure ConvexNGon (N : ℕ) where
  triangles : ℕ
  diagonals : ℕ
  triangles_eq : triangles = N - 2
  diagonals_eq : diagonals = N - 3

/-- Coloring of triangles in black and white -/
structure Coloring (N : ℕ) where
  ngon : ConvexNGon N
  white : ℕ
  black : ℕ
  sum_eq : white + black = ngon.triangles
  adjacent_diff : white ≠ black → white > black

/-- Maximum difference between white and black triangles -/
def max_diff (N : ℕ) : ℕ :=
  if N % 3 = 1 then N / 3 - 1 else N / 3

theorem max_diff_correct (N : ℕ) (c : Coloring N) :
  c.white - c.black ≤ max_diff N :=
sorry

end max_diff_correct_l1340_134083


namespace total_pizza_slices_l1340_134061

theorem total_pizza_slices :
  let num_pizzas : ℕ := 21
  let slices_per_pizza : ℕ := 8
  num_pizzas * slices_per_pizza = 168 := by
  sorry

end total_pizza_slices_l1340_134061


namespace simplify_and_evaluate_l1340_134028

theorem simplify_and_evaluate (a : ℤ) : 
  2 * (4 * a ^ 2 - a) - (3 * a ^ 2 - 2 * a + 5) = 40 ↔ a = -3 :=
by sorry

end simplify_and_evaluate_l1340_134028


namespace problem_solution_l1340_134099

noncomputable def f (a k x : ℝ) : ℝ := a^x - (k-1) * a^(-x)

theorem problem_solution (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  -- Part 1: k = 2
  (∃ k : ℝ, ∀ x : ℝ, f a k x = -f a k (-x)) ∧
  -- Part 2: f is monotonically decreasing
  (f a 2 1 < 0 → ∀ x y : ℝ, x < y → f a 2 x > f a 2 y) ∧
  -- Part 3: range of t
  (∃ t1 t2 : ℝ, t1 = -3 ∧ t2 = 5 ∧
    ∀ t : ℝ, (∀ x : ℝ, f a 2 (x^2 + t*x) + f a 2 (4-x) < 0) ↔ t1 < t ∧ t < t2) :=
by sorry

end problem_solution_l1340_134099


namespace license_plate_count_l1340_134076

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The number of digits in the license plate -/
def num_plate_digits : ℕ := 5

/-- The number of letters in the license plate -/
def num_plate_letters : ℕ := 2

/-- The number of possible positions for the letter block (start or end) -/
def num_letter_positions : ℕ := 2

/-- The total number of distinct license plates -/
def total_license_plates : ℕ := 
  num_digits ^ num_plate_digits * 
  num_letters ^ num_plate_letters * 
  num_letter_positions

theorem license_plate_count : total_license_plates = 2704000 := by
  sorry

end license_plate_count_l1340_134076


namespace meaningful_expression_range_l1340_134021

-- Define the set of x values that make the expression meaningful
def meaningful_x : Set ℝ := {x | x ≥ -5 ∧ x ≠ 0}

-- Theorem statement
theorem meaningful_expression_range : 
  {x : ℝ | (∃ y : ℝ, y = Real.sqrt (x + 5) / x) ∧ x ≠ 0} = meaningful_x := by
  sorry

end meaningful_expression_range_l1340_134021


namespace marys_marbles_l1340_134050

/-- Given that Mary has 9.0 yellow marbles initially and gives 3.0 yellow marbles to Joan,
    prove that Mary will have 6.0 yellow marbles left. -/
theorem marys_marbles (initial : ℝ) (given : ℝ) (left : ℝ) 
    (h1 : initial = 9.0) 
    (h2 : given = 3.0) 
    (h3 : left = initial - given) : 
  left = 6.0 := by
  sorry

end marys_marbles_l1340_134050


namespace abs_minus_three_minus_three_eq_zero_l1340_134040

theorem abs_minus_three_minus_three_eq_zero : |(-3 : ℤ)| - 3 = 0 := by sorry

end abs_minus_three_minus_three_eq_zero_l1340_134040


namespace expression_evaluation_l1340_134067

theorem expression_evaluation : 8 / 4 - 3^2 + 4 * 2 + Nat.factorial 5 = 121 := by
  sorry

end expression_evaluation_l1340_134067


namespace survey_is_simple_random_sampling_l1340_134082

/-- Represents a sampling method --/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic
  | ComplexRandom

/-- Represents a population of students --/
structure Population where
  size : Nat
  year : Nat

/-- Represents a sample from a population --/
structure Sample where
  size : Nat
  method : SamplingMethod

/-- Defines the conditions of the survey --/
def survey_conditions (pop : Population) (samp : Sample) : Prop :=
  pop.size = 200 ∧ pop.year = 1 ∧ samp.size = 20

/-- Theorem stating that the sampling method used is Simple Random Sampling --/
theorem survey_is_simple_random_sampling 
  (pop : Population) (samp : Sample) 
  (h : survey_conditions pop samp) : 
  samp.method = SamplingMethod.SimpleRandom := by
  sorry


end survey_is_simple_random_sampling_l1340_134082


namespace gcd_problem_l1340_134015

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * 1187 * k) :
  Nat.gcd (Int.natAbs (2 * b^2 + 31 * b + 67)) (Int.natAbs (b + 15)) = 1 := by
sorry

end gcd_problem_l1340_134015


namespace combined_average_age_l1340_134031

/-- Given two groups of people with their respective sizes and average ages,
    calculate the average age of all people combined. -/
theorem combined_average_age
  (size_a : ℕ) (avg_a : ℚ) (size_b : ℕ) (avg_b : ℚ)
  (h1 : size_a = 8)
  (h2 : avg_a = 45)
  (h3 : size_b = 6)
  (h4 : avg_b = 20) :
  (size_a : ℚ) * avg_a + (size_b : ℚ) * avg_b = 240 ∧
  (size_a : ℚ) + (size_b : ℚ) = 14 →
  (size_a : ℚ) * avg_a + (size_b : ℚ) * avg_b / ((size_a : ℚ) + (size_b : ℚ)) = 240 / 7 :=
by sorry

#check combined_average_age

end combined_average_age_l1340_134031


namespace total_books_on_shelves_l1340_134001

theorem total_books_on_shelves (num_shelves : ℕ) (books_per_shelf : ℕ) 
  (h1 : num_shelves = 150) 
  (h2 : books_per_shelf = 15) : 
  num_shelves * books_per_shelf = 2250 := by
sorry

end total_books_on_shelves_l1340_134001


namespace joint_order_savings_l1340_134072

/-- Represents the cost and discount structure for photocopies -/
structure PhotocopyOrder where
  cost_per_copy : ℚ
  discount_rate : ℚ
  discount_threshold : ℕ

/-- Calculates the total cost of an order with potential discount -/
def total_cost (order : PhotocopyOrder) (num_copies : ℕ) : ℚ :=
  let base_cost := order.cost_per_copy * num_copies
  if num_copies > order.discount_threshold then
    base_cost * (1 - order.discount_rate)
  else
    base_cost

/-- Theorem: Steve and David each save $0.40 by submitting a joint order -/
theorem joint_order_savings (steve_copies david_copies : ℕ) :
  let order := PhotocopyOrder.mk 0.02 0.25 100
  let individual_cost := total_cost order steve_copies
  let joint_copies := steve_copies + david_copies
  let joint_cost := total_cost order joint_copies
  steve_copies = 80 ∧ david_copies = 80 →
  individual_cost - (joint_cost / 2) = 0.40 := by
  sorry

end joint_order_savings_l1340_134072


namespace art_club_enrollment_l1340_134071

theorem art_club_enrollment (total : ℕ) (biology : ℕ) (chemistry : ℕ) (both : ℕ)
  (h1 : total = 80)
  (h2 : biology = 50)
  (h3 : chemistry = 40)
  (h4 : both = 30) :
  total - (biology + chemistry - both) = 20 := by
  sorry

end art_club_enrollment_l1340_134071


namespace evaluate_expression_l1340_134016

theorem evaluate_expression : 6 - 8 * (9 - 4^2) * 5 = 286 := by
  sorry

end evaluate_expression_l1340_134016


namespace team_point_difference_l1340_134074

/-- The difference in points between two teams -/
def point_difference (beth_score jan_score judy_score angel_score : ℕ) : ℕ :=
  (beth_score + jan_score) - (judy_score + angel_score)

/-- Theorem stating the point difference between the two teams -/
theorem team_point_difference :
  point_difference 12 10 8 11 = 3 := by
  sorry

end team_point_difference_l1340_134074


namespace reflection_result_l1340_134048

/-- Reflects a point over the y-axis -/
def reflectOverYAxis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Reflects a point over the x-axis -/
def reflectOverXAxis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The final position of point F after two reflections -/
def finalPosition (F : ℝ × ℝ) : ℝ × ℝ :=
  reflectOverXAxis (reflectOverYAxis F)

theorem reflection_result :
  finalPosition (-1, -1) = (1, 1) := by
  sorry

end reflection_result_l1340_134048


namespace sophomore_selection_l1340_134078

/-- Represents the number of students selected from a grade -/
structure GradeSelection where
  total : Nat
  selected : Nat

/-- Represents the stratified sampling of students across grades -/
structure StratifiedSampling where
  freshmen : GradeSelection
  sophomores : GradeSelection
  seniors : GradeSelection

/-- 
Given a stratified sampling where:
- There are 210 freshmen, 270 sophomores, and 300 seniors
- 7 freshmen were selected
- The same selection rate is applied across all grades

Prove that 9 sophomores were selected
-/
theorem sophomore_selection (s : StratifiedSampling) 
  (h1 : s.freshmen.total = 210)
  (h2 : s.sophomores.total = 270)
  (h3 : s.seniors.total = 300)
  (h4 : s.freshmen.selected = 7)
  (h5 : s.freshmen.selected * s.sophomores.total = s.sophomores.selected * s.freshmen.total) :
  s.sophomores.selected = 9 := by
  sorry


end sophomore_selection_l1340_134078


namespace polynomial_division_remainder_l1340_134000

/-- Given a polynomial division, prove that the remainder is 1 -/
theorem polynomial_division_remainder : 
  let P (z : ℝ) := 4 * z^3 - 5 * z^2 - 17 * z + 4
  let D (z : ℝ) := 4 * z + 6
  let Q (z : ℝ) := z^2 - 4 * z + 1/2
  ∃ (R : ℝ → ℝ), (∀ z, P z = D z * Q z + R z) ∧ (∀ z, R z = 1) :=
sorry

end polynomial_division_remainder_l1340_134000


namespace actual_toddler_count_l1340_134091

theorem actual_toddler_count (bill_count : ℕ) (double_counted : ℕ) (missed : ℕ) 
  (h1 : bill_count = 26) 
  (h2 : double_counted = 8) 
  (h3 : missed = 3) : 
  bill_count - double_counted + missed = 21 := by
  sorry

end actual_toddler_count_l1340_134091


namespace a_zero_necessary_not_sufficient_l1340_134068

/-- A complex number is purely imaginary if its real part is zero. -/
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0

/-- For a, b ∈ ℝ, "a = 0" is a necessary but not sufficient condition 
    for the complex number a + bi to be purely imaginary. -/
theorem a_zero_necessary_not_sufficient (a b : ℝ) :
  (is_purely_imaginary (Complex.mk a b) → a = 0) ∧
  ¬(a = 0 → is_purely_imaginary (Complex.mk a b)) :=
sorry

end a_zero_necessary_not_sufficient_l1340_134068
