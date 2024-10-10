import Mathlib

namespace f_is_even_and_increasing_l2860_286028

def f (x : ℝ) : ℝ := |x| + 1

theorem f_is_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end f_is_even_and_increasing_l2860_286028


namespace fuel_mixture_theorem_l2860_286070

/-- Represents the state of the fuel tank -/
structure TankState where
  z : Rat  -- Amount of brand Z gasoline
  y : Rat  -- Amount of brand Y gasoline

/-- Fills the tank with brand Z gasoline -/
def fill_z (s : TankState) : TankState :=
  { z := s.z + (1 - s.z - s.y), y := s.y }

/-- Fills the tank with brand Y gasoline -/
def fill_y (s : TankState) : TankState :=
  { z := s.z, y := s.y + (1 - s.z - s.y) }

/-- Removes half of the fuel from the tank -/
def half_empty (s : TankState) : TankState :=
  { z := s.z / 2, y := s.y / 2 }

theorem fuel_mixture_theorem : 
  let s0 : TankState := { z := 0, y := 0 }
  let s1 := fill_z s0
  let s2 := fill_y (TankState.mk (3/4) 0)
  let s3 := fill_z (half_empty s2)
  let s4 := fill_y (half_empty s3)
  s4.z = 7/16 := by sorry

end fuel_mixture_theorem_l2860_286070


namespace shopping_cost_after_discount_l2860_286056

/-- Calculate the total cost after discount for a shopping trip --/
theorem shopping_cost_after_discount :
  let tshirt_cost : ℕ := 20
  let pants_cost : ℕ := 80
  let shoes_cost : ℕ := 150
  let discount_rate : ℚ := 1 / 10
  let tshirt_quantity : ℕ := 4
  let pants_quantity : ℕ := 3
  let shoes_quantity : ℕ := 2
  let total_cost_before_discount : ℕ := 
    tshirt_cost * tshirt_quantity + 
    pants_cost * pants_quantity + 
    shoes_cost * shoes_quantity
  let discount_amount : ℚ := discount_rate * total_cost_before_discount
  let total_cost_after_discount : ℚ := total_cost_before_discount - discount_amount
  total_cost_after_discount = 558 := by sorry

end shopping_cost_after_discount_l2860_286056


namespace zero_points_count_l2860_286061

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem zero_points_count 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : has_period f π) 
  (h_shifted : ∀ x, f (x - π) = f (x + π) ∧ f (x - π) = f x) : 
  ∃ (S : Finset ℝ), S.card = 7 ∧ (∀ x ∈ S, f x = 0 ∧ x ∈ Set.Icc 0 8) ∧
    (∀ x ∈ Set.Icc 0 8, f x = 0 → x ∈ S) :=
sorry

end zero_points_count_l2860_286061


namespace binomial_20_19_l2860_286058

theorem binomial_20_19 : Nat.choose 20 19 = 20 := by sorry

end binomial_20_19_l2860_286058


namespace problem_solution_l2860_286087

theorem problem_solution (a n : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * n * 45 * 49) : n = 945 := by
  sorry

end problem_solution_l2860_286087


namespace solution_set_f_leq_x_range_of_a_l2860_286044

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 7| + 1

-- Theorem for the solution set of f(x) ≤ x
theorem solution_set_f_leq_x :
  {x : ℝ | f x ≤ x} = {x : ℝ | 8/3 ≤ x ∧ x ≤ 6} := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x - 2*|x - 1| ≤ a) ↔ a ≥ -4 := by sorry

end solution_set_f_leq_x_range_of_a_l2860_286044


namespace dodecahedron_triangles_l2860_286077

/-- Represents a dodecahedron -/
structure Dodecahedron where
  num_faces : ℕ
  faces_are_pentagonal : num_faces = 12
  vertices_per_face : ℕ
  vertices_shared_by_three_faces : vertices_per_face = 3

/-- Calculates the number of vertices in a dodecahedron -/
def num_vertices (d : Dodecahedron) : ℕ := 20

/-- Calculates the number of triangles that can be formed using the vertices of a dodecahedron -/
def num_triangles (d : Dodecahedron) : ℕ := (num_vertices d).choose 3

/-- Theorem: The number of triangles that can be formed using the vertices of a dodecahedron is 1140 -/
theorem dodecahedron_triangles (d : Dodecahedron) : num_triangles d = 1140 := by
  sorry

end dodecahedron_triangles_l2860_286077


namespace neil_charge_theorem_l2860_286030

def trim_cost : ℕ → ℝ := λ n => 5 * n
def shape_cost : ℕ → ℝ := λ n => 15 * n

theorem neil_charge_theorem (num_trim : ℕ) (num_shape : ℕ) 
  (h1 : num_trim = 30) (h2 : num_shape = 4) : 
  trim_cost num_trim + shape_cost num_shape = 210 := by
  sorry

end neil_charge_theorem_l2860_286030


namespace division_problem_l2860_286048

theorem division_problem (n : ℕ) : n % 21 = 1 ∧ n / 21 = 9 → n = 190 := by
  sorry

end division_problem_l2860_286048


namespace group_collection_theorem_l2860_286034

/-- Calculates the total collection amount in rupees for a group of students -/
def totalCollectionInRupees (groupSize : ℕ) : ℚ :=
  (groupSize * groupSize : ℚ) / 100

/-- Theorem: The total collection amount for a group of 45 students is 20.25 rupees -/
theorem group_collection_theorem :
  totalCollectionInRupees 45 = 20.25 := by
  sorry

#eval totalCollectionInRupees 45

end group_collection_theorem_l2860_286034


namespace cubic_sum_in_terms_of_products_l2860_286019

theorem cubic_sum_in_terms_of_products (x y z p q r : ℝ) 
  (h_xy : x * y = p)
  (h_xz : x * z = q)
  (h_yz : y * z = r)
  (h_x_nonzero : x ≠ 0)
  (h_y_nonzero : y ≠ 0)
  (h_z_nonzero : z ≠ 0) :
  x^3 + y^3 + z^3 = (p^2 * q^2 + p^2 * r^2 + q^2 * r^2) / (p * q * r) :=
by sorry

end cubic_sum_in_terms_of_products_l2860_286019


namespace root_sum_absolute_value_l2860_286075

theorem root_sum_absolute_value (m : ℤ) (p q r : ℤ) : 
  (∀ x : ℤ, x^3 - 2022*x + m = 0 ↔ x = p ∨ x = q ∨ x = r) →
  |p| + |q| + |r| = 104 := by
sorry

end root_sum_absolute_value_l2860_286075


namespace expression_evaluation_l2860_286002

theorem expression_evaluation (a b c : ℝ) 
  (ha : a = 3) (hb : b = 2) (hc : c = 1) : 
  (a^2 + b + c)^2 - (a^2 - b - c)^2 = 108 := by
  sorry

end expression_evaluation_l2860_286002


namespace p_sufficient_not_necessary_for_q_l2860_286017

theorem p_sufficient_not_necessary_for_q :
  ∃ (p q : Prop),
    (p ↔ (∃ x : ℝ, x = 2)) ∧
    (q ↔ (∃ x : ℝ, (x - 2) * (x + 3) = 0)) ∧
    (p → q) ∧
    ¬(q → p) :=
by sorry

end p_sufficient_not_necessary_for_q_l2860_286017


namespace unique_prime_triple_l2860_286082

theorem unique_prime_triple (p : ℕ) : 
  Prime p ∧ Prime (2 * p + 1) ∧ Prime (4 * p + 1) ↔ p = 3 :=
by sorry

end unique_prime_triple_l2860_286082


namespace unknown_number_value_l2860_286007

theorem unknown_number_value (a x : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * x * 35 * 63) : x = 25 := by
  sorry

end unknown_number_value_l2860_286007


namespace pawpaw_count_l2860_286049

/-- Represents the contents of fruit baskets -/
structure FruitBaskets where
  total_fruits : Nat
  num_baskets : Nat
  mangoes : Nat
  pears : Nat
  lemons : Nat
  kiwi : Nat
  pawpaws : Nat

/-- Theorem stating the number of pawpaws in one basket -/
theorem pawpaw_count (fb : FruitBaskets) 
  (h1 : fb.total_fruits = 58)
  (h2 : fb.num_baskets = 5)
  (h3 : fb.mangoes = 18)
  (h4 : fb.pears = 10)
  (h5 : fb.lemons = 9)
  (h6 : fb.kiwi = fb.lemons)
  (h7 : fb.total_fruits = fb.mangoes + fb.pears + fb.lemons + fb.kiwi + fb.pawpaws) :
  fb.pawpaws = 12 := by
  sorry

end pawpaw_count_l2860_286049


namespace inequality_solution_implies_a_bound_l2860_286006

theorem inequality_solution_implies_a_bound (a : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2*x^2 - 8*x - 4 - a > 0) → a < -4 :=
by sorry

end inequality_solution_implies_a_bound_l2860_286006


namespace symmetric_circle_l2860_286054

/-- Given a point P(a, b) symmetric to line l with symmetric point P'(b + 1, a - 1),
    and a circle C with equation x^2 + y^2 - 6x - 2y = 0,
    prove that the equation of the circle C' symmetric to C with respect to line l
    is (x - 2)^2 + (y - 2)^2 = 10 -/
theorem symmetric_circle (a b : ℝ) :
  let P : ℝ × ℝ := (a, b)
  let P' : ℝ × ℝ := (b + 1, a - 1)
  let C (x y : ℝ) := x^2 + y^2 - 6*x - 2*y = 0
  let C' (x y : ℝ) := (x - 2)^2 + (y - 2)^2 = 10
  (∀ x y, C x y ↔ C' y x) := by
  sorry

end symmetric_circle_l2860_286054


namespace complex_set_sum_l2860_286094

def is_closed_under_multiplication (S : Set ℂ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → (x * y) ∈ S

theorem complex_set_sum (a b c d : ℂ) :
  let S : Set ℂ := {a, b, c, d}
  is_closed_under_multiplication S →
  a^2 = 1 →
  b = 1 →
  c^2 = a →
  b + c + d = 1 := by
sorry

end complex_set_sum_l2860_286094


namespace discriminant_nonnegativity_l2860_286039

theorem discriminant_nonnegativity (x : ℤ) :
  x^2 * (49 - 40 * x^2) ≥ 0 ↔ x = 0 ∨ x = 1 ∨ x = -1 := by
  sorry

end discriminant_nonnegativity_l2860_286039


namespace common_divisor_and_remainder_l2860_286042

theorem common_divisor_and_remainder (a b c d : ℕ) : 
  a = 2613 ∧ b = 2243 ∧ c = 1503 ∧ d = 985 →
  ∃ (k : ℕ), k > 0 ∧ 
    k ∣ (a - b) ∧ k ∣ (b - c) ∧ k ∣ (c - d) ∧
    ∀ m : ℕ, m > k → ¬(m ∣ (a - b) ∧ m ∣ (b - c) ∧ m ∣ (c - d)) ∧
    a % k = b % k ∧ b % k = c % k ∧ c % k = d % k ∧
    k = 74 ∧ a % k = 23 :=
by sorry

end common_divisor_and_remainder_l2860_286042


namespace book_prices_l2860_286015

def total_cost : ℕ := 104

def is_valid_price (price : ℕ) : Prop :=
  ∃ (n : ℕ), 10 < n ∧ n < 60 ∧ n * price = total_cost

theorem book_prices :
  {p : ℕ | is_valid_price p} = {2, 4, 8} :=
by sorry

end book_prices_l2860_286015


namespace square_of_cube_of_smallest_prime_l2860_286098

def smallest_prime : ℕ := 2

theorem square_of_cube_of_smallest_prime : 
  (smallest_prime ^ 3) ^ 2 = 64 := by
  sorry

end square_of_cube_of_smallest_prime_l2860_286098


namespace area_DEF_eq_sum_of_parts_l2860_286089

/-- Represents a triangle with an area -/
structure Triangle :=
  (area : ℝ)

/-- Represents the main triangle DEF -/
def DEF : Triangle := sorry

/-- Represents the point Q inside triangle DEF -/
def Q : Point := sorry

/-- Represents the three smaller triangles created by lines through Q -/
def u₁ : Triangle := { area := 16 }
def u₂ : Triangle := { area := 25 }
def u₃ : Triangle := { area := 36 }

/-- The theorem stating that the area of DEF is the sum of areas of u₁, u₂, and u₃ -/
theorem area_DEF_eq_sum_of_parts : DEF.area = u₁.area + u₂.area + u₃.area := by
  sorry

#check area_DEF_eq_sum_of_parts

end area_DEF_eq_sum_of_parts_l2860_286089


namespace triangle_formation_l2860_286025

/-- Triangle inequality check for three sides -/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if a set of three numbers can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ satisfies_triangle_inequality a b c

theorem triangle_formation :
  can_form_triangle 7 12 17 ∧
  ¬ can_form_triangle 3 3 7 ∧
  ¬ can_form_triangle 4 5 9 ∧
  ¬ can_form_triangle 5 8 15 :=
by sorry

end triangle_formation_l2860_286025


namespace polygon_sides_l2860_286095

theorem polygon_sides (n : ℕ) : n > 2 →
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 := by
  sorry

end polygon_sides_l2860_286095


namespace largest_three_digit_square_base7_l2860_286073

/-- The number of digits of a natural number in base 7 -/
def numDigitsBase7 (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log 7 n + 1

/-- Conversion from base 10 to base 7 -/
def toBase7 (n : ℕ) : ℕ :=
  sorry

theorem largest_three_digit_square_base7 :
  ∃ N : ℕ, N = 45 ∧
  (∀ m : ℕ, m > N → numDigitsBase7 (m^2) > 3) ∧
  numDigitsBase7 (N^2) = 3 :=
sorry

end largest_three_digit_square_base7_l2860_286073


namespace chord_segment_lengths_l2860_286076

theorem chord_segment_lengths (r : ℝ) (chord_length : ℝ) :
  r = 7 ∧ chord_length = 12 →
  ∃ (ak kb : ℝ),
    ak = 7 - Real.sqrt 13 ∧
    kb = 7 + Real.sqrt 13 ∧
    ak + kb = 2 * r :=
by sorry

end chord_segment_lengths_l2860_286076


namespace initial_average_weight_l2860_286086

theorem initial_average_weight (n : ℕ) (A : ℝ) : 
  (n * A + 90 = (n + 1) * (A - 1)) ∧ 
  (n * A + 110 = (n + 1) * (A + 4)) →
  A = 94 := by
  sorry

end initial_average_weight_l2860_286086


namespace range_of_g_on_large_interval_l2860_286064

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def range_of (f : ℝ → ℝ) (a b : ℝ) : Set ℝ :=
  {y | ∃ x ∈ Set.Icc a b, f x = y}

theorem range_of_g_on_large_interval
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h_periodic : is_periodic f 1)
  (h_g_def : ∀ x, g x = f x + 2 * x)
  (h_range_small : range_of g 1 2 = Set.Icc (-1) 5) :
  range_of g (-2020) 2020 = Set.Icc (-4043) 4041 := by
sorry

end range_of_g_on_large_interval_l2860_286064


namespace composite_sum_of_power_l2860_286057

theorem composite_sum_of_power (n : ℕ) (h : n > 1) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^4 + 4^n = a * b :=
sorry

end composite_sum_of_power_l2860_286057


namespace flashlight_distance_difference_l2860_286055

/-- The visibility distance of Veronica's flashlight in feet -/
def veronica_distance : ℕ := 1000

/-- The visibility distance of Freddie's flashlight in feet -/
def freddie_distance : ℕ := 3 * veronica_distance

/-- The visibility distance of Velma's flashlight in feet -/
def velma_distance : ℕ := 5 * freddie_distance - 2000

/-- The difference in visibility distance between Velma's and Veronica's flashlights -/
theorem flashlight_distance_difference : velma_distance - veronica_distance = 12000 := by
  sorry

end flashlight_distance_difference_l2860_286055


namespace y_intercept_of_line_l2860_286047

/-- The y-intercept of the line 6x + 10y = 40 is (0, 4) -/
theorem y_intercept_of_line (x y : ℝ) : 6 * x + 10 * y = 40 → x = 0 → y = 4 := by
  sorry

end y_intercept_of_line_l2860_286047


namespace harmonic_sum_identity_l2860_286005

def h (n : ℕ) : ℚ :=
  (Finset.range n).sum (fun i => 1 / (i + 1 : ℚ))

theorem harmonic_sum_identity (n : ℕ) (h_n : n ≥ 2) :
  (n : ℚ) + (Finset.range (n - 1)).sum h = n * h n :=
by sorry

end harmonic_sum_identity_l2860_286005


namespace large_sphere_radius_l2860_286012

theorem large_sphere_radius (n : ℕ) (r : ℝ) (R : ℝ) : 
  n = 12 → r = 2 → (4 / 3 * Real.pi * R^3) = n * (4 / 3 * Real.pi * r^3) → R = (96 : ℝ)^(1/3) :=
sorry

end large_sphere_radius_l2860_286012


namespace teds_age_l2860_286084

theorem teds_age (ted sally : ℝ) 
  (h1 : ted = 3 * sally - 20) 
  (h2 : ted + sally = 78) : 
  ted = 53.5 := by
sorry

end teds_age_l2860_286084


namespace toby_friends_count_l2860_286097

theorem toby_friends_count (total : ℕ) (boys girls : ℕ) : 
  (boys : ℚ) / total = 55 / 100 →
  (girls : ℚ) / total = 30 / 100 →
  boys = 33 →
  girls = 18 :=
by sorry

end toby_friends_count_l2860_286097


namespace recurring_decimal_ratio_l2860_286078

-- Define the recurring decimals
def recurring_81 : ℚ := 81 / 99
def recurring_54 : ℚ := 54 / 99

-- State the theorem
theorem recurring_decimal_ratio :
  recurring_81 / recurring_54 = 3 / 2 := by
  sorry

end recurring_decimal_ratio_l2860_286078


namespace fifteen_people_handshakes_l2860_286085

/-- The number of handshakes in a group where each person shakes hands once with every other person -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a group of 15 people, where each person shakes hands exactly once with every other person, the total number of handshakes is 105 -/
theorem fifteen_people_handshakes :
  handshakes 15 = 105 := by
  sorry

end fifteen_people_handshakes_l2860_286085


namespace common_points_characterization_l2860_286001

-- Define the square S
def S : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define the set C_t
def C (t : ℝ) : Set (ℝ × ℝ) :=
  {p ∈ S | p.2 ≥ ((1 - t) / t) * p.1 + (1 - t)}

-- Define the intersection of all C_t
def CommonPoints : Set (ℝ × ℝ) :=
  ⋂ t ∈ {t | 0 < t ∧ t < 1}, C t

-- State the theorem
theorem common_points_characterization :
  ∀ p ∈ S, p ∈ CommonPoints ↔ Real.sqrt p.1 + Real.sqrt p.2 ≥ 1 := by sorry

end common_points_characterization_l2860_286001


namespace sqrt_of_sqrt_81_plus_sqrt_81_over_2_l2860_286029

theorem sqrt_of_sqrt_81_plus_sqrt_81_over_2 : 
  Real.sqrt ((Real.sqrt 81 + Real.sqrt 81) / 2) = 3 := by sorry

end sqrt_of_sqrt_81_plus_sqrt_81_over_2_l2860_286029


namespace nala_seashells_l2860_286093

/-- The number of seashells Nala found on the first day -/
def first_day : ℕ := 5

/-- The total number of seashells Nala has -/
def total : ℕ := 36

/-- The number of seashells Nala found on the second day -/
def second_day : ℕ := 7

theorem nala_seashells : 
  first_day + second_day + 2 * (first_day + second_day) = total := by
  sorry

#check nala_seashells

end nala_seashells_l2860_286093


namespace negation_of_universal_statement_l2860_286092

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 > 0) := by sorry

end negation_of_universal_statement_l2860_286092


namespace vet_donation_l2860_286036

theorem vet_donation (dog_fee : ℕ) (cat_fee : ℕ) (dog_adoptions : ℕ) (cat_adoptions : ℕ) 
  (h1 : dog_fee = 15)
  (h2 : cat_fee = 13)
  (h3 : dog_adoptions = 8)
  (h4 : cat_adoptions = 3) :
  (dog_fee * dog_adoptions + cat_fee * cat_adoptions) / 3 = 53 := by
  sorry

end vet_donation_l2860_286036


namespace arithmetic_grid_solution_l2860_286022

/-- Represents a 7x1 arithmetic sequence -/
def RowSequence := Fin 7 → ℤ

/-- Represents a 4x1 arithmetic sequence -/
def ColumnSequence := Fin 4 → ℤ

/-- The problem setup -/
structure ArithmeticGrid :=
  (row : RowSequence)
  (col1 : ColumnSequence)
  (col2 : ColumnSequence)
  (is_arithmetic_row : ∀ i j k : Fin 7, i.val + 1 = j.val ∧ j.val + 1 = k.val → 
    row j - row i = row k - row j)
  (is_arithmetic_col1 : ∀ i j k : Fin 4, i.val + 1 = j.val ∧ j.val + 1 = k.val → 
    col1 j - col1 i = col1 k - col1 j)
  (is_arithmetic_col2 : ∀ i j k : Fin 4, i.val + 1 = j.val ∧ j.val + 1 = k.val → 
    col2 j - col2 i = col2 k - col2 j)
  (distinct_sequences : 
    (∀ i j : Fin 7, i ≠ j → row i - row j ≠ 0) ∧ 
    (∀ i j : Fin 4, i ≠ j → col1 i - col1 j ≠ 0) ∧ 
    (∀ i j : Fin 4, i ≠ j → col2 i - col2 j ≠ 0))
  (top_left : row 0 = 25)
  (middle_column : col1 1 = 12 ∧ col1 2 = 16)
  (bottom_right : col2 3 = -13)

/-- The main theorem -/
theorem arithmetic_grid_solution (grid : ArithmeticGrid) : grid.col2 0 = -16 := by
  sorry

end arithmetic_grid_solution_l2860_286022


namespace circle_properties_l2860_286004

/-- The equation of a circle in the form ax² + by² + cx + dy + e = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center and radius of a circle -/
structure CircleProperties where
  center : ℝ × ℝ
  radius : ℝ

def circle_equation : CircleEquation :=
  { a := 1, b := 1, c := -2, d := 4, e := 3 }

theorem circle_properties :
  ∃ (props : CircleProperties),
    props.center = (1, -2) ∧ 
    props.radius = Real.sqrt 2 ∧
    ∀ (x y : ℝ),
      (circle_equation.a * x^2 + circle_equation.b * y^2 + 
       circle_equation.c * x + circle_equation.d * y + 
       circle_equation.e = 0) ↔
      ((x - props.center.1)^2 + (y - props.center.2)^2 = props.radius^2) :=
by sorry

end circle_properties_l2860_286004


namespace triangle_cosine_inequality_l2860_286038

theorem triangle_cosine_inequality (A B C : ℝ) (h_non_obtuse : 0 ≤ A ∧ 0 ≤ B ∧ 0 ≤ C ∧ A + B + C = π) :
  (1 - Real.cos (2 * A)) * (1 - Real.cos (2 * B)) / (1 - Real.cos (2 * C)) +
  (1 - Real.cos (2 * C)) * (1 - Real.cos (2 * A)) / (1 - Real.cos (2 * B)) +
  (1 - Real.cos (2 * B)) * (1 - Real.cos (2 * C)) / (1 - Real.cos (2 * A)) ≥ 9 / 2 := by
  sorry

end triangle_cosine_inequality_l2860_286038


namespace peter_glasses_purchase_l2860_286080

/-- Represents the purchase of glasses by Peter --/
def glassesPurchase (smallCost largeCost initialMoney smallCount change : ℕ) : Prop :=
  ∃ (largeCount : ℕ),
    smallCost * smallCount + largeCost * largeCount = initialMoney - change

theorem peter_glasses_purchase :
  glassesPurchase 3 5 50 8 1 →
  ∃ (largeCount : ℕ), largeCount = 5 ∧ glassesPurchase 3 5 50 8 1 := by
  sorry

end peter_glasses_purchase_l2860_286080


namespace most_likely_outcome_l2860_286096

def n : ℕ := 5

def p_boy : ℚ := 1/2
def p_girl : ℚ := 1/2

def prob_all_same_gender : ℚ := p_boy^n + p_girl^n

def prob_three_two : ℚ := (Nat.choose n 3) * (p_boy^3 * p_girl^2 + p_boy^2 * p_girl^3)

theorem most_likely_outcome :
  prob_three_two > prob_all_same_gender ∧
  prob_three_two = 5/16 :=
sorry

end most_likely_outcome_l2860_286096


namespace juice_bottle_savings_l2860_286027

/-- Represents the volume and cost of a juice bottle -/
structure Bottle :=
  (volume : ℕ)
  (cost : ℕ)

/-- Calculates the savings when buying a big bottle instead of equivalent small bottles -/
def calculate_savings (big : Bottle) (small : Bottle) : ℕ :=
  let small_bottles_needed := big.volume / small.volume
  let small_bottles_cost := small_bottles_needed * small.cost
  small_bottles_cost - big.cost

/-- Theorem stating the savings when buying a big bottle instead of equivalent small bottles -/
theorem juice_bottle_savings :
  let big_bottle := Bottle.mk 30 2700
  let small_bottle := Bottle.mk 6 600
  calculate_savings big_bottle small_bottle = 300 := by
sorry

end juice_bottle_savings_l2860_286027


namespace power_division_twentythree_l2860_286088

theorem power_division_twentythree : (23 : ℕ)^11 / (23 : ℕ)^8 = 12167 := by sorry

end power_division_twentythree_l2860_286088


namespace vector_magnitude_l2860_286053

/-- Given two vectors a and b in R², if (a - 2b) is perpendicular to a, then the magnitude of b is √5. -/
theorem vector_magnitude (a b : ℝ × ℝ) (h : a = (-1, 3) ∧ b.1 = 1) :
  (a - 2 • b) • a = 0 → ‖b‖ = Real.sqrt 5 := by
  sorry

end vector_magnitude_l2860_286053


namespace ladder_wall_distance_l2860_286014

/-- The distance between two walls problem -/
theorem ladder_wall_distance
  (w : ℝ) -- Distance between walls
  (a : ℝ) -- Length of the ladder
  (k : ℝ) -- Height of point Q
  (h : ℝ) -- Height of point R
  (hw_pos : w > 0)
  (ha_pos : a > 0)
  (hk_pos : k > 0)
  (hh_pos : h > 0)
  (h_45_deg : a = k * Real.sqrt 2) -- Condition for 45° angle
  (h_75_deg : a = h * Real.sqrt (4 - 2 * Real.sqrt 3)) -- Condition for 75° angle
  : w = h :=
by sorry

end ladder_wall_distance_l2860_286014


namespace range_of_a_given_false_proposition_l2860_286032

theorem range_of_a_given_false_proposition : 
  (¬ ∃ x : ℝ, 2 * x^2 + (a - 1) * x + 1/2 ≤ 0) → 
  (∀ a : ℝ, -1 < a ∧ a < 3) :=
by sorry

end range_of_a_given_false_proposition_l2860_286032


namespace divisor_totient_sum_theorem_l2860_286021

def divisor_count (n : ℕ) : ℕ := (Nat.divisors n).card

theorem divisor_totient_sum_theorem (n : ℕ) (c : ℕ) :
  (n > 0) →
  (divisor_count n + Nat.totient n = n + c) ↔
  ((c = 1 ∧ (n = 1 ∨ Nat.Prime n ∨ n = 4)) ∨
   (c = 0 ∧ (n = 6 ∨ n = 8 ∨ n = 9))) :=
by sorry

end divisor_totient_sum_theorem_l2860_286021


namespace even_function_symmetry_is_universal_l2860_286008

-- Define what a universal proposition is
def is_universal_proposition (p : Prop) : Prop :=
  ∃ (U : Type) (P : U → Prop), p = ∀ (x : U), P x

-- Define what an even function is
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Define symmetry about y-axis for a function's graph
def symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- State the theorem
theorem even_function_symmetry_is_universal :
  is_universal_proposition (∀ f : ℝ → ℝ, is_even_function f → symmetric_about_y_axis f) :=
sorry

end even_function_symmetry_is_universal_l2860_286008


namespace exists_distinct_subaveraging_equal_entries_imply_infinite_equal_pairs_l2860_286003

-- Define a subaveraging sequence
def IsSubaveraging (s : ℤ → ℝ) : Prop :=
  ∀ n : ℤ, s n = (s (n - 1) + s (n + 1)) / 4

-- Part (a): Existence of a subaveraging sequence with all distinct entries
theorem exists_distinct_subaveraging :
  ∃ s : ℤ → ℝ, IsSubaveraging s ∧ (∀ m n : ℤ, m ≠ n → s m ≠ s n) :=
sorry

-- Part (b): If two entries are equal, infinitely many pairs are equal
theorem equal_entries_imply_infinite_equal_pairs
  (s : ℤ → ℝ) (h : IsSubaveraging s) :
  (∃ m n : ℤ, m ≠ n ∧ s m = s n) →
  (∀ k : ℕ, ∃ i j : ℤ, i ≠ j ∧ s i = s j ∧ |i - j| > k) :=
sorry

end exists_distinct_subaveraging_equal_entries_imply_infinite_equal_pairs_l2860_286003


namespace vector_equation_solution_l2860_286081

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Two vectors are not collinear -/
def NotCollinear (a b : V) : Prop := ∀ (k : ℝ), k • a ≠ b

theorem vector_equation_solution
  (a b : V) (x : ℝ)
  (h_not_collinear : NotCollinear a b)
  (h_c : ∃ c : V, c = x • a + b)
  (h_d : ∃ d : V, d = a + (2*x - 1) • b)
  (h_collinear : ∃ (k : ℝ) (c d : V), c = x • a + b ∧ d = a + (2*x - 1) • b ∧ d = k • c) :
  x = 1 ∨ x = -1/2 := by
sorry

end vector_equation_solution_l2860_286081


namespace p_and_q_sufficient_not_necessary_for_not_p_false_l2860_286046

theorem p_and_q_sufficient_not_necessary_for_not_p_false (p q : Prop) :
  (∃ (p q : Prop), (p ∧ q → ¬¬p) ∧ ¬(¬¬p → p ∧ q)) :=
sorry

end p_and_q_sufficient_not_necessary_for_not_p_false_l2860_286046


namespace trapezoid_base_midpoint_relation_shorter_base_length_l2860_286000

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  long_base : ℝ
  midpoint_segment : ℝ
  short_base : ℝ

/-- The theorem stating the relationship between the bases and midpoint segment in a trapezoid -/
theorem trapezoid_base_midpoint_relation (t : Trapezoid) 
  (h1 : t.long_base = 113)
  (h2 : t.midpoint_segment = 4) :
  t.short_base = 105 := by
  sorry

/-- The main theorem proving the length of the shorter base -/
theorem shorter_base_length :
  ∃ t : Trapezoid, t.long_base = 113 ∧ t.midpoint_segment = 4 ∧ t.short_base = 105 := by
  sorry

end trapezoid_base_midpoint_relation_shorter_base_length_l2860_286000


namespace intersection_parallel_perpendicular_l2860_286059

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x + y - 5 = 0
def line2 (x y : ℝ) : Prop := x - 2 * y = 0
def line_l (x y : ℝ) : Prop := 3 * x - y - 7 = 0

-- Define point P as the intersection of line1 and line2
def point_p : ℝ × ℝ := (2, 1)

-- Define the parallel and perpendicular lines
def parallel_line (x y : ℝ) : Prop := 3 * x - y - 5 = 0
def perpendicular_line (x y : ℝ) : Prop := x + 3 * y - 5 = 0

theorem intersection_parallel_perpendicular :
  (∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ (x, y) = point_p) →
  (parallel_line (point_p.1) (point_p.2)) ∧
  (∀ (x y : ℝ), parallel_line x y → (y - point_p.2) = 3 * (x - point_p.1)) ∧
  (perpendicular_line (point_p.1) (point_p.2)) ∧
  (∀ (x y : ℝ), perpendicular_line x y → (y - point_p.2) = -(1/3) * (x - point_p.1)) :=
by sorry


end intersection_parallel_perpendicular_l2860_286059


namespace point_on_x_axis_with_distance_l2860_286090

/-- A point P on the x-axis that is √30 distance from P₁(4,1,2) has x-coordinate 9 or -1 -/
theorem point_on_x_axis_with_distance (x : ℝ) :
  (x - 4)^2 + 1^2 + 2^2 = 30 → x = 9 ∨ x = -1 := by
  sorry

#check point_on_x_axis_with_distance

end point_on_x_axis_with_distance_l2860_286090


namespace angle_between_vectors_l2860_286091

def a : ℝ × ℝ := (3, 0)
def b : ℝ × ℝ := (-5, 5)

theorem angle_between_vectors : 
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  θ = 3 * π / 4 := by
sorry

end angle_between_vectors_l2860_286091


namespace not_all_cells_marked_l2860_286062

/-- Represents a cell in the grid --/
structure Cell :=
  (x : Nat) (y : Nat)

/-- The grid of cells --/
def Grid := List Cell

/-- Checks if two cells are neighbors --/
def isNeighbor (c1 c2 : Cell) : Bool :=
  (c1.x = c2.x ∧ (c1.y = c2.y + 1 ∨ c1.y = c2.y - 1)) ∨
  (c1.y = c2.y ∧ (c1.x = c2.x + 1 ∨ c1.x = c2.x - 1))

/-- Counts the number of marked neighbors for a cell --/
def countMarkedNeighbors (cell : Cell) (markedCells : List Cell) : Nat :=
  (markedCells.filter (isNeighbor cell)).length

/-- Spreads the marking to cells with at least two marked neighbors --/
def spread (grid : Grid) (markedCells : List Cell) : List Cell :=
  markedCells ++ (grid.filter (fun c => countMarkedNeighbors c markedCells ≥ 2))

/-- Creates a 10x10 grid --/
def createGrid : Grid :=
  List.range 10 >>= fun x => List.range 10 >>= fun y => [Cell.mk x y]

/-- The main theorem --/
theorem not_all_cells_marked (initialMarked : List Cell) 
  (h : initialMarked.length = 9) : 
  ∃ (finalMarked : List Cell), finalMarked = spread (createGrid) initialMarked ∧ 
  finalMarked.length < 100 := by
  sorry

end not_all_cells_marked_l2860_286062


namespace probability_union_mutually_exclusive_l2860_286013

theorem probability_union_mutually_exclusive (A B : Set Ω) (P : Set Ω → ℝ) 
  (h_mutex : A ∩ B = ∅) (h_prob_A : P A = 0.25) (h_prob_B : P B = 0.18) :
  P (A ∪ B) = 0.43 := by
  sorry

end probability_union_mutually_exclusive_l2860_286013


namespace only_zero_solution_l2860_286052

theorem only_zero_solution (n : ℕ) : 
  (∃ k : ℤ, (30 * n + 2) = k * (12 * n + 1)) ↔ n = 0 :=
by sorry

end only_zero_solution_l2860_286052


namespace expression_evaluation_l2860_286033

theorem expression_evaluation : 
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / 
  (2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10 - 11 + 12) = 6 / 7 := by
  sorry

end expression_evaluation_l2860_286033


namespace tangent_lines_to_circle_l2860_286051

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A circle in 2D space represented by (x - h)² + (y - k)² = r² -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Check if a point (x, y) lies on a line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop :=
  ∃ (x y : ℝ), l.contains x y ∧ (x - c.h)^2 + (y - c.k)^2 = c.r^2 ∧
  ∀ (x' y' : ℝ), l.contains x' y' → (x' - c.h)^2 + (y' - c.k)^2 ≥ c.r^2

theorem tangent_lines_to_circle (c : Circle) (p : ℝ × ℝ) :
  c.h = 0 ∧ c.k = 0 ∧ c.r = 3 ∧ p = (3, 1) →
  ∃ (l1 l2 : Line),
    (l1.a = 4 ∧ l1.b = 3 ∧ l1.c = -15) ∧
    (l2.a = 1 ∧ l2.b = 0 ∧ l2.c = -3) ∧
    l1.contains p.1 p.2 ∧
    l2.contains p.1 p.2 ∧
    isTangent l1 c ∧
    isTangent l2 c ∧
    ∀ (l : Line), l.contains p.1 p.2 ∧ isTangent l c → l = l1 ∨ l = l2 :=
by sorry

end tangent_lines_to_circle_l2860_286051


namespace smallest_three_digit_odd_multiple_of_three_l2860_286026

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def first_digit (n : ℕ) : ℕ := n / 100

theorem smallest_three_digit_odd_multiple_of_three :
  ∃ (n : ℕ), is_three_digit n ∧ 
             Odd (first_digit n) ∧ 
             n % 3 = 0 ∧
             (∀ m : ℕ, is_three_digit m ∧ Odd (first_digit m) ∧ m % 3 = 0 → n ≤ m) ∧
             n = 102 :=
sorry

end smallest_three_digit_odd_multiple_of_three_l2860_286026


namespace density_not_vector_l2860_286041

/-- A type representing physical quantities --/
inductive PhysicalQuantity
| Buoyancy
| WindSpeed
| Displacement
| Density

/-- Definition of a vector --/
def isVector (q : PhysicalQuantity) : Prop :=
  ∃ (magnitude : ℝ) (direction : ℝ × ℝ × ℝ), True

/-- Theorem stating that density is not a vector --/
theorem density_not_vector : ¬ isVector PhysicalQuantity.Density := by
  sorry

end density_not_vector_l2860_286041


namespace cucumber_weight_problem_l2860_286037

/-- Proves that the initial weight of cucumbers is 100 pounds given the conditions -/
theorem cucumber_weight_problem (initial_water_percent : Real) 
                                 (final_water_percent : Real)
                                 (final_weight : Real) :
  initial_water_percent = 0.99 →
  final_water_percent = 0.98 →
  final_weight = 50 →
  ∃ (initial_weight : Real),
    initial_weight = 100 ∧
    (1 - initial_water_percent) * initial_weight = (1 - final_water_percent) * final_weight :=
by
  sorry

#check cucumber_weight_problem

end cucumber_weight_problem_l2860_286037


namespace river_flow_rate_l2860_286069

/-- Given a river with specified dimensions and flow rate, calculate its flow speed in km/h -/
theorem river_flow_rate (depth : ℝ) (width : ℝ) (flow_volume : ℝ) :
  depth = 2 →
  width = 45 →
  flow_volume = 9000 →
  (flow_volume / (depth * width) / 1000 * 60) = 6 := by
  sorry

end river_flow_rate_l2860_286069


namespace store_profit_calculation_l2860_286063

/-- Represents the pricing strategy and profit calculation for a store selling turtleneck sweaters. -/
theorem store_profit_calculation (C : ℝ) : 
  let initial_markup := 0.20
  let new_year_markup := 0.25
  let february_discount := 0.07
  
  let SP1 := C * (1 + initial_markup)
  let SP2 := SP1 * (1 + new_year_markup)
  let SPF := SP2 * (1 - february_discount)
  let profit := SPF - C
  
  profit / C = 0.395 := by sorry

end store_profit_calculation_l2860_286063


namespace train_crossing_time_l2860_286079

/-- Proves that a train of given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 600 → 
  train_speed_kmh = 144 → 
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) → 
  crossing_time = 15 := by sorry

end train_crossing_time_l2860_286079


namespace clock_resale_price_l2860_286020

theorem clock_resale_price (original_cost : ℝ) : 
  -- Conditions
  original_cost > 0 → 
  -- Store sold to collector for 20% more than original cost
  let collector_price := 1.2 * original_cost
  -- Store bought back at 50% of collector's price
  let buyback_price := 0.5 * collector_price
  -- Difference between original cost and buyback price is $100
  original_cost - buyback_price = 100 →
  -- Store resold at 80% profit on buyback price
  let final_price := buyback_price + 0.8 * buyback_price
  -- Theorem: The final selling price is $270
  final_price = 270 := by
sorry

end clock_resale_price_l2860_286020


namespace weight_loss_days_l2860_286074

/-- Calculates the number of days required to lose a given amount of weight
    under specific calorie intake and expenditure conditions. -/
def days_to_lose_weight (pounds_to_lose : ℕ) (calories_per_pound : ℕ) 
    (calories_burned_per_day : ℕ) (calories_eaten_per_day : ℕ) : ℕ :=
  let total_calories_to_burn := pounds_to_lose * calories_per_pound
  let net_calories_burned_per_day := calories_burned_per_day - calories_eaten_per_day
  total_calories_to_burn / net_calories_burned_per_day

/-- Theorem stating that it takes 35 days to lose 5 pounds under the given conditions -/
theorem weight_loss_days : 
  days_to_lose_weight 5 3500 2500 2000 = 35 := by
  sorry

#eval days_to_lose_weight 5 3500 2500 2000

end weight_loss_days_l2860_286074


namespace complex_fraction_magnitude_l2860_286083

theorem complex_fraction_magnitude (z w : ℂ) 
  (hz : Complex.abs z = 1)
  (hw : Complex.abs w = 3)
  (hzw : Complex.abs (z + w) = 2) :
  Complex.abs (1 / z + 1 / w) = 2 / 3 := by
sorry

end complex_fraction_magnitude_l2860_286083


namespace system_of_equations_solution_l2860_286050

theorem system_of_equations_solution :
  ∃ (x y : ℚ), (3 * x - 4 * y = -6) ∧ (6 * x - 5 * y = 9) ∧ (x = 22 / 3) ∧ (y = 7) := by
  sorry

end system_of_equations_solution_l2860_286050


namespace exists_22_same_age_l2860_286009

/-- Represents a villager in Roche -/
structure Villager where
  age : ℕ

/-- The village of Roche -/
structure Village where
  inhabitants : Finset Villager
  total_count : inhabitants.card = 2020
  knows_same_age : ∀ v ∈ inhabitants, ∃ w ∈ inhabitants, v ≠ w ∧ v.age = w.age
  three_same_age_in_192 : ∀ (group : Finset Villager), group ⊆ inhabitants → group.card = 192 →
    ∃ (a : ℕ) (v₁ v₂ v₃ : Villager), v₁ ∈ group ∧ v₂ ∈ group ∧ v₃ ∈ group ∧
      v₁ ≠ v₂ ∧ v₁ ≠ v₃ ∧ v₂ ≠ v₃ ∧ v₁.age = a ∧ v₂.age = a ∧ v₃.age = a

/-- There exists a group of at least 22 villagers of the same age in Roche -/
theorem exists_22_same_age (roche : Village) : 
  ∃ (a : ℕ) (group : Finset Villager), group ⊆ roche.inhabitants ∧ group.card ≥ 22 ∧
    ∀ v ∈ group, v.age = a :=
sorry

end exists_22_same_age_l2860_286009


namespace marks_jump_height_l2860_286040

theorem marks_jump_height :
  ∀ (mark_height lisa_height jacob_height james_height : ℝ),
    lisa_height = 2 * mark_height →
    jacob_height = 2 * lisa_height →
    james_height = 16 →
    james_height = 2/3 * jacob_height →
    mark_height = 6 := by
  sorry

end marks_jump_height_l2860_286040


namespace smallest_integer_cube_root_l2860_286072

theorem smallest_integer_cube_root (m n : ℕ) (r : ℝ) : 
  (0 < n) →
  (0 < r) →
  (r < 1/100) →
  (m = ((n : ℝ) + r)^3) →
  (∀ k < m, ¬∃ (s : ℝ), 0 < s ∧ s < 1/100 ∧ (k : ℝ)^(1/3) = (k : ℝ) + s) →
  (n = 6) := by
sorry

end smallest_integer_cube_root_l2860_286072


namespace range_of_f_l2860_286071

def f (x : Int) : Int := (x - 1)^2 + 1

def domain : Set Int := {-1, 0, 1, 2, 3}

theorem range_of_f : 
  {y | ∃ x ∈ domain, f x = y} = {1, 2, 5} := by sorry

end range_of_f_l2860_286071


namespace earthquake_relief_team_selection_l2860_286023

/-- The number of ways to select a team of 5 doctors from 3 specialties -/
def select_team (orthopedic neurosurgeon internist : ℕ) : ℕ :=
  let total := orthopedic + neurosurgeon + internist
  let team_size := 5
  -- Add the number of ways for each valid combination
  (orthopedic.choose 3 * neurosurgeon.choose 1 * internist.choose 1) +
  (orthopedic.choose 1 * neurosurgeon.choose 3 * internist.choose 1) +
  (orthopedic.choose 1 * neurosurgeon.choose 1 * internist.choose 3) +
  (orthopedic.choose 2 * neurosurgeon.choose 2 * internist.choose 1) +
  (orthopedic.choose 1 * neurosurgeon.choose 2 * internist.choose 2) +
  (orthopedic.choose 2 * neurosurgeon.choose 1 * internist.choose 2)

/-- Theorem: The number of ways to select 5 people from 3 orthopedic doctors, 
    4 neurosurgeons, and 5 internists, with at least one from each specialty, is 590 -/
theorem earthquake_relief_team_selection : select_team 3 4 5 = 590 := by
  sorry

end earthquake_relief_team_selection_l2860_286023


namespace mans_downstream_rate_l2860_286045

/-- The man's rate when rowing downstream, given his rate in still water and the current's rate -/
def downstream_rate (still_water_rate current_rate : ℝ) : ℝ :=
  still_water_rate + current_rate

/-- Theorem: The man's rate when rowing downstream is 32 kmph -/
theorem mans_downstream_rate :
  let still_water_rate : ℝ := 24.5
  let current_rate : ℝ := 7.5
  downstream_rate still_water_rate current_rate = 32 := by
  sorry

end mans_downstream_rate_l2860_286045


namespace train_length_l2860_286035

/-- Proves that a train with the given conditions has a length of 1500 meters -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (train_length : ℝ) : 
  train_speed = 180 * (1000 / 3600) →  -- Convert 180 km/hr to m/s
  crossing_time = 60 →  -- Convert 1 minute to seconds
  train_length * 2 = train_speed * crossing_time →
  train_length = 1500 := by
  sorry

#check train_length

end train_length_l2860_286035


namespace max_value_cos_sin_l2860_286068

theorem max_value_cos_sin (θ : Real) (h : 0 < θ ∧ θ < π) :
  ∃ (M : Real), M = (3 : Real) / 2 ∧
  ∀ φ, 0 < φ ∧ φ < π →
    Real.cos (φ / 2) * (2 - Real.sin φ) ≤ M ∧
    ∃ ψ, 0 < ψ ∧ ψ < π ∧ Real.cos (ψ / 2) * (2 - Real.sin ψ) = M :=
by sorry

end max_value_cos_sin_l2860_286068


namespace lineup_count_l2860_286066

def team_size : ℕ := 18

def lineup_positions : List String := ["goalkeeper", "center-back", "center-back", "left-back", "right-back", "midfielder", "midfielder", "midfielder"]

def number_of_lineups : ℕ :=
  team_size *
  (team_size - 1) * (team_size - 2) *
  (team_size - 3) *
  (team_size - 4) *
  (team_size - 5) * (team_size - 6) * (team_size - 7)

theorem lineup_count :
  number_of_lineups = 95414400 :=
by sorry

end lineup_count_l2860_286066


namespace smallest_number_in_ratio_l2860_286099

theorem smallest_number_in_ratio (a b c : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  a * 5 = b * 3 →
  a * 7 = c * 3 →
  c = 56 →
  c - a = 32 →
  a = 24 := by sorry

end smallest_number_in_ratio_l2860_286099


namespace largest_divisor_of_expression_l2860_286043

theorem largest_divisor_of_expression : 
  ∃ (x : ℕ), x = 18 ∧ 
  (∀ (y : ℕ), x ∣ (7^y + 12*y - 1)) ∧
  (∀ (z : ℕ), z > x → ∃ (w : ℕ), ¬(z ∣ (7^w + 12*w - 1))) := by
  sorry

end largest_divisor_of_expression_l2860_286043


namespace angle_sum_eq_pi_fourth_l2860_286018

theorem angle_sum_eq_pi_fourth (α β : Real) 
  (h1 : α ∈ Set.Ioo 0 (π / 2))
  (h2 : β ∈ Set.Ioo 0 (π / 2))
  (h3 : Real.tan α = 1 / 7)
  (h4 : Real.tan β = 1 / 3) :
  α + 2 * β = π / 4 := by
  sorry

end angle_sum_eq_pi_fourth_l2860_286018


namespace handshake_theorem_l2860_286011

theorem handshake_theorem (n : ℕ) (k : ℕ) (h : n = 30 ∧ k = 3) :
  (n * k) / 2 = 45 := by
  sorry

end handshake_theorem_l2860_286011


namespace cylinder_volume_change_l2860_286065

/-- Given a cylinder with volume 15 cubic feet, prove that tripling its radius
    and halving its height results in a new volume of 67.5 cubic feet. -/
theorem cylinder_volume_change (r h : ℝ) (h1 : r > 0) (h2 : h > 0) : 
  π * r^2 * h = 15 → π * (3*r)^2 * (h/2) = 67.5 := by
  sorry

end cylinder_volume_change_l2860_286065


namespace midpoint_distance_after_move_l2860_286060

/-- Given two points P(p,q) and Q(r,s) on a Cartesian plane with midpoint N(x,y),
    prove that after moving P 3 units right and 5 units up, and Q 5 units left and 3 units down,
    the distance between N and the new midpoint N' is √2. -/
theorem midpoint_distance_after_move (p q r s x y : ℝ) :
  x = (p + r) / 2 →
  y = (q + s) / 2 →
  let x' := (p + 3 + r - 5) / 2
  let y' := (q + 5 + s - 3) / 2
  Real.sqrt ((x - x')^2 + (y - y')^2) = Real.sqrt 2 := by
  sorry

end midpoint_distance_after_move_l2860_286060


namespace equation_solution_l2860_286016

theorem equation_solution : 
  ∃! x : ℚ, (x - 100) / 3 = (5 - 3 * x) / 7 ∧ x = 715 / 16 := by sorry

end equation_solution_l2860_286016


namespace remaining_coins_value_is_1030_l2860_286067

-- Define the initial number of coins
def initial_quarters : ℕ := 33
def initial_nickels : ℕ := 87
def initial_dimes : ℕ := 52

-- Define the number of borrowed coins
def borrowed_quarters : ℕ := 15
def borrowed_nickels : ℕ := 75

-- Define the value of each coin type in cents
def quarter_value : ℕ := 25
def nickel_value : ℕ := 5
def dime_value : ℕ := 10

-- Define the function to calculate the total value of remaining coins
def remaining_coins_value : ℕ :=
  (initial_quarters * quarter_value + 
   initial_nickels * nickel_value + 
   initial_dimes * dime_value) - 
  (borrowed_quarters * quarter_value + 
   borrowed_nickels * nickel_value)

-- Theorem statement
theorem remaining_coins_value_is_1030 : 
  remaining_coins_value = 1030 := by sorry

end remaining_coins_value_is_1030_l2860_286067


namespace equation_solution_l2860_286031

theorem equation_solution :
  ∃ x : ℝ, 45 - (28 - (37 - (15 - x))) = 55 ∧ x = 16 := by
  sorry

end equation_solution_l2860_286031


namespace hyperbola_equation_l2860_286024

/-- Given a hyperbola C and a parabola, prove the equation of the hyperbola -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →  -- Hyperbola equation
  2 * Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 5 →  -- Focal distance
  (∃ k : ℝ, ∀ x : ℝ, (1/16 * x^2 + 1 - k*x = 0 → 
    (k = b/a ∨ k = -b/a))) →  -- Parabola tangent to asymptotes
  (∀ x y : ℝ, x^2 / 4 - y^2 = 1) :=  -- Conclusion: Specific hyperbola equation
by sorry

end hyperbola_equation_l2860_286024


namespace quadruplet_babies_l2860_286010

theorem quadruplet_babies (total_babies : ℕ) 
  (h_total : total_babies = 1200)
  (h_triplets_quadruplets : ∃ (t q : ℕ), t = 5 * q)
  (h_twins_triplets : ∃ (w t : ℕ), w = 2 * t)
  (h_sum : ∃ (w t q : ℕ), 2 * w + 3 * t + 4 * q = total_babies) :
  ∃ (q : ℕ), 4 * q = 123 := by
sorry

end quadruplet_babies_l2860_286010
