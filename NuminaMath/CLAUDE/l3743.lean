import Mathlib

namespace ben_pea_picking_l3743_374310

/-- Given that Ben can pick 56 sugar snap peas in 7 minutes,
    prove that it will take him 9 minutes to pick 72 sugar snap peas. -/
theorem ben_pea_picking (rate : ℝ) (h : rate * 7 = 56) : rate * 9 = 72 := by
  sorry

end ben_pea_picking_l3743_374310


namespace least_number_with_remainder_l3743_374372

theorem least_number_with_remainder (n : ℕ) : n = 130 ↔ 
  (∀ m, m < n → ¬(m % 6 = 4 ∧ m % 7 = 4 ∧ m % 9 = 4 ∧ m % 18 = 4)) ∧
  n % 6 = 4 ∧ n % 7 = 4 ∧ n % 9 = 4 ∧ n % 18 = 4 := by
sorry

end least_number_with_remainder_l3743_374372


namespace set_relations_l3743_374352

def A (a : ℝ) : Set ℝ := {x | 1 - a ≤ x ∧ x ≤ 1 + a}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

theorem set_relations (a : ℝ) :
  (A a ∩ B = ∅ ↔ 0 ≤ a ∧ a ≤ 4) ∧
  (A a ∪ B = B ↔ a < -4) := by
  sorry

end set_relations_l3743_374352


namespace solution_absolute_value_equation_l3743_374322

theorem solution_absolute_value_equation :
  ∀ x : ℝ, 2 * |x - 5| = 6 ↔ x = 2 ∨ x = 8 := by
  sorry

end solution_absolute_value_equation_l3743_374322


namespace no_polyhedron_with_seven_edges_l3743_374369

-- Define a polyhedron structure
structure Polyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  euler_formula : vertices - edges + faces = 2
  min_edges_per_vertex : edges ≥ (3 * vertices) / 2

-- Theorem statement
theorem no_polyhedron_with_seven_edges : 
  ∀ p : Polyhedron, p.edges ≠ 7 := by
  sorry

end no_polyhedron_with_seven_edges_l3743_374369


namespace round_to_nearest_integer_l3743_374361

def number : ℝ := 7293847.2635142

theorem round_to_nearest_integer : 
  Int.floor (number + 0.5) = 7293847 := by sorry

end round_to_nearest_integer_l3743_374361


namespace salary_solution_l3743_374342

def salary_problem (salary : ℝ) : Prop :=
  let food_expense := (1 / 5 : ℝ) * salary
  let rent_expense := (1 / 10 : ℝ) * salary
  let clothes_expense := (3 / 5 : ℝ) * salary
  let remaining := 15000
  salary - food_expense - rent_expense - clothes_expense = remaining

theorem salary_solution :
  ∃ (salary : ℝ), salary_problem salary ∧ salary = 150000 := by
  sorry

end salary_solution_l3743_374342


namespace minimum_occupied_seats_theorem_l3743_374304

/-- Represents a row of seats -/
structure SeatRow where
  total_seats : ℕ
  occupied_seats : ℕ

/-- Checks if the next person must sit next to someone already seated -/
def next_person_sits_next (row : SeatRow) : Prop :=
  row.occupied_seats * 2 ≥ row.total_seats

/-- The theorem to be proved -/
theorem minimum_occupied_seats_theorem (row : SeatRow) 
  (h1 : row.total_seats = 180) 
  (h2 : row.occupied_seats = 90) :
  (∀ n : ℕ, n < 90 → ¬(next_person_sits_next ⟨180, n⟩)) ∧ 
  next_person_sits_next row :=
sorry

end minimum_occupied_seats_theorem_l3743_374304


namespace min_keys_required_l3743_374360

/-- Represents a hotel with rooms and guests -/
structure Hotel where
  rooms : ℕ
  guests : ℕ

/-- Represents the key distribution system for the hotel -/
structure KeyDistribution where
  hotel : Hotel
  keys : ℕ
  returningGuests : ℕ

/-- Checks if the key distribution is valid for the hotel -/
def isValidDistribution (kd : KeyDistribution) : Prop :=
  kd.returningGuests ≤ kd.hotel.guests ∧
  kd.returningGuests ≤ kd.hotel.rooms ∧
  kd.keys ≥ kd.hotel.rooms * (kd.hotel.guests - kd.hotel.rooms + 1)

/-- Theorem: The minimum number of keys required for the given hotel scenario is 990 -/
theorem min_keys_required (h : Hotel) (kd : KeyDistribution) 
  (hrooms : h.rooms = 90)
  (hguests : h.guests = 100)
  (hreturning : kd.returningGuests = 90)
  (hhotel : kd.hotel = h)
  (hvalid : isValidDistribution kd) :
  kd.keys ≥ 990 := by
  sorry

end min_keys_required_l3743_374360


namespace range_of_a_l3743_374356

theorem range_of_a (a : ℝ) 
  (h1 : ∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x)
  (h2 : ∃ x : ℝ, x^2 + 4*x + a = 0) :
  a ∈ Set.Icc (Real.exp 1) 4 :=
sorry

end range_of_a_l3743_374356


namespace area_ratio_constant_l3743_374313

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define points A, B, O, and T
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)
def O : ℝ × ℝ := (0, 0)
def T : ℝ × ℝ := (4, 0)

-- Define a line l passing through T
def line_l (m : ℝ) (x y : ℝ) : Prop := x = m * y + 4

-- Define the intersection points M and N
def M (m : ℝ) : ℝ × ℝ := sorry
def N (m : ℝ) : ℝ × ℝ := sorry

-- Define point P as the intersection of BM and x=1
def P (m : ℝ) : ℝ × ℝ := sorry

-- Define point Q as the intersection of AN and y-axis
def Q (m : ℝ) : ℝ × ℝ := sorry

-- Define the area of a triangle given three points
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_ratio_constant (m : ℝ) : 
  triangle_area O A (Q m) / triangle_area O T (P m) = 1/3 := by sorry

end area_ratio_constant_l3743_374313


namespace nine_digit_divisible_by_11_l3743_374308

def is_divisible_by_11 (n : ℕ) : Prop :=
  ∃ k : ℤ, n = 11 * k

def sum_odd_positions (m : ℕ) : ℕ :=
  8 + 4 + m + 6 + 8

def sum_even_positions : ℕ :=
  5 + 2 + 7 + 1

def number (m : ℕ) : ℕ :=
  8542000000 + m * 10000 + 7618

theorem nine_digit_divisible_by_11 (m : ℕ) (h : m < 10) :
  is_divisible_by_11 (number m) → m = 0 := by
  sorry

end nine_digit_divisible_by_11_l3743_374308


namespace perimeter_ratio_of_folded_paper_l3743_374358

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

theorem perimeter_ratio_of_folded_paper : 
  let original_side : ℝ := 6
  let large_rectangle : Rectangle := { length := original_side, width := original_side / 2 }
  let small_rectangle : Rectangle := { length := original_side / 2, width := original_side / 2 }
  (perimeter small_rectangle) / (perimeter large_rectangle) = 2 / 3 := by
  sorry


end perimeter_ratio_of_folded_paper_l3743_374358


namespace butanoic_acid_molecular_weight_l3743_374397

/-- The molecular weight of one mole of Butanoic acid. -/
def molecular_weight_one_mole : ℝ := 88

/-- The number of moles given in the problem. -/
def num_moles : ℝ := 9

/-- The total molecular weight of the given number of moles. -/
def total_molecular_weight : ℝ := 792

/-- Theorem stating that the molecular weight of one mole of Butanoic acid is 88 g/mol,
    given that the molecular weight of 9 moles is 792. -/
theorem butanoic_acid_molecular_weight :
  molecular_weight_one_mole = total_molecular_weight / num_moles :=
by sorry

end butanoic_acid_molecular_weight_l3743_374397


namespace green_socks_count_l3743_374344

theorem green_socks_count (total : ℕ) (white : ℕ) (blue : ℕ) (red : ℕ) (green : ℕ) :
  total = 900 ∧
  white = total / 3 ∧
  blue = total / 4 ∧
  red = total / 5 ∧
  green = total - (white + blue + red) →
  green = 195 := by
sorry

end green_socks_count_l3743_374344


namespace product_mod_twenty_l3743_374309

theorem product_mod_twenty : 58 * 73 * 84 ≡ 16 [MOD 20] := by sorry

end product_mod_twenty_l3743_374309


namespace jisha_walking_speed_l3743_374320

/-- Jisha's walking problem -/
theorem jisha_walking_speed :
  -- Day 1 conditions
  let day1_distance : ℝ := 18
  let day1_speed : ℝ := 3
  let day1_hours : ℝ := day1_distance / day1_speed

  -- Day 2 conditions
  let day2_hours : ℝ := day1_hours - 1

  -- Day 3 conditions
  let day3_hours : ℝ := day1_hours

  -- Total distance
  let total_distance : ℝ := 62

  -- Unknown speed for Day 2 and 3
  ∀ day2_speed : ℝ,
    -- Total distance equation
    day1_distance + day2_speed * day2_hours + day2_speed * day3_hours = total_distance →
    -- Conclusion: Day 2 speed is 4 mph
    day2_speed = 4 :=
by
  sorry

end jisha_walking_speed_l3743_374320


namespace hose_fill_time_proof_l3743_374386

/-- Represents the time (in hours) it takes for the hose to fill the pool -/
def hose_fill_time (pool_capacity : ℝ) (drain_time : ℝ) (time_elapsed : ℝ) (remaining_water : ℝ) : ℝ :=
  3

/-- Proves that the hose fill time is correct given the problem conditions -/
theorem hose_fill_time_proof (pool_capacity : ℝ) (drain_time : ℝ) (time_elapsed : ℝ) (remaining_water : ℝ)
  (h1 : pool_capacity = 120)
  (h2 : drain_time = 4)
  (h3 : time_elapsed = 3)
  (h4 : remaining_water = 90) :
  hose_fill_time pool_capacity drain_time time_elapsed remaining_water = 3 := by
  sorry

#eval hose_fill_time 120 4 3 90

end hose_fill_time_proof_l3743_374386


namespace factorization_equality_l3743_374364

theorem factorization_equality (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) := by
  sorry

end factorization_equality_l3743_374364


namespace equivalent_discount_l3743_374362

/-- Proves that a single discount of 40.5% on a $50 item results in the same final price
    as applying a 30% discount followed by a 15% discount on the discounted price. -/
theorem equivalent_discount (original_price : ℝ) (first_discount second_discount single_discount : ℝ) :
  original_price = 50 ∧
  first_discount = 0.3 ∧
  second_discount = 0.15 ∧
  single_discount = 0.405 →
  original_price * (1 - single_discount) =
  original_price * (1 - first_discount) * (1 - second_discount) :=
by sorry

end equivalent_discount_l3743_374362


namespace f_monotone_decreasing_l3743_374385

-- Define the function f(x) = x^2 - 2x
def f (x : ℝ) := x^2 - 2*x

-- State the theorem
theorem f_monotone_decreasing :
  MonotoneOn f (Set.Iic 1) := by sorry

end f_monotone_decreasing_l3743_374385


namespace solve_for_y_l3743_374303

theorem solve_for_y (x y : ℝ) (h1 : x - y = 6) (h2 : x + y = 12) : y = 3 := by
  sorry

end solve_for_y_l3743_374303


namespace bingo_prize_distribution_l3743_374306

theorem bingo_prize_distribution (total_prize : ℝ) (first_winner_share : ℝ) (remaining_winners : ℕ) : 
  total_prize = 2400 →
  first_winner_share = total_prize / 3 →
  remaining_winners = 10 →
  (total_prize - first_winner_share) / remaining_winners = 160 := by
  sorry

end bingo_prize_distribution_l3743_374306


namespace complex_number_properties_l3743_374368

theorem complex_number_properties (z : ℂ) (h : z * Complex.I = -3 + 2 * Complex.I) :
  z.im = 3 ∧ Complex.abs z = Real.sqrt 13 := by
  sorry

end complex_number_properties_l3743_374368


namespace negation_of_universal_proposition_l3743_374384

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℕ, x^2 > 1) ↔ (∃ x : ℕ, x^2 ≤ 1) := by
  sorry

end negation_of_universal_proposition_l3743_374384


namespace difference_of_squares_special_case_l3743_374378

theorem difference_of_squares_special_case : (3 + Real.sqrt 2) * (3 - Real.sqrt 2) = 7 := by
  sorry

end difference_of_squares_special_case_l3743_374378


namespace sum_a_b_equals_one_l3743_374347

theorem sum_a_b_equals_one (a b : ℝ) (h : (a + 1)^2 + |b - 2| = 0) : a + b = 1 := by
  sorry

end sum_a_b_equals_one_l3743_374347


namespace raisin_cost_fraction_l3743_374353

theorem raisin_cost_fraction (raisin_cost : ℝ) : 
  let nut_cost : ℝ := 3 * raisin_cost
  let raisin_weight : ℝ := 3
  let nut_weight : ℝ := 3
  let total_raisin_cost : ℝ := raisin_cost * raisin_weight
  let total_nut_cost : ℝ := nut_cost * nut_weight
  let total_cost : ℝ := total_raisin_cost + total_nut_cost
  total_raisin_cost / total_cost = 1 / 4 :=
by sorry

end raisin_cost_fraction_l3743_374353


namespace triangle_midpoint_intersection_min_value_l3743_374350

theorem triangle_midpoint_intersection_min_value (A B C D E M N : ℝ × ℝ) 
  (hAD : D = (A + B + C) / 3)  -- D is centroid of triangle ABC
  (hE : E = (A + D) / 2)       -- E is midpoint of AD
  (hM : ∃ x : ℝ, M = A + x • (B - A))  -- M is on AB
  (hN : ∃ y : ℝ, N = A + y • (C - A))  -- N is on AC
  (hEMN : ∃ t : ℝ, E = M + t • (N - M))  -- E, M, N are collinear
  : ∀ x y : ℝ, M = A + x • (B - A) → N = A + y • (C - A) → 4*x + y ≥ 9/4 :=
sorry

end triangle_midpoint_intersection_min_value_l3743_374350


namespace second_sum_calculation_l3743_374336

/-- Given a total sum of 2665 Rs divided into two parts, where the interest on the first part
    for 5 years at 3% per annum equals the interest on the second part for 3 years at 5% per annum,
    prove that the second part is equal to 1332.5 Rs. -/
theorem second_sum_calculation (total : ℝ) (first_part : ℝ) (second_part : ℝ) :
  total = 2665 →
  first_part + second_part = total →
  (first_part * 3 * 5) / 100 = (second_part * 5 * 3) / 100 →
  second_part = 1332.5 := by
  sorry

end second_sum_calculation_l3743_374336


namespace students_playing_neither_l3743_374382

theorem students_playing_neither (total : ℕ) (football : ℕ) (tennis : ℕ) (both : ℕ) :
  total = 39 →
  football = 26 →
  tennis = 20 →
  both = 17 →
  total - (football + tennis - both) = 10 :=
by
  sorry

end students_playing_neither_l3743_374382


namespace equation_solution_l3743_374357

theorem equation_solution : 
  ∃ x : ℚ, x + 5/6 = 7/18 + 1/2 ∧ x = -7/18 := by sorry

end equation_solution_l3743_374357


namespace annual_production_after_five_years_l3743_374394

/-- Given an initial value, growth rate, and time span, calculate the final value after compound growth -/
def compound_growth (initial_value : ℝ) (growth_rate : ℝ) (time_span : ℕ) : ℝ :=
  initial_value * (1 + growth_rate) ^ time_span

/-- Theorem: The annual production after 5 years with a given growth rate -/
theorem annual_production_after_five_years 
  (a : ℝ) -- initial production in 2005
  (x : ℝ) -- annual growth rate
  : 
  compound_growth a x 5 = a * (1 + x)^5 := by
  sorry

end annual_production_after_five_years_l3743_374394


namespace range_of_sum_and_abs_l3743_374318

theorem range_of_sum_and_abs (a b : ℝ) 
  (ha : -1 ≤ a ∧ a ≤ 3) 
  (hb : -5 < b ∧ b < 3) : 
  ∀ x, x ∈ Set.Icc (-1 : ℝ) 8 ↔ ∃ (a' b' : ℝ), 
    -1 ≤ a' ∧ a' ≤ 3 ∧ 
    -5 < b' ∧ b' < 3 ∧ 
    x = a' + |b'| :=
by sorry

end range_of_sum_and_abs_l3743_374318


namespace integral_properties_l3743_374340

noncomputable section

-- Define the interval [0,1]
def I : Set ℝ := Set.Icc 0 1

-- Define the properties of function f
class C1_function (f : ℝ → ℝ) :=
  (continuous_on : ContinuousOn f I)
  (differentiable_on : DifferentiableOn ℝ f I)

-- Define the properties of function g
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + (x - 1) * deriv f x

-- Define the properties of function φ
class Convex_function (φ : ℝ → ℝ) :=
  (convex : ConvexOn ℝ I φ)
  (differentiable : DifferentiableOn ℝ φ I)

-- Main theorem
theorem integral_properties
  (f : ℝ → ℝ)
  (hf : C1_function f)
  (hf_mono : MonotoneOn f I)
  (hf_zero : f 0 = 0)
  (φ : ℝ → ℝ)
  (hφ : Convex_function φ)
  (hφ_range : ∀ x ∈ I, φ x ∈ I)
  (hφ_zero : φ 0 = 0)
  (hφ_one : φ 1 = 1) :
  (∫ x in I, g f x) = 0 ∧
  (∫ t in I, g f (φ t)) ≤ 0 := by sorry

end integral_properties_l3743_374340


namespace symmetric_points_count_l3743_374365

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 2 * x^2 + 4 * x + 1 else 2 / Real.exp x

-- Define symmetry about the origin
def symmetric_about_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

-- State the theorem
theorem symmetric_points_count :
  ∃ (p₁ q₁ p₂ q₂ : ℝ × ℝ),
    p₁ ≠ q₁ ∧ p₂ ≠ q₂ ∧ p₁ ≠ p₂ ∧
    symmetric_about_origin p₁ q₁ ∧
    symmetric_about_origin p₂ q₂ ∧
    (∀ x, f x = p₁.2 ↔ x = p₁.1) ∧
    (∀ x, f x = q₁.2 ↔ x = q₁.1) ∧
    (∀ x, f x = p₂.2 ↔ x = p₂.1) ∧
    (∀ x, f x = q₂.2 ↔ x = q₂.1) ∧
    (∀ p q : ℝ × ℝ, 
      p ≠ p₁ ∧ p ≠ q₁ ∧ p ≠ p₂ ∧ p ≠ q₂ ∧
      q ≠ p₁ ∧ q ≠ q₁ ∧ q ≠ p₂ ∧ q ≠ q₂ ∧
      symmetric_about_origin p q ∧
      (∀ x, f x = p.2 ↔ x = p.1) ∧
      (∀ x, f x = q.2 ↔ x = q.1) →
      False) :=
sorry

end symmetric_points_count_l3743_374365


namespace circles_intersect_l3743_374349

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 9

-- Define the centers and radii
def center1 : ℝ × ℝ := (-2, 0)
def center2 : ℝ × ℝ := (2, 1)
def radius1 : ℝ := 2
def radius2 : ℝ := 3

-- Theorem stating that the circles are intersecting
theorem circles_intersect :
  let d := Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)
  radius2 + radius1 > d ∧ d > radius2 - radius1 := by sorry

end circles_intersect_l3743_374349


namespace larger_number_proof_l3743_374327

theorem larger_number_proof (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : L = 1635 := by
  sorry

end larger_number_proof_l3743_374327


namespace volleyball_lineup_theorem_l3743_374328

def volleyball_lineup_count (n : ℕ) (k : ℕ) (mvp_count : ℕ) (trio_count : ℕ) : ℕ :=
  Nat.choose (n - mvp_count - trio_count) (k - mvp_count - 1) * trio_count +
  Nat.choose (n - mvp_count - trio_count) (k - mvp_count - 2) * Nat.choose trio_count 2 +
  Nat.choose (n - mvp_count - trio_count) (k - mvp_count - 3) * Nat.choose trio_count 3

theorem volleyball_lineup_theorem :
  volleyball_lineup_count 15 7 2 3 = 1035 := by
  sorry

end volleyball_lineup_theorem_l3743_374328


namespace event_A_sufficient_not_necessary_for_event_B_l3743_374330

/- Define the number of balls for each color -/
def num_red_balls : ℕ := 5
def num_yellow_balls : ℕ := 3
def num_white_balls : ℕ := 2

/- Define the total number of balls -/
def total_balls : ℕ := num_red_balls + num_yellow_balls + num_white_balls

/- Define Event A: Selecting 1 red ball and 1 yellow ball -/
def event_A : Prop := ∃ (r : Fin num_red_balls) (y : Fin num_yellow_balls), True

/- Define Event B: Selecting any 2 balls from all available balls -/
def event_B : Prop := ∃ (b1 b2 : Fin total_balls), b1 ≠ b2

/- Theorem: Event A is sufficient but not necessary for Event B -/
theorem event_A_sufficient_not_necessary_for_event_B :
  (event_A → event_B) ∧ ¬(event_B → event_A) := by
  sorry


end event_A_sufficient_not_necessary_for_event_B_l3743_374330


namespace find_divisor_l3743_374373

theorem find_divisor (n d : ℕ) (h1 : n % d = 255) (h2 : (2 * n) % d = 112) : d = 398 := by
  sorry

end find_divisor_l3743_374373


namespace symmetric_trapezoid_theorem_l3743_374380

/-- A symmetric trapezoid inscribed in a circle -/
structure SymmetricTrapezoid (R : ℝ) where
  x : ℝ
  h_x_range : 0 ≤ x ∧ x ≤ 2*R

/-- The function y for a symmetric trapezoid -/
def y (R : ℝ) (t : SymmetricTrapezoid R) : ℝ :=
  (t.x - R)^2 + 3*R^2

theorem symmetric_trapezoid_theorem (R : ℝ) (h_R : R > 0) :
  ∀ (t : SymmetricTrapezoid R),
    y R t = (t.x - R)^2 + 3*R^2 ∧
    ∀ (a : ℝ), y R t = a^2 → 3*R^2 ≤ a^2 ∧ a^2 ≤ 4*R^2 := by
  sorry

#check symmetric_trapezoid_theorem

end symmetric_trapezoid_theorem_l3743_374380


namespace sarah_pencil_multiple_l3743_374359

/-- The number of pencils Sarah bought on Monday -/
def monday_pencils : ℕ := 20

/-- The number of pencils Sarah bought on Tuesday -/
def tuesday_pencils : ℕ := 18

/-- The total number of pencils Sarah has -/
def total_pencils : ℕ := 92

/-- The multiple of pencils bought on Wednesday compared to Tuesday -/
def wednesday_multiple : ℕ := (total_pencils - monday_pencils - tuesday_pencils) / tuesday_pencils

theorem sarah_pencil_multiple : wednesday_multiple = 3 := by
  sorry

end sarah_pencil_multiple_l3743_374359


namespace intersection_implies_a_value_l3743_374381

theorem intersection_implies_a_value (a : ℝ) : 
  let A : Set ℝ := {a^2, a+1, -3}
  let B : Set ℝ := {a-3, 3*a-1, a^2+1}
  A ∩ B = {-3} → a = -2/3 := by
  sorry

end intersection_implies_a_value_l3743_374381


namespace quadratic_condition_l3743_374324

/-- The equation (m+1)x^2 - mx + 1 = 0 is quadratic if and only if m ≠ -1 -/
theorem quadratic_condition (m : ℝ) :
  (∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, (m + 1) * x^2 - m * x + 1 = a * x^2 + b * x + c) ↔ m ≠ -1 :=
by sorry

end quadratic_condition_l3743_374324


namespace calculate_expression_l3743_374301

theorem calculate_expression : 
  (0.125 : ℝ)^8 * (-8 : ℝ)^7 = -0.125 := by
  sorry

end calculate_expression_l3743_374301


namespace gcd_8421_4312_l3743_374379

theorem gcd_8421_4312 : Nat.gcd 8421 4312 = 1 := by
  sorry

end gcd_8421_4312_l3743_374379


namespace boys_usual_time_to_school_l3743_374390

/-- 
Given a boy who reaches school 4 minutes early when walking at 9/8 of his usual rate,
prove that his usual time to reach the school is 36 minutes.
-/
theorem boys_usual_time_to_school (usual_rate : ℝ) (usual_time : ℝ) 
  (h1 : usual_rate > 0) 
  (h2 : usual_time > 0)
  (h3 : usual_rate * usual_time = (9/8 * usual_rate) * (usual_time - 4)) : 
  usual_time = 36 := by
  sorry

end boys_usual_time_to_school_l3743_374390


namespace min_value_expression_l3743_374315

theorem min_value_expression (a : ℝ) (ha : a > 0) : 
  ((a - 1) * (4 * a - 1)) / a ≥ -1 ∧ 
  ∃ (a₀ : ℝ), a₀ > 0 ∧ ((a₀ - 1) * (4 * a₀ - 1)) / a₀ = -1 := by
  sorry

#check min_value_expression

end min_value_expression_l3743_374315


namespace hexagon_triangle_ratio_l3743_374363

theorem hexagon_triangle_ratio (s_h s_t : ℝ) (h : s_h > 0) (t : s_t > 0) :
  (3 * s_h^2 * Real.sqrt 3) / 2 = (s_t^2 * Real.sqrt 3) / 4 →
  s_t / s_h = Real.sqrt 6 := by
  sorry

end hexagon_triangle_ratio_l3743_374363


namespace equation_one_l3743_374346

theorem equation_one (x : ℝ) : (3 - x)^2 + x^2 = 5 ↔ x = 1 ∨ x = 2 := by sorry

end equation_one_l3743_374346


namespace mixture_composition_l3743_374317

theorem mixture_composition (x y : ℝ) :
  x + y = 100 →
  0.1 * x + 0.2 * y = 12 →
  x = 80 := by
sorry

end mixture_composition_l3743_374317


namespace odd_function_property_l3743_374367

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The main theorem -/
theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : IsOdd f)
  (h_even : IsEven (fun x ↦ f (x + 1)))
  (h_def : ∀ x ∈ Set.Icc 0 1, f x = x * (3 - 2 * x)) :
  f (31/2) = -1 := by
  sorry


end odd_function_property_l3743_374367


namespace sum_of_squares_l3743_374375

theorem sum_of_squares (x y z a b c d : ℝ) 
  (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) 
  (h4 : a ≠ 0) (h5 : b ≠ 0) (h6 : c ≠ 0) (h7 : d ≠ 0)
  (h8 : x * y = a) (h9 : x * z = b) (h10 : y * z = c) (h11 : x + y + z = d) :
  x^2 + y^2 + z^2 = d^2 - 2*(a + b + c) := by
  sorry

end sum_of_squares_l3743_374375


namespace find_x_l3743_374323

theorem find_x : ∃ X : ℝ, (X + 20 / 90) * 90 = 9020 ∧ X = 9000 := by
  sorry

end find_x_l3743_374323


namespace parallel_vectors_m_value_l3743_374302

/-- Given vectors a and b, if 2a + b is parallel to ma - b, then m = -2 -/
theorem parallel_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) 
    (ha : a = (1, -2))
    (hb : b = (3, 0))
    (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ (2 • a + b) = k • (m • a - b)) :
  m = -2 := by
sorry

end parallel_vectors_m_value_l3743_374302


namespace rodney_ian_money_difference_l3743_374398

def rodney_ian_difference (jessica_money : ℕ) (jessica_rodney_diff : ℕ) : ℕ :=
  let rodney_money := jessica_money - jessica_rodney_diff
  let ian_money := jessica_money / 2
  rodney_money - ian_money

theorem rodney_ian_money_difference :
  rodney_ian_difference 100 15 = 35 :=
by sorry

end rodney_ian_money_difference_l3743_374398


namespace cereal_box_theorem_l3743_374399

/-- The number of clusters of oats in each spoonful -/
def clusters_per_spoonful : ℕ := 4

/-- The number of spoonfuls of cereal in each bowl -/
def spoonfuls_per_bowl : ℕ := 25

/-- The number of clusters of oats in each box -/
def clusters_per_box : ℕ := 500

/-- The number of bowlfuls of cereal in each box -/
def bowls_per_box : ℕ := 5

theorem cereal_box_theorem : 
  clusters_per_box / (clusters_per_spoonful * spoonfuls_per_bowl) = bowls_per_box := by
  sorry

end cereal_box_theorem_l3743_374399


namespace permutations_of_three_eq_six_l3743_374370

/-- The number of permutations of 3 distinct elements -/
def permutations_of_three : ℕ := 3 * 2 * 1

/-- Theorem stating that the number of permutations of 3 distinct elements is 6 -/
theorem permutations_of_three_eq_six : permutations_of_three = 6 := by
  sorry

end permutations_of_three_eq_six_l3743_374370


namespace solve_problem_l3743_374333

def problem (x : ℝ) : Prop :=
  let k_speed := x
  let m_speed := x - 0.5
  let k_time := 40 / k_speed
  let m_time := 40 / m_speed
  (m_time - k_time = 1/3) ∧ (k_time = 5)

theorem solve_problem :
  ∃ x : ℝ, problem x :=
sorry

end solve_problem_l3743_374333


namespace geometric_series_second_term_l3743_374387

theorem geometric_series_second_term 
  (r : ℚ) 
  (S : ℚ) 
  (h1 : r = -1/3) 
  (h2 : S = 25) 
  (h3 : S = a / (1 - r)) 
  (h4 : second_term = a * r) : 
  second_term = -100/9 :=
sorry

end geometric_series_second_term_l3743_374387


namespace point_three_units_from_negative_two_l3743_374338

theorem point_three_units_from_negative_two (x : ℝ) : 
  (|x - (-2)| = 3) ↔ (x = -5 ∨ x = 1) := by
  sorry

end point_three_units_from_negative_two_l3743_374338


namespace gabriel_capsule_days_l3743_374351

/-- The number of days in July -/
def days_in_july : ℕ := 31

/-- The number of days Gabriel forgot to take his capsules -/
def days_forgot : ℕ := 3

/-- The number of days Gabriel took his capsules in July -/
def days_took_capsules : ℕ := days_in_july - days_forgot

theorem gabriel_capsule_days : days_took_capsules = 28 := by
  sorry

end gabriel_capsule_days_l3743_374351


namespace sqrt_inequality_range_l3743_374300

theorem sqrt_inequality_range (x : ℝ) : 
  x > 0 → (Real.sqrt (2 * x) < 3 * x - 4 ↔ x > 2) := by sorry

end sqrt_inequality_range_l3743_374300


namespace triangle_angle_property_l3743_374305

theorem triangle_angle_property (α : Real) :
  (0 < α) ∧ (α < π) →  -- α is an interior angle of a triangle
  (1 / Real.sin α + 1 / Real.cos α = 2) →
  α = π + (1 / 2) * Real.arcsin ((1 - Real.sqrt 5) / 2) := by
  sorry

end triangle_angle_property_l3743_374305


namespace number_puzzle_l3743_374341

theorem number_puzzle : ∃ x : ℝ, x^2 + 95 = (x - 15)^2 ∧ x = 10/3 := by
  sorry

end number_puzzle_l3743_374341


namespace polynomial_division_remainder_l3743_374376

theorem polynomial_division_remainder (p q : ℝ) : 
  (∀ x, (x^3 - 3*x^2 + 9*x - 7) = (x - p) * (ax^2 + bx + c) + (2*x + q) → p = 1 ∧ q = -2) :=
sorry

end polynomial_division_remainder_l3743_374376


namespace inequality_and_equality_condition_l3743_374335

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (non_neg_a : a ≥ 0) (non_neg_b : b ≥ 0) (non_neg_c : c ≥ 0) (non_neg_d : d ≥ 0)
  (sum_squares : a^2 + b^2 + c^2 + d^2 = 1) : 
  a + b + c + d - 1 ≥ 16*a*b*c*d ∧ 
  (a + b + c + d - 1 = 16*a*b*c*d ↔ a = 1/2 ∧ b = 1/2 ∧ c = 1/2 ∧ d = 1/2) :=
by sorry

end inequality_and_equality_condition_l3743_374335


namespace simplify_product_of_square_roots_l3743_374312

theorem simplify_product_of_square_roots (y : ℝ) (h : y ≥ 0) :
  Real.sqrt (32 * y) * Real.sqrt (18 * y) * Real.sqrt (50 * y) * Real.sqrt (72 * y) = 960 * y^2 * Real.sqrt 2 := by
  sorry

end simplify_product_of_square_roots_l3743_374312


namespace pasta_for_reunion_l3743_374374

/-- Calculates the amount of pasta needed for a given number of people, 
    based on a recipe that uses 2 pounds for 7 people. -/
def pasta_needed (people : ℕ) : ℚ :=
  2 * (people / 7 : ℚ)

/-- Proves that 10 pounds of pasta are needed for 35 people. -/
theorem pasta_for_reunion : pasta_needed 35 = 10 := by
  sorry

end pasta_for_reunion_l3743_374374


namespace apple_production_theorem_l3743_374391

/-- The apple production problem -/
theorem apple_production_theorem :
  let first_year : ℕ := 40
  let second_year : ℕ := 2 * first_year + 8
  let third_year : ℕ := (3 * second_year) / 4
  first_year + second_year + third_year = 194 := by
sorry

end apple_production_theorem_l3743_374391


namespace rounding_estimate_greater_l3743_374395

theorem rounding_estimate_greater (x y z x' y' z' : ℤ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hx' : x' ≥ x) (hy' : y' ≤ y) (hz' : z' ≤ z) :
  2 * ((x' : ℚ) / y' - z') > 2 * ((x : ℚ) / y - z) :=
sorry

end rounding_estimate_greater_l3743_374395


namespace angle_equality_l3743_374332

theorem angle_equality (θ : Real) (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : Real.sqrt 2 * Real.sin (20 * π / 180) = Real.cos θ - Real.sin θ) : 
  θ = 25 * π / 180 := by
  sorry

end angle_equality_l3743_374332


namespace sin_cos_equation_solution_l3743_374392

theorem sin_cos_equation_solution (x : Real) : 
  0 ≤ x ∧ x < 2 * Real.pi →
  (Real.sin x)^4 - (Real.cos x)^4 = 1 / (Real.cos x) - 1 / (Real.sin x) ↔ 
  x = Real.pi / 4 ∨ x = 5 * Real.pi / 4 :=
by sorry

end sin_cos_equation_solution_l3743_374392


namespace three_true_propositions_l3743_374325

theorem three_true_propositions :
  (∀ (x : ℝ), x^2 + 1 > 0) ∧
  (∃ (x : ℤ), x^3 < 1) ∧
  (∀ (x : ℚ), x^2 ≠ 2) ∧
  ¬(∀ (x : ℕ), x^4 ≥ 1) := by
  sorry

end three_true_propositions_l3743_374325


namespace pavan_total_distance_l3743_374326

/-- Represents a segment of a journey -/
structure Segment where
  speed : ℝ
  time : ℝ

/-- Calculates the distance traveled in a segment -/
def distance_traveled (s : Segment) : ℝ := s.speed * s.time

/-- Represents Pavan's journey -/
def pavan_journey : List Segment := [
  { speed := 30, time := 4 },
  { speed := 35, time := 5 },
  { speed := 25, time := 6 },
  { speed := 40, time := 5 }
]

/-- The total travel time -/
def total_time : ℝ := 20

/-- Theorem stating the total distance traveled by Pavan -/
theorem pavan_total_distance :
  (pavan_journey.map distance_traveled).sum = 645 := by
  sorry

end pavan_total_distance_l3743_374326


namespace arithmetic_sequence_ninth_term_l3743_374389

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_third : a 3 = 7)
  (h_sixth : a 6 = 16) :
  a 9 = 25 := by
  sorry

end arithmetic_sequence_ninth_term_l3743_374389


namespace pat_has_42_cookies_l3743_374393

-- Define the given conditions
def candy : ℕ := 63
def brownies : ℕ := 21
def family_members : ℕ := 7
def dessert_per_person : ℕ := 18

-- Define the total dessert needed
def total_dessert : ℕ := family_members * dessert_per_person

-- Define the number of cookies
def cookies : ℕ := total_dessert - (candy + brownies)

-- Theorem to prove
theorem pat_has_42_cookies : cookies = 42 := by
  sorry

end pat_has_42_cookies_l3743_374393


namespace count_specially_monotonous_is_65_l3743_374345

/-- A number is specially monotonous if all its digits are either all even or all odd,
    and the digits form either a strictly increasing or a strictly decreasing sequence
    when read from left to right. --/
def SpeciallyMonotonous (n : ℕ) : Prop := sorry

/-- The set of digits we consider (0 to 8) --/
def Digits : Set ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8}

/-- Count of specially monotonous numbers with digits from 0 to 8 --/
def CountSpeciallyMonotonous : ℕ := sorry

/-- Theorem stating that the count of specially monotonous numbers is 65 --/
theorem count_specially_monotonous_is_65 : CountSpeciallyMonotonous = 65 := by sorry

end count_specially_monotonous_is_65_l3743_374345


namespace cupcakes_frosted_l3743_374311

-- Define the frosting rates and working time
def cagney_rate : ℚ := 1 / 25
def lacey_rate : ℚ := 1 / 35
def pat_rate : ℚ := 1 / 45
def working_time : ℕ := 10 * 60  -- 10 minutes in seconds

-- Theorem statement
theorem cupcakes_frosted : 
  ∃ (n : ℕ), n = 54 ∧ 
  (n : ℚ) ≤ (cagney_rate + lacey_rate + pat_rate) * working_time ∧
  (n + 1 : ℚ) > (cagney_rate + lacey_rate + pat_rate) * working_time :=
by sorry

end cupcakes_frosted_l3743_374311


namespace expression_value_l3743_374329

theorem expression_value (a b c : ℚ) (ha : a = 5) (hb : b = -3) (hc : c = 2) :
  (3 * c) / (a + b) + c = 5 := by
  sorry

end expression_value_l3743_374329


namespace arithmetic_sequence_a6_l3743_374371

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a6 (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_a2 : a 2 = 4)
  (h_a4 : a 4 = 2) :
  a 6 = 0 := by
  sorry

end arithmetic_sequence_a6_l3743_374371


namespace simplify_fraction_l3743_374334

theorem simplify_fraction : 25 * (9 / 14) * (2 / 27) = 25 / 21 := by
  sorry

end simplify_fraction_l3743_374334


namespace not_perfect_cube_1967_l3743_374383

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem not_perfect_cube_1967 :
  ∀ (p : Fin 1967 → Fin 1967), Function.Bijective p →
    ¬ (∃ (k : ℕ), sum_of_first_n 1967 = k^3) :=
by sorry

end not_perfect_cube_1967_l3743_374383


namespace multiplication_proof_l3743_374331

theorem multiplication_proof : 
  ∃ (a b : ℕ), 
    a * b = 4485 ∧
    a = 23 ∧
    b = 195 ∧
    (b % 10) * a = 115 ∧
    ((b / 10) % 10) * a = 207 ∧
    (b / 100) * a = 23 :=
by
  sorry

end multiplication_proof_l3743_374331


namespace unique_root_condition_l3743_374339

/-- The equation has exactly one root if and only if p = 3q/4 and q ≠ 0 -/
theorem unique_root_condition (p q : ℝ) : 
  (∃! x : ℝ, (2*x - 2*p + q)/(2*x - 2*p - q) = (2*q + p + x)/(2*q - p - x)) ↔ 
  (p = 3*q/4 ∧ q ≠ 0) := by
sorry

end unique_root_condition_l3743_374339


namespace product_of_numbers_with_given_sum_and_difference_l3743_374316

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 27 ∧ x - y = 7 → x * y = 170 := by
sorry

end product_of_numbers_with_given_sum_and_difference_l3743_374316


namespace second_number_calculation_l3743_374366

theorem second_number_calculation (first_number : ℝ) (second_number : ℝ) : 
  first_number = 640 → 
  (0.5 * first_number) = (0.2 * second_number + 190) → 
  second_number = 650 := by
sorry

end second_number_calculation_l3743_374366


namespace football_joins_l3743_374355

theorem football_joins (pentagonal_panels hexagonal_panels : ℕ) 
  (pentagonal_edges hexagonal_edges : ℕ) : 
  pentagonal_panels = 12 →
  hexagonal_panels = 20 →
  pentagonal_edges = 5 →
  hexagonal_edges = 6 →
  (pentagonal_panels * pentagonal_edges + hexagonal_panels * hexagonal_edges) / 2 = 90 := by
  sorry

end football_joins_l3743_374355


namespace problem_solution_l3743_374343

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.exp (-x)) / a + a / (Real.exp (-x))

theorem problem_solution (a : ℝ) (h_a : a > 0) 
  (h_even : ∀ x, f a x = f a (-x)) :
  (a = 1) ∧
  (∀ x y, x ≥ 0 → y ≥ 0 → x < y → f a x < f a y) ∧
  (∀ m, (∀ x, f 1 x - m^2 + m ≥ 0) ↔ -1 ≤ m ∧ m ≤ 2) :=
by sorry

end problem_solution_l3743_374343


namespace distance_between_points_l3743_374314

/-- The distance between two points with the same x-coordinate in a Cartesian coordinate system. -/
def distance_same_x (y₁ y₂ : ℝ) : ℝ := |y₂ - y₁|

/-- Theorem stating that the distance between (3,-2) and (3,1) is 3. -/
theorem distance_between_points : distance_same_x (-2) 1 = 3 := by sorry

end distance_between_points_l3743_374314


namespace original_selling_price_l3743_374307

theorem original_selling_price (cost_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) :
  cost_price = 12500 ∧ discount_rate = 0.1 ∧ profit_rate = 0.08 →
  ∃ (selling_price : ℝ), selling_price = 15000 ∧
    (1 - discount_rate) * selling_price = (1 + profit_rate) * cost_price :=
by sorry

end original_selling_price_l3743_374307


namespace hexagon_fencing_cost_l3743_374337

/-- The cost of fencing an irregular hexagonal field -/
theorem hexagon_fencing_cost (side1 side2 side3 side4 side5 side6 : ℝ)
  (cost_first_three : ℝ) (cost_last_three : ℝ) :
  side1 = 20 ∧ side2 = 15 ∧ side3 = 25 ∧ side4 = 30 ∧ side5 = 10 ∧ side6 = 35 ∧
  cost_first_three = 3.5 ∧ cost_last_three = 4 →
  (side1 + side2 + side3) * cost_first_three + (side4 + side5 + side6) * cost_last_three = 510 :=
by sorry

end hexagon_fencing_cost_l3743_374337


namespace ratio_c_to_a_is_sqrt2_l3743_374319

/-- A configuration of four points on a plane -/
structure PointConfiguration where
  /-- The length of four segments -/
  a : ℝ
  /-- The length of the longest segment -/
  longest : ℝ
  /-- The length of the remaining segment -/
  c : ℝ
  /-- The longest segment is twice the length of a -/
  longest_eq_2a : longest = 2 * a
  /-- The configuration contains a 45-45-90 triangle -/
  has_45_45_90_triangle : True
  /-- The hypotenuse of the 45-45-90 triangle is the longest segment -/
  hypotenuse_is_longest : True
  /-- All points are distinct -/
  points_distinct : True

/-- The ratio of c to a in the given point configuration is √2 -/
theorem ratio_c_to_a_is_sqrt2 (config : PointConfiguration) : 
  config.c / config.a = Real.sqrt 2 := by
  sorry

end ratio_c_to_a_is_sqrt2_l3743_374319


namespace solutions_to_quartic_equation_l3743_374388

theorem solutions_to_quartic_equation :
  {x : ℂ | x^4 - 16 = 0} = {2, -2, 2*I, -2*I} := by sorry

end solutions_to_quartic_equation_l3743_374388


namespace sufficient_not_necessary_l3743_374321

theorem sufficient_not_necessary : 
  (∃ x : ℝ, |x - 2| < 3 ∧ ¬(0 < x ∧ x < 5)) ∧
  (∀ x : ℝ, (0 < x ∧ x < 5) → |x - 2| < 3) :=
by sorry

end sufficient_not_necessary_l3743_374321


namespace shaded_area_semicircle_pattern_l3743_374377

/-- The area of the shaded region formed by semicircles in a pattern -/
theorem shaded_area_semicircle_pattern (diameter : ℝ) (pattern_length : ℝ) : 
  diameter = 3 →
  pattern_length = 18 →
  (pattern_length / diameter) * (π * (diameter / 2)^2 / 2) = 27 / 4 * π := by
  sorry


end shaded_area_semicircle_pattern_l3743_374377


namespace pool_filling_time_l3743_374348

def spring1_rate : ℚ := 1
def spring2_rate : ℚ := 1/2
def spring3_rate : ℚ := 1/3
def spring4_rate : ℚ := 1/4

def combined_rate : ℚ := spring1_rate + spring2_rate + spring3_rate + spring4_rate

theorem pool_filling_time : (1 : ℚ) / combined_rate = 12/25 := by
  sorry

end pool_filling_time_l3743_374348


namespace quadratic_equation_with_prime_roots_l3743_374354

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

/-- The theorem statement -/
theorem quadratic_equation_with_prime_roots (a b : ℤ) :
  (∃ x y : ℕ, x ≠ y ∧ isPrime x ∧ isPrime y ∧ 
    (a : ℚ) * x^2 + (b : ℚ) * x - 2008 = 0 ∧ 
    (a : ℚ) * y^2 + (b : ℚ) * y - 2008 = 0) →
  3 * a + b = 1000 := by
sorry

end quadratic_equation_with_prime_roots_l3743_374354


namespace equation_equivalence_l3743_374396

theorem equation_equivalence (x : ℝ) : x^2 - 6*x + 5 = 0 ↔ (x - 3)^2 = 14 := by
  sorry

end equation_equivalence_l3743_374396
