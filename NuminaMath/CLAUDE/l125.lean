import Mathlib

namespace NUMINAMATH_CALUDE_function_positive_range_l125_12591

-- Define the function f(x) = -x^2 + 2x + 3
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- State the theorem
theorem function_positive_range :
  ∀ x : ℝ, f x > 0 ↔ -1 < x ∧ x < 3 := by
sorry

end NUMINAMATH_CALUDE_function_positive_range_l125_12591


namespace NUMINAMATH_CALUDE_goldbach_140_max_diff_l125_12505

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem goldbach_140_max_diff :
  ∀ p q : ℕ,
    is_prime p →
    is_prime q →
    p + q = 140 →
    p < q →
    p < 50 →
    q - p ≤ 134 :=
by sorry

end NUMINAMATH_CALUDE_goldbach_140_max_diff_l125_12505


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l125_12548

theorem solve_exponential_equation :
  ∃ x : ℝ, (5 : ℝ) ^ (3 * x) = Real.sqrt 125 ∧ x = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l125_12548


namespace NUMINAMATH_CALUDE_determinant_sin_matrix_l125_12561

theorem determinant_sin_matrix (a b : Real) : 
  Matrix.det !![1, Real.sin (a - b), Real.sin a; 
                 Real.sin (a - b), 1, Real.sin b; 
                 Real.sin a, Real.sin b, 1] = 0 := by
  sorry

end NUMINAMATH_CALUDE_determinant_sin_matrix_l125_12561


namespace NUMINAMATH_CALUDE_only_negative_four_less_than_negative_three_l125_12514

theorem only_negative_four_less_than_negative_three :
  let numbers : List ℝ := [-4, -2.8, 0, |-4|]
  ∀ x ∈ numbers, x < -3 ↔ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_only_negative_four_less_than_negative_three_l125_12514


namespace NUMINAMATH_CALUDE_even_function_product_nonnegative_l125_12588

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

theorem even_function_product_nonnegative
  (f : ℝ → ℝ) (h : is_even_function f) :
  ∀ x : ℝ, f x * f (-x) ≥ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_even_function_product_nonnegative_l125_12588


namespace NUMINAMATH_CALUDE_no_all_prime_arrangement_l125_12532

/-- A card with two digits -/
structure Card where
  digit1 : Nat
  digit2 : Nat
  h_different : digit1 ≠ digit2
  h_range : digit1 < 10 ∧ digit2 < 10

/-- Function to form a two-digit number from two digits -/
def formNumber (tens : Nat) (ones : Nat) : Nat :=
  10 * tens + ones

/-- Predicate to check if a number is prime -/
def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

/-- Main theorem statement -/
theorem no_all_prime_arrangement :
  ¬∃ (card1 card2 : Card),
    card1.digit1 ≠ card2.digit1 ∧
    card1.digit1 ≠ card2.digit2 ∧
    card1.digit2 ≠ card2.digit1 ∧
    card1.digit2 ≠ card2.digit2 ∧
    (∀ (d1 d2 : Nat),
      (d1 = card1.digit1 ∨ d1 = card1.digit2 ∨ d1 = card2.digit1 ∨ d1 = card2.digit2) →
      (d2 = card1.digit1 ∨ d2 = card1.digit2 ∨ d2 = card2.digit1 ∨ d2 = card2.digit2) →
      isPrime (formNumber d1 d2)) :=
sorry

end NUMINAMATH_CALUDE_no_all_prime_arrangement_l125_12532


namespace NUMINAMATH_CALUDE_car_expense_difference_l125_12594

-- Define Alberto's expenses
def alberto_engine : ℝ := 2457
def alberto_transmission : ℝ := 374
def alberto_tires : ℝ := 520
def alberto_discount_rate : ℝ := 0.05

-- Define Samara's expenses
def samara_oil : ℝ := 25
def samara_tires : ℝ := 467
def samara_detailing : ℝ := 79
def samara_stereo : ℝ := 150
def samara_tax_rate : ℝ := 0.07

-- Theorem statement
theorem car_expense_difference : 
  let alberto_total := alberto_engine + alberto_transmission + alberto_tires
  let alberto_discount := alberto_total * alberto_discount_rate
  let alberto_final := alberto_total - alberto_discount
  let samara_total := samara_oil + samara_tires + samara_detailing + samara_stereo
  let samara_tax := samara_total * samara_tax_rate
  let samara_final := samara_total + samara_tax
  alberto_final - samara_final = 2411.98 := by
    sorry

end NUMINAMATH_CALUDE_car_expense_difference_l125_12594


namespace NUMINAMATH_CALUDE_perfect_squares_ending_in_444_and_4444_l125_12587

def ends_in_444 (n : ℕ) : Prop := n % 1000 = 444

def ends_in_4444 (n : ℕ) : Prop := n % 10000 = 4444

theorem perfect_squares_ending_in_444_and_4444 :
  (∀ a : ℕ, (∃ k : ℕ, a * a = k) ∧ ends_in_444 (a * a) ↔ ∃ n : ℕ, a = 500 * n + 38 ∨ a = 500 * n - 38) ∧
  (¬ ∃ a : ℕ, (∃ k : ℕ, a * a = k) ∧ ends_in_4444 (a * a)) :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_ending_in_444_and_4444_l125_12587


namespace NUMINAMATH_CALUDE_son_age_proof_l125_12545

-- Define the variables
def your_age : ℕ := 45
def son_age : ℕ := 15

-- Define the conditions
theorem son_age_proof :
  (your_age = 3 * son_age) ∧
  (your_age + 5 = (5/2) * (son_age + 5)) →
  son_age = 15 := by
sorry


end NUMINAMATH_CALUDE_son_age_proof_l125_12545


namespace NUMINAMATH_CALUDE_divisors_of_power_minus_one_l125_12554

theorem divisors_of_power_minus_one (a b r : ℕ) (ha : a ≥ 2) (hb : b > 0) (hb_composite : ∃ x y, 1 < x ∧ 1 < y ∧ b = x * y) (hr : ∃ (S : Finset ℕ), S.card = r ∧ ∀ x ∈ S, x > 0 ∧ x ∣ b) :
  ∃ (T : Finset ℕ), T.card ≥ r ∧ ∀ x ∈ T, x > 0 ∧ x ∣ (a^b - 1) :=
sorry

end NUMINAMATH_CALUDE_divisors_of_power_minus_one_l125_12554


namespace NUMINAMATH_CALUDE_function_range_theorem_l125_12586

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + 2 * x - a

theorem function_range_theorem (a : ℝ) :
  (∃ x₀ y₀ : ℝ, y₀ = Real.sin x₀ ∧ f a (f a y₀) = y₀) →
  a ∈ Set.Icc (Real.exp (-1) - 1) (Real.exp 1 + 1) :=
by sorry

end NUMINAMATH_CALUDE_function_range_theorem_l125_12586


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l125_12531

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel 
  (a b : Line) (α : Plane) :
  perpendicular a α → perpendicular b α → parallel a b :=
sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l125_12531


namespace NUMINAMATH_CALUDE_cos_alpha_values_l125_12521

theorem cos_alpha_values (α : Real) (h : Real.sin (Real.pi + α) = -3/5) :
  Real.cos α = 4/5 ∨ Real.cos α = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_values_l125_12521


namespace NUMINAMATH_CALUDE_complementary_angles_theorem_l125_12549

theorem complementary_angles_theorem (α β : Real) : 
  (α + β = 180) →  -- complementary angles
  (α - β / 2 = 30) →  -- half of β is 30° less than α
  (α = 80) :=  -- measure of α is 80°
by sorry

end NUMINAMATH_CALUDE_complementary_angles_theorem_l125_12549


namespace NUMINAMATH_CALUDE_plot_length_is_60_l125_12576

/-- Represents a rectangular plot with its dimensions and fencing cost. -/
structure RectangularPlot where
  breadth : ℝ
  lengthDifference : ℝ
  fencingCostPerMeter : ℝ
  totalFencingCost : ℝ

/-- Calculates the length of the plot given its properties. -/
def plotLength (plot : RectangularPlot) : ℝ :=
  plot.breadth + plot.lengthDifference

/-- Calculates the perimeter of the plot. -/
def plotPerimeter (plot : RectangularPlot) : ℝ :=
  2 * (plotLength plot + plot.breadth)

/-- The main theorem stating the length of the plot under given conditions. -/
theorem plot_length_is_60 (plot : RectangularPlot) 
  (h1 : plot.lengthDifference = 20)
  (h2 : plot.fencingCostPerMeter = 26.5)
  (h3 : plot.totalFencingCost = 5300)
  (h4 : plotPerimeter plot = plot.totalFencingCost / plot.fencingCostPerMeter) :
  plotLength plot = 60 := by
  sorry


end NUMINAMATH_CALUDE_plot_length_is_60_l125_12576


namespace NUMINAMATH_CALUDE_specific_seating_arrangements_l125_12577

/-- Represents the seating arrangement in a theater -/
structure TheaterSeating where
  front_row : ℕ
  back_row : ℕ
  unusable_middle_seats : ℕ

/-- Calculates the number of ways to seat two people in the theater -/
def seating_arrangements (theater : TheaterSeating) : ℕ :=
  sorry

/-- Theorem stating the number of seating arrangements for the given problem -/
theorem specific_seating_arrangements :
  let theater : TheaterSeating := {
    front_row := 10,
    back_row := 11,
    unusable_middle_seats := 3
  }
  seating_arrangements theater = 276 := by
  sorry

end NUMINAMATH_CALUDE_specific_seating_arrangements_l125_12577


namespace NUMINAMATH_CALUDE_monomial_properties_l125_12558

/-- Represents a monomial in variables a and b -/
structure Monomial where
  coeff : ℤ
  a_exp : ℕ
  b_exp : ℕ

/-- The coefficient of a monomial -/
def coefficient (m : Monomial) : ℤ := m.coeff

/-- The degree of a monomial -/
def degree (m : Monomial) : ℕ := m.a_exp + m.b_exp

/-- The monomial 2a²b -/
def m : Monomial := { coeff := 2, a_exp := 2, b_exp := 1 }

theorem monomial_properties :
  coefficient m = 2 ∧ degree m = 3 := by sorry

end NUMINAMATH_CALUDE_monomial_properties_l125_12558


namespace NUMINAMATH_CALUDE_tabitha_current_age_l125_12516

/- Define the problem parameters -/
def start_age : ℕ := 15
def start_colors : ℕ := 2
def future_colors : ℕ := 8
def years_to_future : ℕ := 3

/- Define Tabitha's age as a function of the number of colors -/
def tabitha_age (colors : ℕ) : ℕ := start_age + (colors - start_colors)

/- Define the number of colors Tabitha has now -/
def current_colors : ℕ := future_colors - years_to_future

/- The theorem to prove -/
theorem tabitha_current_age :
  tabitha_age current_colors = 18 := by
  sorry


end NUMINAMATH_CALUDE_tabitha_current_age_l125_12516


namespace NUMINAMATH_CALUDE_S_inter_T_finite_l125_12522

/-- Set S defined as {y | y = 3^x, x ∈ ℝ} -/
def S : Set ℝ := {y | ∃ x, y = Real.exp (Real.log 3 * x)}

/-- Set T defined as {y | y = x^2 - 1, x ∈ ℝ} -/
def T : Set ℝ := {y | ∃ x, y = x^2 - 1}

/-- The intersection of S and T is a finite set -/
theorem S_inter_T_finite : Set.Finite (S ∩ T) := by sorry

end NUMINAMATH_CALUDE_S_inter_T_finite_l125_12522


namespace NUMINAMATH_CALUDE_band_members_count_l125_12509

theorem band_members_count : ∃! N : ℕ, 
  100 < N ∧ N < 200 ∧ 
  (∃ k : ℕ, N + 2 = 8 * k) ∧ 
  (∃ m : ℕ, N + 3 = 9 * m) ∧ 
  N = 150 := by
  sorry

end NUMINAMATH_CALUDE_band_members_count_l125_12509


namespace NUMINAMATH_CALUDE_group_purchase_equation_system_l125_12579

/-- Represents a group purchase scenario -/
structure GroupPurchase where
  people : ℕ
  price : ℕ
  excess_9 : ℕ
  shortage_6 : ℕ

/-- The group purchase scenario satisfies the given conditions -/
def satisfies_conditions (gp : GroupPurchase) : Prop :=
  9 * gp.people - gp.price = gp.excess_9 ∧
  gp.price - 6 * gp.people = gp.shortage_6

/-- The system of equations correctly represents the group purchase scenario -/
theorem group_purchase_equation_system (gp : GroupPurchase) 
  (h : satisfies_conditions gp) (h_excess : gp.excess_9 = 4) (h_shortage : gp.shortage_6 = 5) :
  9 * gp.people - gp.price = 4 ∧ gp.price - 6 * gp.people = 5 := by
  sorry

#check group_purchase_equation_system

end NUMINAMATH_CALUDE_group_purchase_equation_system_l125_12579


namespace NUMINAMATH_CALUDE_num_lines_in_4x4_grid_l125_12518

/-- Represents a 4-by-4 grid of lattice points -/
structure Grid :=
  (size : Nat)
  (h_size : size = 4)

/-- Represents a line in the grid -/
structure Line :=
  (points : Finset (Nat × Nat))
  (h_distinct : points.card ≥ 2)
  (h_in_grid : ∀ p ∈ points, p.1 < 4 ∧ p.2 < 4)

/-- The set of all lines in the grid -/
def allLines (g : Grid) : Finset Line := sorry

/-- The number of distinct lines passing through at least two points in a 4-by-4 grid -/
def numLines (g : Grid) : Nat :=
  (allLines g).card

/-- Theorem stating that the number of distinct lines in a 4-by-4 grid is 70 -/
theorem num_lines_in_4x4_grid (g : Grid) : numLines g = 70 := by
  sorry

end NUMINAMATH_CALUDE_num_lines_in_4x4_grid_l125_12518


namespace NUMINAMATH_CALUDE_cos_300_degrees_l125_12541

theorem cos_300_degrees (θ : Real) : 
  θ = 300 * Real.pi / 180 → Real.cos θ = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_300_degrees_l125_12541


namespace NUMINAMATH_CALUDE_shadow_length_sams_shadow_length_l125_12581

/-- Given a lamp post and a person walking towards it, this theorem calculates
    the length of the person's shadow at a new position. -/
theorem shadow_length (lamp_height : ℝ) (initial_distance : ℝ) (initial_shadow : ℝ) 
                      (new_distance : ℝ) : ℝ :=
  let person_height := lamp_height * initial_shadow / (initial_distance + initial_shadow)
  let new_shadow := person_height * new_distance / (lamp_height - person_height)
  new_shadow

/-- The main theorem that proves the specific shadow length for the given scenario. -/
theorem sams_shadow_length : 
  shadow_length 8 12 4 8 = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_shadow_length_sams_shadow_length_l125_12581


namespace NUMINAMATH_CALUDE_product_of_first_three_odd_numbers_l125_12546

theorem product_of_first_three_odd_numbers : 
  (∀ a b c : ℕ, a * b * c = 38 → a = 3 ∧ b = 5 ∧ c = 7) →
  (∀ x y z : ℕ, x * y * z = 268 → x = 13 ∧ y = 15 ∧ z = 17) →
  1 * 3 * 5 = 15 :=
by sorry

end NUMINAMATH_CALUDE_product_of_first_three_odd_numbers_l125_12546


namespace NUMINAMATH_CALUDE_f_monotonically_decreasing_l125_12599

-- Define the function f(x) = e^x / x
noncomputable def f (x : ℝ) : ℝ := Real.exp x / x

-- Define the property of being monotonically decreasing on an interval
def MonotonicallyDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x → x < y → y < b → f y < f x

-- State the theorem
theorem f_monotonically_decreasing :
  (∀ x y, x < y → y < 0 → f y < f x) ∧
  (∀ x y, 0 < x → x < y → y ≤ 1 → f y < f x) :=
sorry

end NUMINAMATH_CALUDE_f_monotonically_decreasing_l125_12599


namespace NUMINAMATH_CALUDE_sum_of_factors_360_l125_12567

/-- The sum of positive factors of a natural number n -/
def sum_of_factors (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of the positive factors of 360 is 1170 -/
theorem sum_of_factors_360 : sum_of_factors 360 = 1170 := by sorry

end NUMINAMATH_CALUDE_sum_of_factors_360_l125_12567


namespace NUMINAMATH_CALUDE_correct_sum_l125_12556

theorem correct_sum (x y : ℕ+) (h1 : x - y = 4) (h2 : x * y = 132) : x + y = 24 := by
  sorry

end NUMINAMATH_CALUDE_correct_sum_l125_12556


namespace NUMINAMATH_CALUDE_total_tickets_sold_l125_12596

/-- Represents the ticket sales for a theater performance --/
structure TicketSales where
  orchestra : ℕ
  balcony : ℕ

/-- The pricing and sales data for the theater --/
def theaterData : TicketSales → Prop := fun ts =>
  12 * ts.orchestra + 8 * ts.balcony = 3320 ∧
  ts.balcony = ts.orchestra + 240

/-- Theorem stating that the total number of tickets sold is 380 --/
theorem total_tickets_sold (ts : TicketSales) (h : theaterData ts) : 
  ts.orchestra + ts.balcony = 380 := by
  sorry

#check total_tickets_sold

end NUMINAMATH_CALUDE_total_tickets_sold_l125_12596


namespace NUMINAMATH_CALUDE_square_difference_sum_l125_12508

theorem square_difference_sum : 
  21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 = 240 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_sum_l125_12508


namespace NUMINAMATH_CALUDE_pure_imaginary_solutions_l125_12552

-- Define the polynomial
def p (x : ℂ) : ℂ := x^4 + 2*x^3 + 6*x^2 + 34*x + 49

-- State the theorem
theorem pure_imaginary_solutions :
  p (Complex.I * Real.sqrt 17) = 0 ∧ p (-Complex.I * Real.sqrt 17) = 0 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_solutions_l125_12552


namespace NUMINAMATH_CALUDE_bricks_per_square_meter_l125_12592

-- Define the parameters
def num_rooms : ℕ := 5
def room_length : ℝ := 4
def room_width : ℝ := 5
def room_height : ℝ := 2
def bricks_per_room : ℕ := 340

-- Define the theorem
theorem bricks_per_square_meter :
  let room_area : ℝ := room_length * room_width
  let bricks_per_sq_meter : ℝ := bricks_per_room / room_area
  bricks_per_sq_meter = 17 := by sorry

end NUMINAMATH_CALUDE_bricks_per_square_meter_l125_12592


namespace NUMINAMATH_CALUDE_permutation_of_two_equals_twelve_l125_12544

theorem permutation_of_two_equals_twelve (n : ℕ) : n * (n - 1) = 12 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_permutation_of_two_equals_twelve_l125_12544


namespace NUMINAMATH_CALUDE_round_85960_to_three_sig_figs_l125_12529

/-- Rounds a number to a specified number of significant figures using the round-half-up method -/
def roundToSigFigs (x : ℝ) (sigFigs : ℕ) : ℝ :=
  sorry

/-- Theorem: Rounding 85960 to three significant figures using the round-half-up method results in 8.60 × 10^4 -/
theorem round_85960_to_three_sig_figs :
  roundToSigFigs 85960 3 = 8.60 * (10 : ℝ)^4 :=
sorry

end NUMINAMATH_CALUDE_round_85960_to_three_sig_figs_l125_12529


namespace NUMINAMATH_CALUDE_tan_255_degrees_l125_12535

theorem tan_255_degrees : Real.tan (255 * π / 180) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_255_degrees_l125_12535


namespace NUMINAMATH_CALUDE_quadratic_function_property_l125_12506

theorem quadratic_function_property (b c : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + b*x + c
  (f 1 = 0) → (f 3 = 0) → (f (-1) = 8) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l125_12506


namespace NUMINAMATH_CALUDE_cafe_tables_l125_12528

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : ℕ) : ℕ := sorry

/-- Calculates the number of tables needed given the number of people and people per table -/
def tablesNeeded (people : ℕ) (peoplePerTable : ℕ) : ℕ := 
  (people + peoplePerTable - 1) / peoplePerTable

theorem cafe_tables : 
  let totalPeople : ℕ := base7ToBase10 310
  let peoplePerTable : ℕ := 3
  tablesNeeded totalPeople peoplePerTable = 52 := by sorry

end NUMINAMATH_CALUDE_cafe_tables_l125_12528


namespace NUMINAMATH_CALUDE_roots_properties_l125_12582

theorem roots_properties (a b m : ℝ) (h1 : 2 * a^2 - 8 * a + m = 0)
                                    (h2 : 2 * b^2 - 8 * b + m = 0)
                                    (h3 : m > 0) :
  (a^2 + b^2 ≥ 8) ∧
  (Real.sqrt a + Real.sqrt b ≤ 2 * Real.sqrt 2) ∧
  (1 / (a + 2) + 1 / (2 * b) ≥ (3 + 2 * Real.sqrt 2) / 12) := by
  sorry

end NUMINAMATH_CALUDE_roots_properties_l125_12582


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l125_12540

theorem triangle_area_theorem (x : ℝ) (h1 : x > 0) :
  (1/2 : ℝ) * x * 3*x = 72 → x = 4 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l125_12540


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l125_12583

/-- A right circular cone with a sphere inscribed inside it. -/
structure ConeWithSphere where
  /-- The diameter of the cone's base in inches. -/
  base_diameter : ℝ
  /-- The vertex angle of the cross-section triangle in degrees. -/
  vertex_angle : ℝ
  /-- The sphere is tangent to the sides of the cone and rests on the table. -/
  sphere_tangent : Bool

/-- The volume of the inscribed sphere in cubic inches. -/
def sphere_volume (cone : ConeWithSphere) : ℝ := sorry

/-- Theorem stating the volume of the inscribed sphere for specific cone dimensions. -/
theorem inscribed_sphere_volume (cone : ConeWithSphere) 
  (h1 : cone.base_diameter = 24)
  (h2 : cone.vertex_angle = 90)
  (h3 : cone.sphere_tangent = true) :
  sphere_volume cone = 2304 * Real.pi := by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l125_12583


namespace NUMINAMATH_CALUDE_train_passing_time_l125_12575

/-- Prove that a train with the given length and speed takes the specified time to pass a stationary point. -/
theorem train_passing_time
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (h1 : train_length = 70)
  (h2 : train_speed_kmh = 36) :
  (train_length / (train_speed_kmh * 1000 / 3600)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l125_12575


namespace NUMINAMATH_CALUDE_product_from_lcm_gcd_l125_12517

theorem product_from_lcm_gcd : 
  ∀ a b : ℤ, (Nat.lcm a.natAbs b.natAbs = 72) → (Int.gcd a b = 8) → a * b = 576 := by
  sorry

end NUMINAMATH_CALUDE_product_from_lcm_gcd_l125_12517


namespace NUMINAMATH_CALUDE_common_root_is_neg_half_l125_12555

/-- Definition of the first polynomial -/
def p (a b c : ℝ) (x : ℝ) : ℝ := 50 * x^4 + a * x^3 + b * x^2 + c * x + 16

/-- Definition of the second polynomial -/
def q (d e f g : ℝ) (x : ℝ) : ℝ := 16 * x^5 + d * x^4 + e * x^3 + f * x^2 + g * x + 50

/-- Theorem stating that if p and q have a common negative rational root, it must be -1/2 -/
theorem common_root_is_neg_half (a b c d e f g : ℝ) :
  (∃ (k : ℚ), k < 0 ∧ p a b c k = 0 ∧ q d e f g k = 0) →
  (p a b c (-1/2 : ℚ) = 0 ∧ q d e f g (-1/2 : ℚ) = 0) :=
by sorry

end NUMINAMATH_CALUDE_common_root_is_neg_half_l125_12555


namespace NUMINAMATH_CALUDE_alan_eggs_purchase_l125_12539

theorem alan_eggs_purchase (egg_price chicken_price total_spent : ℚ) 
  (num_chickens : ℕ) (h1 : egg_price = 2) 
  (h2 : chicken_price = 8) (h3 : num_chickens = 6) 
  (h4 : total_spent = 88) : ∃ (num_eggs : ℕ), 
  num_eggs = 20 ∧ 
  (egg_price * num_eggs + chicken_price * num_chickens : ℚ) = total_spent :=
sorry

end NUMINAMATH_CALUDE_alan_eggs_purchase_l125_12539


namespace NUMINAMATH_CALUDE_sum_of_floors_l125_12526

def floor (x : ℚ) : ℤ := Int.floor x

theorem sum_of_floors : 
  (floor (2017 * 3 / 11 : ℚ)) + 
  (floor (2017 * 4 / 11 : ℚ)) + 
  (floor (2017 * 5 / 11 : ℚ)) + 
  (floor (2017 * 6 / 11 : ℚ)) + 
  (floor (2017 * 7 / 11 : ℚ)) + 
  (floor (2017 * 8 / 11 : ℚ)) = 6048 := by
sorry

end NUMINAMATH_CALUDE_sum_of_floors_l125_12526


namespace NUMINAMATH_CALUDE_equation_one_solution_system_of_equations_solution_l125_12557

-- Equation (1)
theorem equation_one_solution :
  ∃! x : ℚ, (2*x + 5) / 6 - (3*x - 2) / 8 = 1 :=
by sorry

-- System of equations (2)
theorem system_of_equations_solution :
  ∃! (x y : ℚ), (2*x - 1) / 5 + (3*y - 2) / 4 = 2 ∧
                (3*x + 1) / 5 - (3*y + 2) / 4 = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_one_solution_system_of_equations_solution_l125_12557


namespace NUMINAMATH_CALUDE_largest_package_size_l125_12510

theorem largest_package_size (ming_pencils catherine_pencils luke_pencils : ℕ) 
  (h_ming : ming_pencils = 30)
  (h_catherine : catherine_pencils = 45)
  (h_luke : luke_pencils = 75) :
  Nat.gcd ming_pencils (Nat.gcd catherine_pencils luke_pencils) = 15 := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l125_12510


namespace NUMINAMATH_CALUDE_tan_alpha_value_l125_12598

theorem tan_alpha_value (α : Real) 
  (h1 : Real.sin (π - α) = Real.sqrt 5 / 5)
  (h2 : π / 2 < α ∧ α < π) : 
  Real.tan α = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l125_12598


namespace NUMINAMATH_CALUDE_unique_natural_number_with_specific_properties_l125_12537

theorem unique_natural_number_with_specific_properties :
  ∀ (x n : ℕ),
    x = 5^n - 1 →
    (∃ (p q r : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ x = p * q * r) →
    11 ∣ x →
    x = 3124 := by
  sorry

end NUMINAMATH_CALUDE_unique_natural_number_with_specific_properties_l125_12537


namespace NUMINAMATH_CALUDE_min_x_prime_factorization_l125_12562

theorem min_x_prime_factorization (x y : ℕ+) (h : 5 * x^7 = 13 * y^17) :
  ∃ (a b c d : ℕ),
    x.val = a^c * b^d ∧
    a.Prime ∧ b.Prime ∧
    x.val ≥ 13^5 * 5^10 ∧
    (x.val = 13^5 * 5^10 → a + b + c + d = 33) :=
sorry

end NUMINAMATH_CALUDE_min_x_prime_factorization_l125_12562


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l125_12595

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i * i = -1 →
  (5 * i / (2 - i)).im = 2 :=
by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l125_12595


namespace NUMINAMATH_CALUDE_all_odd_in_M_product_in_M_l125_12589

-- Define the set M
def M : Set ℤ := {n : ℤ | ∃ (x y : ℤ), n = x^2 - y^2}

-- Statement 1: All odd numbers belong to M
theorem all_odd_in_M : ∀ (k : ℤ), (2 * k + 1) ∈ M := by sorry

-- Statement 3: If a ∈ M and b ∈ M, then ab ∈ M
theorem product_in_M : ∀ (a b : ℤ), a ∈ M → b ∈ M → (a * b) ∈ M := by sorry

end NUMINAMATH_CALUDE_all_odd_in_M_product_in_M_l125_12589


namespace NUMINAMATH_CALUDE_probability_white_or_red_ball_l125_12585

theorem probability_white_or_red_ball (white black red : ℕ) 
  (h_white : white = 8)
  (h_black : black = 7)
  (h_red : red = 4) :
  (white + red : ℚ) / (white + black + red) = 12 / 19 :=
by sorry

end NUMINAMATH_CALUDE_probability_white_or_red_ball_l125_12585


namespace NUMINAMATH_CALUDE_max_tiles_on_floor_l125_12503

/-- Represents the dimensions of a rectangle -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of tiles that can be placed in a checkerboard pattern -/
def maxTiles (floor : Dimensions) (tile : Dimensions) : ℕ :=
  let lengthTiles := floor.length / tile.length
  let widthTiles := floor.width / tile.width
  (lengthTiles / 2) * (widthTiles / 2)

/-- Theorem stating the maximum number of tiles that can be placed on the given floor -/
theorem max_tiles_on_floor :
  let floor := Dimensions.mk 280 240
  let tile := Dimensions.mk 40 28
  maxTiles floor tile = 12 := by
  sorry

end NUMINAMATH_CALUDE_max_tiles_on_floor_l125_12503


namespace NUMINAMATH_CALUDE_problem_statement_l125_12584

def additive_inverse (a b : ℚ) : Prop := a + b = 0

def multiplicative_inverse (b c : ℚ) : Prop := b * c = 1

def cubic_identity (m : ℚ) : Prop := m^3 = m

theorem problem_statement 
  (a b c m : ℚ) 
  (h1 : additive_inverse a b) 
  (h2 : multiplicative_inverse b c) 
  (h3 : cubic_identity m) :
  (∃ S : ℚ, 
    (2*a + 2*b) / (m + 2) + a*c = -1 ∧
    (a > 1 → m < 0 → 
      S = |2*a - 3*b| - 2*|b - m| - |b + 1/2| →
      4*(2*a - S) + 2*(2*a - S) - (2*a - S) = -25/2) ∧
    (m ≠ 0 → ∃ (max_val : ℚ), 
      (∀ (x : ℚ), |x + m| - |x - m| ≤ max_val) ∧
      (∃ (x : ℚ), |x + m| - |x - m| = max_val) ∧
      max_val = 2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l125_12584


namespace NUMINAMATH_CALUDE_max_value_of_trigonometric_function_l125_12533

theorem max_value_of_trigonometric_function :
  let y : ℝ → ℝ := λ x => Real.tan (x + 3 * Real.pi / 4) - Real.tan (x + Real.pi / 4) + Real.sin (x + Real.pi / 4)
  ∃ (max_y : ℝ), max_y = Real.sqrt 2 / 2 ∧
    ∀ x, -2 * Real.pi / 3 ≤ x ∧ x ≤ -Real.pi / 2 → y x ≤ max_y :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_trigonometric_function_l125_12533


namespace NUMINAMATH_CALUDE_tom_lasagna_noodles_l125_12543

/-- The number of packages of noodles Tom needs to buy for his lasagna -/
def noodle_packages_needed (beef_amount : ℕ) (noodle_ratio : ℕ) (existing_noodles : ℕ) (package_size : ℕ) : ℕ :=
  let total_noodles_needed := beef_amount * noodle_ratio
  let additional_noodles_needed := total_noodles_needed - existing_noodles
  (additional_noodles_needed + package_size - 1) / package_size

theorem tom_lasagna_noodles : noodle_packages_needed 10 2 4 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_tom_lasagna_noodles_l125_12543


namespace NUMINAMATH_CALUDE_complex_sum_zero_implies_b_equals_two_l125_12527

theorem complex_sum_zero_implies_b_equals_two (b : ℝ) : 
  (2 : ℂ) - Complex.I * b = (2 : ℂ) - Complex.I * b ∧ 
  (2 : ℝ) + (-b) = 0 → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_zero_implies_b_equals_two_l125_12527


namespace NUMINAMATH_CALUDE_min_value_x_plus_reciprocal_l125_12569

theorem min_value_x_plus_reciprocal (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 ∧ (x + 1/x = 2 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_reciprocal_l125_12569


namespace NUMINAMATH_CALUDE_new_average_commission_is_550_l125_12571

/-- Represents a salesperson's commission data -/
structure SalespersonData where
  totalSales : ℕ
  lastCommission : ℝ
  averageIncrease : ℝ

/-- Calculates the new average commission for a salesperson -/
def newAverageCommission (data : SalespersonData) : ℝ :=
  -- The actual calculation is not implemented here
  sorry

/-- Theorem stating that under given conditions, the new average commission is $550 -/
theorem new_average_commission_is_550 (data : SalespersonData) 
  (h1 : data.totalSales = 6)
  (h2 : data.lastCommission = 1300)
  (h3 : data.averageIncrease = 150) :
  newAverageCommission data = 550 := by
  sorry

end NUMINAMATH_CALUDE_new_average_commission_is_550_l125_12571


namespace NUMINAMATH_CALUDE_unicorn_journey_flowers_l125_12519

/-- The number of flowers that bloom when unicorns walk across a forest -/
def flowers_bloomed (num_unicorns : ℕ) (distance_km : ℕ) (step_length_m : ℕ) (flowers_per_step : ℕ) : ℕ :=
  num_unicorns * (distance_km * 1000 / step_length_m) * flowers_per_step

/-- Proof that 6 unicorns walking 9 km with 3m steps, each causing 4 flowers to bloom, results in 72000 flowers -/
theorem unicorn_journey_flowers : flowers_bloomed 6 9 3 4 = 72000 := by
  sorry

#eval flowers_bloomed 6 9 3 4

end NUMINAMATH_CALUDE_unicorn_journey_flowers_l125_12519


namespace NUMINAMATH_CALUDE_a_10_value_a_satisfies_conditions_l125_12563

def sequence_a (n : ℕ+) : ℚ :=
  1 / (3 * n - 2)

theorem a_10_value :
  sequence_a 10 = 1 / 28 :=
by sorry

theorem a_satisfies_conditions :
  sequence_a 1 = 1 ∧
  ∀ n : ℕ+, 1 / sequence_a (n + 1) = 1 / sequence_a n + 3 :=
by sorry

end NUMINAMATH_CALUDE_a_10_value_a_satisfies_conditions_l125_12563


namespace NUMINAMATH_CALUDE_x_range_l125_12513

theorem x_range (x : ℝ) (h : ∀ a > 0, x^2 < 1 + a) : -1 ≤ x ∧ x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l125_12513


namespace NUMINAMATH_CALUDE_positive_difference_of_average_l125_12573

theorem positive_difference_of_average (y : ℝ) : 
  (50 + y) / 2 = 35 → |50 - y| = 30 := by
  sorry

end NUMINAMATH_CALUDE_positive_difference_of_average_l125_12573


namespace NUMINAMATH_CALUDE_second_day_sales_correct_l125_12580

/-- Represents the sales of sportswear in a clothing store over two days -/
structure SportswearSales where
  first_day : ℕ
  second_day : ℕ

/-- Calculates the sales on the second day based on the first day's sales -/
def second_day_sales (m : ℕ) : ℕ := 2 * m - 3

/-- Theorem stating the relationship between first and second day sales -/
theorem second_day_sales_correct (sales : SportswearSales) :
  sales.first_day = m →
  sales.second_day = 2 * sales.first_day - 3 →
  sales.second_day = second_day_sales m :=
by sorry

end NUMINAMATH_CALUDE_second_day_sales_correct_l125_12580


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_is_ten_l125_12572

theorem sum_of_x_and_y_is_ten (x y : ℝ) (h1 : x = 25 / y) (h2 : x^2 + y^2 = 50) : x + y = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_is_ten_l125_12572


namespace NUMINAMATH_CALUDE_complement_of_union_l125_12500

-- Define the sets A and B
def A : Set ℝ := {x | x < 0}
def B : Set ℝ := {x | x ≥ 2}

-- Define the set C as the complement of A ∪ B in ℝ
def C : Set ℝ := (A ∪ B)ᶜ

-- Theorem statement
theorem complement_of_union :
  C = {x : ℝ | 0 ≤ x ∧ x < 2} :=
sorry

end NUMINAMATH_CALUDE_complement_of_union_l125_12500


namespace NUMINAMATH_CALUDE_angle_measure_l125_12511

theorem angle_measure (x : ℝ) : 
  (90 - x = (180 - x) / 3 + 20) → x = 75 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l125_12511


namespace NUMINAMATH_CALUDE_gum_distribution_l125_12578

theorem gum_distribution (num_cousins : ℕ) (gum_per_cousin : ℕ) (total_gum : ℕ) : 
  num_cousins = 4 → gum_per_cousin = 5 → total_gum = num_cousins * gum_per_cousin → total_gum = 20 := by
  sorry

end NUMINAMATH_CALUDE_gum_distribution_l125_12578


namespace NUMINAMATH_CALUDE_complex_magnitude_proof_l125_12536

/-- Proves that for a complex number z with argument 60°, 
    if |z-1| is the geometric mean of |z| and |z-2|, then |z| = √2 + 1 -/
theorem complex_magnitude_proof (z : ℂ) :
  Complex.arg z = π / 3 →
  Complex.abs (z - 1) ^ 2 = Complex.abs z * Complex.abs (z - 2) →
  Complex.abs z = Real.sqrt 2 + 1 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_proof_l125_12536


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l125_12501

/-- Represents a trapezoid with sides AB, BC, CD, and DA -/
structure Trapezoid where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ

/-- The perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ := t.AB + t.BC + t.CD + t.DA

/-- Theorem: Perimeter of a specific trapezoid ABCD -/
theorem trapezoid_perimeter (x y : ℝ) (hx : x ≠ 0) :
  ∃ (ABCD : Trapezoid),
    ABCD.AB = 2 * x ∧
    ABCD.CD = 4 * x ∧
    ABCD.BC = y ∧
    ABCD.DA = 2 * y ∧
    perimeter ABCD = 6 * x + 3 * y := by
  sorry

#check trapezoid_perimeter

end NUMINAMATH_CALUDE_trapezoid_perimeter_l125_12501


namespace NUMINAMATH_CALUDE_log_equation_solution_l125_12547

-- Define the logarithm function (base 10)
noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_equation_solution (a : ℝ) (h : log a - 2 * log 2 = 1) : a = 40 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l125_12547


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l125_12504

theorem polynomial_remainder_theorem (a b : ℚ) : 
  let f : ℚ → ℚ := λ x ↦ a * x^3 - 6 * x^2 + b * x - 5
  (f 2 = 3 ∧ f (-1) = 7) → (a = -2/3 ∧ b = -52/3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l125_12504


namespace NUMINAMATH_CALUDE_ratio_shoes_to_total_earned_l125_12590

def rate_per_hour : ℕ := 14
def hours_per_day : ℕ := 2
def days_worked : ℕ := 7
def money_left : ℕ := 49

def total_hours : ℕ := hours_per_day * days_worked
def total_earned : ℕ := total_hours * rate_per_hour
def money_before_mom : ℕ := money_left * 2
def money_spent_shoes : ℕ := total_earned - money_before_mom

theorem ratio_shoes_to_total_earned :
  (money_spent_shoes : ℚ) / total_earned = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_shoes_to_total_earned_l125_12590


namespace NUMINAMATH_CALUDE_anna_scores_proof_l125_12565

def anna_scores : List ℕ := [94, 87, 86, 78, 71, 58]

theorem anna_scores_proof :
  -- The list has 6 elements
  anna_scores.length = 6 ∧
  -- All elements are less than 95
  (∀ x ∈ anna_scores, x < 95) ∧
  -- All elements are different
  anna_scores.Nodup ∧
  -- The list is sorted in descending order
  anna_scores.Sorted (· ≥ ·) ∧
  -- The first three scores are 86, 78, and 71
  [86, 78, 71].Sublist anna_scores ∧
  -- The mean of all scores is 79
  anna_scores.sum / anna_scores.length = 79 := by
sorry

end NUMINAMATH_CALUDE_anna_scores_proof_l125_12565


namespace NUMINAMATH_CALUDE_unique_fraction_sum_l125_12568

theorem unique_fraction_sum : ∃! (a₂ a₃ a₄ a₅ a₆ a₇ : ℕ),
  (5 : ℚ) / 7 = a₂ / 2 + a₃ / 6 + a₄ / 24 + a₅ / 120 + a₆ / 720 + a₇ / 5040 ∧
  a₂ < 2 ∧ a₃ < 3 ∧ a₄ < 4 ∧ a₅ < 5 ∧ a₆ < 6 ∧ a₇ < 7 →
  a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 9 := by
sorry

end NUMINAMATH_CALUDE_unique_fraction_sum_l125_12568


namespace NUMINAMATH_CALUDE_lcm_factor_proof_l125_12524

theorem lcm_factor_proof (A B : ℕ) (X : ℕ) : 
  A > 0 → B > 0 →
  Nat.gcd A B = 59 →
  Nat.lcm A B = 59 * X * 16 →
  A = 944 →
  X = 1 := by
sorry

end NUMINAMATH_CALUDE_lcm_factor_proof_l125_12524


namespace NUMINAMATH_CALUDE_least_with_eight_factors_l125_12507

/-- A function that returns the number of distinct positive factors of a natural number. -/
def num_factors (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a natural number has exactly eight distinct positive factors. -/
def has_eight_factors (n : ℕ) : Prop := num_factors n = 8

/-- The theorem stating that 54 is the least positive integer with exactly eight distinct positive factors. -/
theorem least_with_eight_factors : 
  has_eight_factors 54 ∧ ∀ m : ℕ, m < 54 → ¬(has_eight_factors m) := by sorry

end NUMINAMATH_CALUDE_least_with_eight_factors_l125_12507


namespace NUMINAMATH_CALUDE_adult_tickets_sold_l125_12553

def adult_ticket_price : ℕ := 5
def child_ticket_price : ℕ := 2
def total_tickets : ℕ := 85
def total_amount : ℕ := 275

theorem adult_tickets_sold (a c : ℕ) : 
  a + c = total_tickets → 
  a * adult_ticket_price + c * child_ticket_price = total_amount → 
  a = 35 := by
sorry

end NUMINAMATH_CALUDE_adult_tickets_sold_l125_12553


namespace NUMINAMATH_CALUDE_rounded_avg_mb_per_minute_is_one_l125_12597

/-- Represents the number of days of music in the library -/
def days_of_music : ℕ := 15

/-- Represents the total disk space occupied by the library in megabytes -/
def total_disk_space : ℕ := 20000

/-- Calculates the total number of minutes of music in the library -/
def total_minutes : ℕ := days_of_music * 24 * 60

/-- Calculates the average megabytes per minute of music -/
def avg_mb_per_minute : ℚ := total_disk_space / total_minutes

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

/-- Theorem stating that the rounded average megabytes per minute is 1 -/
theorem rounded_avg_mb_per_minute_is_one :
  round_to_nearest avg_mb_per_minute = 1 := by sorry

end NUMINAMATH_CALUDE_rounded_avg_mb_per_minute_is_one_l125_12597


namespace NUMINAMATH_CALUDE_sum_bound_for_positive_reals_l125_12593

theorem sum_bound_for_positive_reals (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (h : x^2 + y^2 + z^2 + 2*x*y*z = 1) : 
  2*(x + y + z) ≤ 3 := by sorry

end NUMINAMATH_CALUDE_sum_bound_for_positive_reals_l125_12593


namespace NUMINAMATH_CALUDE_tan_678_degrees_equals_138_l125_12512

theorem tan_678_degrees_equals_138 :
  ∃ (n : ℤ), -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (678 * π / 180) ∧ n = 138 := by
  sorry

end NUMINAMATH_CALUDE_tan_678_degrees_equals_138_l125_12512


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l125_12559

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (3 * x) % 17 = 14 % 17 ∧ ∀ (y : ℕ), y > 0 → (3 * y) % 17 = 14 % 17 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l125_12559


namespace NUMINAMATH_CALUDE_cosine_relationship_triangle_area_l125_12525

/-- Represents a triangle with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem stating the relationship between cosines in a triangle -/
theorem cosine_relationship (t : Triangle) 
  (h : t.a * Real.cos t.C = (2 * t.b - t.c) * Real.cos t.A) : 
  Real.cos t.A = 1 / 2 := by sorry

/-- Theorem for calculating the area of a specific triangle -/
theorem triangle_area (t : Triangle) 
  (h1 : t.a = 6) 
  (h2 : t.b + t.c = 8) 
  (h3 : Real.cos t.A = 1 / 2) : 
  (1 / 2) * t.a * t.b * Real.sin t.C = 7 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_cosine_relationship_triangle_area_l125_12525


namespace NUMINAMATH_CALUDE_quotient_problem_l125_12520

theorem quotient_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
    (h1 : dividend = 166)
    (h2 : divisor = 20)
    (h3 : remainder = 6)
    (h4 : dividend = divisor * quotient + remainder) :
  quotient = 8 := by
  sorry

end NUMINAMATH_CALUDE_quotient_problem_l125_12520


namespace NUMINAMATH_CALUDE_extended_segment_endpoint_l125_12530

/-- Given a segment with endpoints A and B, extended to point C such that BC = 1/2 * AB,
    prove that C has the calculated coordinates. -/
theorem extended_segment_endpoint (A B C : ℝ × ℝ) : 
  A = (1, -3) → 
  B = (11, 3) → 
  C.1 - B.1 = (B.1 - A.1) / 2 → 
  C.2 - B.2 = (B.2 - A.2) / 2 → 
  C = (16, 6) := by
  sorry

end NUMINAMATH_CALUDE_extended_segment_endpoint_l125_12530


namespace NUMINAMATH_CALUDE_equal_roots_condition_l125_12574

theorem equal_roots_condition (x m : ℝ) : 
  (x * (x - 2) - (m + 2)) / ((x - 2) * (m - 2)) = x / m → 
  (∃ (a : ℝ), ∀ (x : ℝ), x * (x - 2) - (m + 2) = (x - 2) * (m - 2) * (x / m) → x = a) →
  m = -3/2 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_condition_l125_12574


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l125_12542

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other. -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

/-- The theorem states that if points A(m-1, -3) and B(2, n) are symmetric with respect to the origin,
    then m + n = 2. -/
theorem symmetric_points_sum (m n : ℝ) :
  symmetric_wrt_origin (m - 1) (-3) 2 n → m + n = 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l125_12542


namespace NUMINAMATH_CALUDE_toucan_female_fraction_l125_12566

theorem toucan_female_fraction (total_birds : ℝ) (h1 : total_birds > 0) :
  let parrot_fraction : ℝ := 3/5
  let toucan_fraction : ℝ := 1 - parrot_fraction
  let female_parrot_fraction : ℝ := 1/3
  let male_bird_fraction : ℝ := 1/2
  let female_toucan_count : ℝ := toucan_fraction * total_birds * female_toucan_fraction
  let female_parrot_count : ℝ := parrot_fraction * total_birds * female_parrot_fraction
  let total_female_count : ℝ := female_toucan_count + female_parrot_count
  female_toucan_count + female_parrot_count = male_bird_fraction * total_birds →
  female_toucan_fraction = 3/4 :=
by sorry

end NUMINAMATH_CALUDE_toucan_female_fraction_l125_12566


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l125_12534

theorem power_fraction_simplification :
  (10 ^ 0.7) * (10 ^ 0.4) / ((10 ^ 0.2) * (10 ^ 0.6) * (10 ^ 0.3)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l125_12534


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l125_12564

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 - 6*x + 2 = 0 ↔ (x - 3)^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l125_12564


namespace NUMINAMATH_CALUDE_remaining_money_l125_12551

def octal_to_decimal (n : ℕ) : ℕ := sorry

def john_savings : ℕ := 5273
def rental_car_cost : ℕ := 1500

theorem remaining_money :
  octal_to_decimal john_savings - rental_car_cost = 1247 := by sorry

end NUMINAMATH_CALUDE_remaining_money_l125_12551


namespace NUMINAMATH_CALUDE_parallel_vectors_angle_l125_12502

theorem parallel_vectors_angle (α : ℝ) 
  (h_acute : 0 < α ∧ α < π / 2)
  (h_parallel : (3/2, Real.sin α) = (Real.cos α * k, 1/3 * k) → k ≠ 0) :
  α = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_angle_l125_12502


namespace NUMINAMATH_CALUDE_problem_statement_l125_12560

theorem problem_statement :
  ∀ (a b x y z : ℝ),
    (a^2 + b^2 + 3 ≥ a*b + Real.sqrt 3 * (a + b)) ∧
    (let c := z^2 + 2*x + Real.pi/6;
     a = x^2 + 2*y + Real.pi/2 ∧
     b = y^2 + 2*z + Real.pi/3 →
     max a (max b c) > 0) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l125_12560


namespace NUMINAMATH_CALUDE_cone_base_radius_l125_12515

theorem cone_base_radius (S : ℝ) (r : ℝ) : 
  S = 9 * Real.pi → -- Surface area is 9π cm²
  S = 3 * Real.pi * r^2 → -- Surface area formula for a cone with semicircular lateral surface
  r = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cone_base_radius_l125_12515


namespace NUMINAMATH_CALUDE_r_value_when_m_is_3_l125_12538

theorem r_value_when_m_is_3 (m : ℕ) (t : ℕ) (r : ℕ) : 
  m = 3 → 
  t = 3^m + 2 → 
  r = 4^t - 2*t → 
  r = 4^29 - 58 := by
sorry

end NUMINAMATH_CALUDE_r_value_when_m_is_3_l125_12538


namespace NUMINAMATH_CALUDE_first_concert_attendance_calculation_l125_12523

/-- The number of people attending the second concert -/
def second_concert_attendance : ℕ := 66018

/-- The difference in attendance between the second and first concerts -/
def attendance_difference : ℕ := 119

/-- The number of people attending the first concert -/
def first_concert_attendance : ℕ := second_concert_attendance - attendance_difference

theorem first_concert_attendance_calculation :
  first_concert_attendance = 65899 :=
by sorry

end NUMINAMATH_CALUDE_first_concert_attendance_calculation_l125_12523


namespace NUMINAMATH_CALUDE_segment_ratio_l125_12570

/-- Given a line segment GH with points E and F lying on it, 
    where GE is 3 times EH and GF is 7 times FH, 
    prove that EF is 1/8 of GH. -/
theorem segment_ratio (G E F H : Real) : 
  (E - G) = 3 * (H - E) →
  (F - G) = 7 * (H - F) →
  (F - E) = (1/8) * (H - G) := by
  sorry

end NUMINAMATH_CALUDE_segment_ratio_l125_12570


namespace NUMINAMATH_CALUDE_unique_four_digit_square_with_repeated_digits_l125_12550

theorem unique_four_digit_square_with_repeated_digits : 
  ∃! n : ℕ, 
    1000 ≤ n ∧ n ≤ 9999 ∧ 
    (∃ m : ℕ, n = m^2) ∧
    (∃ a b : ℕ, a < 10 ∧ b < 10 ∧ n = 1100 * a + 11 * b) ∧
    n = 7744 :=
by sorry

end NUMINAMATH_CALUDE_unique_four_digit_square_with_repeated_digits_l125_12550
