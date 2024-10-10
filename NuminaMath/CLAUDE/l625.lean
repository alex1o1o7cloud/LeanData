import Mathlib

namespace inverse_function_solution_l625_62516

/-- Given a function g(x) = 1 / (cx + d) where c and d are nonzero constants,
    prove that the solution to g^(-1)(x) = 0 is x = 1/d -/
theorem inverse_function_solution (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) :
  let g : ℝ → ℝ := fun x ↦ 1 / (c * x + d)
  (Function.invFun g) 0 = 1 / d := by
  sorry


end inverse_function_solution_l625_62516


namespace peach_distribution_theorem_l625_62525

/-- Represents the number of peaches each child received -/
structure PeachDistribution where
  anya : Nat
  katya : Nat
  liza : Nat
  dasha : Nat
  kolya : Nat
  petya : Nat
  tolya : Nat
  vasya : Nat

/-- Represents the last names of the children -/
inductive LastName
  | Ivanov
  | Grishin
  | Andreyev
  | Sergeyev

/-- Represents a child with their name and last name -/
structure Child where
  name : String
  lastName : LastName

/-- The theorem stating the correct distribution of peaches and last names -/
theorem peach_distribution_theorem (d : PeachDistribution) 
  (h1 : d.anya = 1)
  (h2 : d.katya = 2)
  (h3 : d.liza = 3)
  (h4 : d.dasha = 4)
  (h5 : d.kolya = d.liza)
  (h6 : d.petya = 2 * d.dasha)
  (h7 : d.tolya = 3 * d.anya)
  (h8 : d.vasya = 4 * d.katya)
  (h9 : d.anya + d.katya + d.liza + d.dasha + d.kolya + d.petya + d.tolya + d.vasya = 32) :
  ∃ (c1 c2 c3 c4 : Child),
    c1 = { name := "Liza", lastName := LastName.Ivanov } ∧
    c2 = { name := "Dasha", lastName := LastName.Grishin } ∧
    c3 = { name := "Anya", lastName := LastName.Andreyev } ∧
    c4 = { name := "Katya", lastName := LastName.Sergeyev } := by
  sorry

end peach_distribution_theorem_l625_62525


namespace certain_number_equation_l625_62535

theorem certain_number_equation (x : ℝ) : ((x + 2 - 6) * 3) / 4 = 3 ↔ x = 8 := by
  sorry

end certain_number_equation_l625_62535


namespace xaxaxa_divisible_by_seven_l625_62573

theorem xaxaxa_divisible_by_seven (X A : ℕ) 
  (h_digits : X < 10 ∧ A < 10) 
  (h_distinct : X ≠ A) : 
  ∃ k : ℕ, 101010 * X + 10101 * A = 7 * k := by
sorry

end xaxaxa_divisible_by_seven_l625_62573


namespace rex_cards_left_l625_62584

-- Define the number of cards each person has
def nicole_cards : ℕ := 700
def cindy_cards : ℕ := (3 * nicole_cards + (40 * 3 * nicole_cards) / 100)
def tim_cards : ℕ := (4 * cindy_cards) / 5
def rex_joe_cards : ℕ := ((60 * (nicole_cards + cindy_cards + tim_cards)) / 100)

-- Define the number of people sharing Rex and Joe's cards
def num_sharing_people : ℕ := 9

-- Theorem to prove
theorem rex_cards_left : 
  (rex_joe_cards / num_sharing_people) = 399 := by sorry

end rex_cards_left_l625_62584


namespace museum_trip_l625_62530

theorem museum_trip (people_first : ℕ) : 
  (people_first + 
   2 * people_first + 
   (2 * people_first - 6) + 
   (people_first + 9) = 75) → 
  people_first = 12 := by
  sorry

end museum_trip_l625_62530


namespace difference_of_expressions_l625_62521

theorem difference_of_expressions : 
  ((0.85 * 250)^2 / 2.3) - ((3/5 * 175) / 2.3) = 19587.5 := by
  sorry

end difference_of_expressions_l625_62521


namespace cylinder_lateral_area_l625_62594

/-- A cylinder with a rectangular front view of area 6 has a lateral area of 6π -/
theorem cylinder_lateral_area (h : ℝ) (h_pos : h > 0) : 
  let d := 6 / h  -- diameter of the base
  let lateral_area := π * d * h
  lateral_area = 6 * π := by
sorry

end cylinder_lateral_area_l625_62594


namespace least_subtraction_l625_62550

theorem least_subtraction (x : ℕ) : x = 22 ↔ 
  x ≠ 0 ∧
  (∀ y : ℕ, y < x → ¬(1398 - y) % 7 = 5 ∨ ¬(1398 - y) % 9 = 5 ∨ ¬(1398 - y) % 11 = 5) ∧
  (1398 - x) % 7 = 5 ∧
  (1398 - x) % 9 = 5 ∧
  (1398 - x) % 11 = 5 :=
by sorry

end least_subtraction_l625_62550


namespace intersection_point_in_circle_range_l625_62504

theorem intersection_point_in_circle_range (m : ℝ) : 
  let M : ℝ × ℝ := (1, 1)
  let line1 : ℝ × ℝ → Prop := λ p => p.1 + p.2 - 2 = 0
  let line2 : ℝ × ℝ → Prop := λ p => 3 * p.1 - p.2 - 2 = 0
  let circle : ℝ × ℝ → Prop := λ p => (p.1 - m)^2 + p.2^2 < 5
  (line1 M ∧ line2 M ∧ circle M) ↔ -1 < m ∧ m < 3 :=
by sorry

end intersection_point_in_circle_range_l625_62504


namespace sum_of_powers_equals_power_of_sum_l625_62542

theorem sum_of_powers_equals_power_of_sum : 5^5 + 5^5 + 5^5 + 5^5 = 5^6 := by
  sorry

end sum_of_powers_equals_power_of_sum_l625_62542


namespace cube_dimension_ratio_l625_62599

theorem cube_dimension_ratio (v1 v2 : ℝ) (h1 : v1 = 27) (h2 : v2 = 216) :
  (v2 / v1) ^ (1/3 : ℝ) = 2 := by
  sorry

end cube_dimension_ratio_l625_62599


namespace squirrel_climb_l625_62527

theorem squirrel_climb (x : ℝ) : 
  (∀ n : ℕ, n > 0 → (2 * n - 1) * x - 2 * (n - 1) = 26) → x = 5 := by
  sorry

end squirrel_climb_l625_62527


namespace marble_density_l625_62545

/-- Density of a rectangular prism made of marble -/
theorem marble_density (height : ℝ) (base_side : ℝ) (weight : ℝ) :
  height = 8 →
  base_side = 2 →
  weight = 86400 →
  weight / (base_side * base_side * height) = 2700 := by
  sorry

end marble_density_l625_62545


namespace investmentPlansCount_l625_62572

/-- The number of ways to distribute 3 distinct projects among 6 locations,
    with no more than 2 projects per location. -/
def investmentPlans : ℕ :=
  Nat.descFactorial 6 3 + (Nat.choose 3 2 * Nat.descFactorial 6 2)

/-- Theorem stating that the number of distinct investment plans is 210. -/
theorem investmentPlansCount : investmentPlans = 210 := by
  sorry

end investmentPlansCount_l625_62572


namespace hexagon_extended_side_length_l625_62569

/-- Regular hexagon with side length 3 -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 3)

/-- Point Y on the extension of side CD such that CY = 4CD -/
def extend_side (h : RegularHexagon) (CD : ℝ) (Y : ℝ) : Prop :=
  CD = h.side_length ∧ Y = 4 * CD

/-- The length of segment FY in the described configuration -/
def segment_FY_length (h : RegularHexagon) (Y : ℝ) : ℝ := sorry

/-- Theorem stating the length of FY is 5.5√3 -/
theorem hexagon_extended_side_length (h : RegularHexagon) (CD Y : ℝ) 
  (h_extend : extend_side h CD Y) : 
  segment_FY_length h Y = 5.5 * Real.sqrt 3 := by sorry

end hexagon_extended_side_length_l625_62569


namespace ratio_problem_l625_62548

theorem ratio_problem (p q : ℚ) (h : 25 / 7 + (2 * q - p) / (2 * q + p) = 4) : p / q = -1 := by
  sorry

end ratio_problem_l625_62548


namespace diophantine_equation_solution_l625_62570

theorem diophantine_equation_solution :
  ∀ a b c : ℕ+,
  a + b = c - 1 →
  a^3 + b^3 = c^2 - 1 →
  ((a = 2 ∧ b = 3 ∧ c = 6) ∨ (a = 3 ∧ b = 2 ∧ c = 6)) :=
by sorry

end diophantine_equation_solution_l625_62570


namespace equal_expressions_count_l625_62524

theorem equal_expressions_count (x : ℝ) (h : x > 0) : 
  (∃! (count : ℕ), count = 2 ∧ 
    count = (Bool.toNat (2 * x^x = x^x + x^x) + 
             Bool.toNat (x^(x+1) = x^x + x^x) + 
             Bool.toNat ((x+1)^x = x^x + x^x) + 
             Bool.toNat (x^(2*(x+1)) = x^x + x^x))) :=
by sorry

end equal_expressions_count_l625_62524


namespace max_students_in_class_l625_62543

theorem max_students_in_class (x : ℕ) : 
  x > 0 ∧ 
  2 ∣ x ∧ 
  4 ∣ x ∧ 
  7 ∣ x ∧ 
  x - (x / 2 + x / 4 + x / 7) < 6 →
  x ≤ 28 :=
by sorry

end max_students_in_class_l625_62543


namespace shaded_area_calculation_l625_62536

theorem shaded_area_calculation (large_square_area medium_square_area small_square_area : ℝ)
  (h1 : large_square_area = 49)
  (h2 : medium_square_area = 25)
  (h3 : small_square_area = 9) :
  small_square_area + (large_square_area - medium_square_area) = 33 := by
sorry

end shaded_area_calculation_l625_62536


namespace jacket_cost_calculation_l625_62554

def shorts_cost : ℚ := 13.99
def shirt_cost : ℚ := 12.14
def total_cost : ℚ := 33.56

theorem jacket_cost_calculation :
  total_cost - shorts_cost - shirt_cost = 7.43 := by sorry

end jacket_cost_calculation_l625_62554


namespace complex_vector_difference_l625_62565

theorem complex_vector_difference (z : ℂ) (h : z = 1 - I) :
  z^2 - z = -1 - I := by sorry

end complex_vector_difference_l625_62565


namespace cubic_function_extrema_l625_62537

/-- Given a cubic function f(x) = (1/3)x³ - x + m with a maximum value of 1,
    prove that its minimum value is -1/3 -/
theorem cubic_function_extrema (f : ℝ → ℝ) (m : ℝ) 
    (h1 : ∀ x, f x = (1/3) * x^3 - x + m) 
    (h2 : ∃ x₀, ∀ x, f x ≤ f x₀ ∧ f x₀ = 1) : 
    ∃ x₁, ∀ x, f x ≥ f x₁ ∧ f x₁ = -(1/3) := by
  sorry

end cubic_function_extrema_l625_62537


namespace quadratic_roots_problem_l625_62571

theorem quadratic_roots_problem (k : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 + (2*k - 1)*x₁ - k - 1 = 0) → 
  (x₂^2 + (2*k - 1)*x₂ - k - 1 = 0) → 
  (x₁ + x₂ - 4*x₁*x₂ = 2) → 
  (k = -3/2) := by
sorry

end quadratic_roots_problem_l625_62571


namespace power_sum_equality_l625_62549

theorem power_sum_equality : (-2)^2004 + 3 * (-2)^2003 = -2^2003 := by
  sorry

end power_sum_equality_l625_62549


namespace min_sum_consecutive_multiples_l625_62585

theorem min_sum_consecutive_multiples : 
  ∃ (a b c d : ℕ), 
    (b = a + 1) ∧ 
    (c = b + 1) ∧ 
    (d = c + 1) ∧
    (∃ k : ℕ, a = 11 * k) ∧
    (∃ l : ℕ, b = 7 * l) ∧
    (∃ m : ℕ, c = 5 * m) ∧
    (∃ n : ℕ, d = 3 * n) ∧
    (∀ w x y z : ℕ, 
      (x = w + 1) ∧ 
      (y = x + 1) ∧ 
      (z = y + 1) ∧
      (∃ p : ℕ, w = 11 * p) ∧
      (∃ q : ℕ, x = 7 * q) ∧
      (∃ r : ℕ, y = 5 * r) ∧
      (∃ s : ℕ, z = 3 * s) →
      (a + b + c + d ≤ w + x + y + z)) ∧
    (a + b + c + d = 1458) :=
by sorry

end min_sum_consecutive_multiples_l625_62585


namespace line_segment_endpoint_l625_62513

/-- Given a line segment from (0, 2) to (3, y) with length 10 and y > 0, prove y = 2 + √91 -/
theorem line_segment_endpoint (y : ℝ) (h1 : y > 0) : 
  (((3 - 0)^2 + (y - 2)^2 : ℝ) = 10^2) → y = 2 + Real.sqrt 91 := by
  sorry

end line_segment_endpoint_l625_62513


namespace f_increasing_min_value_sum_tangent_line_l625_62577

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos x ^ 2 + 1/2

-- Statement 1: f(x) is monotonically increasing on [5π/6, π]
theorem f_increasing : ∀ x y, 5*Real.pi/6 ≤ x ∧ x < y ∧ y ≤ Real.pi → f x < f y := by sorry

-- Statement 2: The minimum value of f(x) + f(x + π/4) is -√2
theorem min_value_sum : ∃ m : ℝ, (∀ x, m ≤ f x + f (x + Real.pi/4)) ∧ m = -Real.sqrt 2 := by sorry

-- Statement 3: The line y = √3x - 1/2 is a tangent line to y = f(x)
theorem tangent_line : ∃ x₀ : ℝ, f x₀ = Real.sqrt 3 * x₀ - 1/2 ∧ 
  (∀ x, f x ≤ Real.sqrt 3 * x - 1/2) := by sorry

end f_increasing_min_value_sum_tangent_line_l625_62577


namespace parabola_theorem_l625_62559

/-- Parabola type -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop
  h_p_pos : p > 0
  h_eq : ∀ x y, eq x y ↔ y^2 = 2*p*x

/-- Line passing through (1,0) -/
structure Line where
  k : ℝ
  eq : ℝ → ℝ → Prop
  h_eq : ∀ x y, eq x y ↔ y = k*(x-1)

/-- Point on the parabola -/
structure ParabolaPoint (para : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : para.eq x y

/-- Circle passing through three points -/
def circle_passes_through (x1 y1 x2 y2 x3 y3 : ℝ) : Prop :=
  (x3-x1)*(x2-x1) + (y3-y1)*(y2-y1) = 0

/-- Main theorem -/
theorem parabola_theorem (para : Parabola) :
  (∀ l : Line, ∀ P Q : ParabolaPoint para,
    l.eq P.x P.y ∧ l.eq Q.x Q.y →
    circle_passes_through P.x P.y Q.x Q.y 0 0) →
  para.p = 1/2 ∧
  (∀ R : ℝ × ℝ,
    (∃ P Q : ParabolaPoint para,
      R.1 = P.x + Q.x - 1/4 ∧
      R.2 = P.y + Q.y) →
    R.2^2 = R.1 - 7/4) :=
sorry

end parabola_theorem_l625_62559


namespace series_sum_l625_62519

/-- The sum of the infinite series 2 + ∑(k=1 to ∞) ((k+2)*(1/1000)^(k-1)) is equal to 3000000/998001 -/
theorem series_sum : 
  let S := 2 + ∑' k, (k + 2) * (1 / 1000) ^ (k - 1)
  S = 3000000 / 998001 := by
  sorry

end series_sum_l625_62519


namespace circles_intersect_l625_62586

/-- Two circles are intersecting if the distance between their centers is less than the sum of their radii
    and greater than the absolute difference of their radii. -/
def are_intersecting (r₁ r₂ d : ℝ) : Prop :=
  d < r₁ + r₂ ∧ d > |r₁ - r₂|

/-- Given two circles with radii 3 and 5, whose centers are 2 units apart, prove they are intersecting. -/
theorem circles_intersect : are_intersecting 3 5 2 := by
  sorry

#check circles_intersect

end circles_intersect_l625_62586


namespace identity_function_property_l625_62509

theorem identity_function_property (f : ℕ → ℕ) : 
  (∀ m n : ℕ, (f m + f n) ∣ (m + n)) → 
  (∀ m : ℕ, f m = m) := by
  sorry

end identity_function_property_l625_62509


namespace num_lizards_seen_l625_62533

/-- The number of legs Borgnine wants to see at the zoo -/
def total_legs : ℕ := 1100

/-- The number of chimps Borgnine has seen -/
def num_chimps : ℕ := 12

/-- The number of lions Borgnine has seen -/
def num_lions : ℕ := 8

/-- The number of tarantulas Borgnine will see -/
def num_tarantulas : ℕ := 125

/-- The number of legs a chimp has -/
def chimp_legs : ℕ := 4

/-- The number of legs a lion has -/
def lion_legs : ℕ := 4

/-- The number of legs a tarantula has -/
def tarantula_legs : ℕ := 8

/-- The number of legs a lizard has -/
def lizard_legs : ℕ := 4

/-- The theorem stating the number of lizards Borgnine has seen -/
theorem num_lizards_seen : 
  (total_legs - (num_chimps * chimp_legs + num_lions * lion_legs + num_tarantulas * tarantula_legs)) / lizard_legs = 5 := by
  sorry

end num_lizards_seen_l625_62533


namespace johns_beef_purchase_l625_62562

/-- Given that John uses all but 1 pound of beef in soup, uses twice as many pounds of vegetables 
    as beef, and uses 6 pounds of vegetables, prove that John bought 4 pounds of beef. -/
theorem johns_beef_purchase (beef_used : ℝ) (vegetables_used : ℝ) (beef_leftover : ℝ) : 
  beef_leftover = 1 →
  vegetables_used = 2 * beef_used →
  vegetables_used = 6 →
  beef_used + beef_leftover = 4 := by
  sorry

end johns_beef_purchase_l625_62562


namespace floor_plus_self_eq_fifteen_fourths_l625_62598

theorem floor_plus_self_eq_fifteen_fourths :
  ∃! (x : ℚ), (⌊x⌋ : ℚ) + x = 15/4 :=
by sorry

end floor_plus_self_eq_fifteen_fourths_l625_62598


namespace unit_digit_of_seven_power_ten_l625_62505

theorem unit_digit_of_seven_power_ten (n : ℕ) : n = 10 → (7^n) % 10 = 9 := by
  sorry

end unit_digit_of_seven_power_ten_l625_62505


namespace total_elephants_count_l625_62560

def elephants_we_preserve : ℕ := 70

def elephants_gestures_for_good : ℕ := 3 * elephants_we_preserve

def total_elephants : ℕ := elephants_we_preserve + elephants_gestures_for_good

theorem total_elephants_count : total_elephants = 280 := by
  sorry

end total_elephants_count_l625_62560


namespace triangle_exradius_theorem_l625_62511

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  R : ℝ
  r_a : ℝ
  r_b : ℝ
  r_c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  pos_R : 0 < R
  pos_r_a : 0 < r_a
  pos_r_b : 0 < r_b
  pos_r_c : 0 < r_c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

theorem triangle_exradius_theorem (t : Triangle) (h : 2 * t.R ≤ t.r_a) :
  t.a > t.b ∧ t.a > t.c ∧ 2 * t.R > t.r_b ∧ 2 * t.R > t.r_c := by
  sorry

end triangle_exradius_theorem_l625_62511


namespace grid_figure_boundary_theorem_l625_62555

/-- A grid figure is a shape cut from grid paper along grid lines without holes. -/
structure GridFigure where
  -- Add necessary fields here
  no_holes : Bool

/-- Represents a set of straight cuts along grid lines. -/
structure GridCuts where
  total_length : ℕ
  divides_into_cells : Bool

/-- Checks if a grid figure has a straight boundary segment of at least given length. -/
def has_straight_boundary_segment (figure : GridFigure) (length : ℕ) : Prop :=
  sorry

theorem grid_figure_boundary_theorem (figure : GridFigure) (cuts : GridCuts) :
  figure.no_holes ∧ 
  cuts.total_length = 2017 ∧
  cuts.divides_into_cells →
  has_straight_boundary_segment figure 2 :=
by sorry

end grid_figure_boundary_theorem_l625_62555


namespace sugar_distribution_l625_62539

/-- The number of sugar boxes -/
def num_boxes : ℕ := 21

/-- The weight of sugar per box in kilograms -/
def sugar_per_box : ℚ := 6

/-- The amount of sugar distributed to each neighbor in kilograms -/
def sugar_per_neighbor : ℚ := 32 / 41

/-- The maximum number of neighbors who can receive sugar -/
def max_neighbors : ℕ := 161

theorem sugar_distribution :
  ⌊(num_boxes * sugar_per_box) / sugar_per_neighbor⌋ = max_neighbors := by
  sorry

end sugar_distribution_l625_62539


namespace complex_equation_solution_l625_62540

theorem complex_equation_solution (z : ℂ) :
  (2 * z - Complex.I) * (2 - Complex.I) = 5 → z = 1 + Complex.I := by
  sorry

end complex_equation_solution_l625_62540


namespace rook_tour_existence_l625_62558

/-- A rook move on an m × n board. -/
inductive RookMove
  | up : RookMove
  | right : RookMove
  | down : RookMove
  | left : RookMove

/-- A valid sequence of rook moves on an m × n board. -/
def ValidMoveSequence (m n : ℕ) : List RookMove → Prop :=
  sorry

/-- A sequence of moves visits all squares exactly once and returns to start. -/
def VisitsAllSquaresOnce (m n : ℕ) (moves : List RookMove) : Prop :=
  sorry

theorem rook_tour_existence (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) :
  (∃ moves : List RookMove, ValidMoveSequence m n moves ∧ VisitsAllSquaresOnce m n moves) ↔
  (Even m ∧ Even n) :=
sorry

end rook_tour_existence_l625_62558


namespace one_seventh_minus_one_eleventh_equals_100_l625_62551

theorem one_seventh_minus_one_eleventh_equals_100 :
  let N : ℚ := 1925
  (N / 7) - (N / 11) = 100 := by
  sorry

end one_seventh_minus_one_eleventh_equals_100_l625_62551


namespace problem_solution_l625_62522

theorem problem_solution (x y : ℕ) (h1 : x > y) (h2 : x + x * y = 391) : x + y = 39 := by
  sorry

end problem_solution_l625_62522


namespace village_population_l625_62596

/-- The population change over two years -/
def population_change (initial : ℝ) : ℝ := initial * 1.3 * 0.7

/-- The problem statement -/
theorem village_population : 
  ∃ (initial : ℝ), 
    population_change initial = 13650 ∧ 
    initial = 15000 := by
  sorry

end village_population_l625_62596


namespace inequality_proof_l625_62506

theorem inequality_proof (a b : ℝ) (ha : a ≠ 0) :
  a^2 + b^2 + 1/a^2 + b/a ≥ Real.sqrt 3 := by
  sorry

end inequality_proof_l625_62506


namespace fraction_comparison_l625_62563

theorem fraction_comparison : (2 : ℚ) / 3 - 66666666 / 100000000 = 2 / (3 * 100000000) := by sorry

end fraction_comparison_l625_62563


namespace stream_speed_l625_62531

/-- Given a boat with a speed in still water and its travel time and distance downstream,
    calculate the speed of the stream. -/
theorem stream_speed (boat_speed : ℝ) (time : ℝ) (distance : ℝ) : 
  boat_speed = 24 →
  time = 4 →
  distance = 112 →
  distance = (boat_speed + (distance / time - boat_speed)) * time →
  distance / time - boat_speed = 4 := by
  sorry

end stream_speed_l625_62531


namespace quadratic_equation_roots_l625_62526

theorem quadratic_equation_roots (x : ℝ) : 
  (∃! r : ℝ, x^2 - 2*x + 1 = 0) ↔ (x^2 - 2*x + 1 = 0) := by
  sorry

end quadratic_equation_roots_l625_62526


namespace stratified_sampling_class_c_l625_62500

theorem stratified_sampling_class_c (total_students : ℕ) (class_a class_b class_c class_d sample_size : ℕ) : 
  total_students = class_a + class_b + class_c + class_d →
  class_a = 75 →
  class_b = 75 →
  class_c = 200 →
  class_d = 150 →
  sample_size = 20 →
  (class_c * sample_size) / total_students = 8 :=
by sorry

end stratified_sampling_class_c_l625_62500


namespace tan_pi_plus_theta_l625_62529

theorem tan_pi_plus_theta (θ : Real) :
  (∃ (x y : Real), x^2 + y^2 = 1 ∧ x = 3/5 ∧ y = 4/5 ∧ 
   x = Real.cos θ ∧ y = Real.sin θ) →
  Real.tan (π + θ) = 4/3 := by
sorry

end tan_pi_plus_theta_l625_62529


namespace function_existence_iff_divisibility_l625_62581

theorem function_existence_iff_divisibility (k a : ℕ) :
  (∃ f : ℕ → ℕ, ∀ n : ℕ, (f^[k] n = n + a)) ↔ (a ≥ 0 ∧ k ∣ a) :=
sorry

end function_existence_iff_divisibility_l625_62581


namespace no_positive_roots_l625_62583

theorem no_positive_roots :
  ∀ x : ℝ, x > 0 → x^3 + 6*x^2 + 11*x + 6 ≠ 0 := by
  sorry

end no_positive_roots_l625_62583


namespace simplify_expression_l625_62515

theorem simplify_expression (x : ℝ) : 2*x - 3*(2-x) + 4*(3+x) - 5*(2+3*x) = -6*x - 4 := by
  sorry

end simplify_expression_l625_62515


namespace integer_pair_divisibility_l625_62580

theorem integer_pair_divisibility (x y : ℕ+) : 
  (((x : ℤ) * y - 6)^2 ∣ (x : ℤ)^2 + y^2) ↔ 
  ((x = 7 ∧ y = 1) ∨ (x = 4 ∧ y = 2) ∨ (x = 3 ∧ y = 3)) :=
by sorry

end integer_pair_divisibility_l625_62580


namespace fifteenth_term_ratio_l625_62557

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℚ  -- first term
  d : ℚ  -- common difference

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.a + (n - 1) * seq.d) / 2

theorem fifteenth_term_ratio
  (seq1 seq2 : ArithmeticSequence)
  (h : ∀ n : ℕ, (sum_n seq1 n) / (sum_n seq2 n) = (9 * n + 3) / (5 * n + 35)) :
  (seq1.a + 14 * seq1.d) / (seq2.a + 14 * seq2.d) = 7 / 5 := by
  sorry

end fifteenth_term_ratio_l625_62557


namespace quadratic_equation_sum_product_l625_62564

theorem quadratic_equation_sum_product (m p : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - m * x + p = 0 ∧ 3 * y^2 - m * y + p = 0 ∧ x + y = 9 ∧ x * y = 14) →
  m + p = 69 := by
  sorry

end quadratic_equation_sum_product_l625_62564


namespace monday_greatest_range_l625_62595

/-- Temperature range for a day -/
def temp_range (high low : Int) : Int := high - low

/-- Temperature data for each day -/
def monday_high : Int := 6
def monday_low : Int := -4
def tuesday_high : Int := 3
def tuesday_low : Int := -6
def wednesday_high : Int := 4
def wednesday_low : Int := -2
def thursday_high : Int := 4
def thursday_low : Int := -5
def friday_high : Int := 8
def friday_low : Int := 0

/-- Theorem: Monday has the greatest temperature range -/
theorem monday_greatest_range :
  let monday_range := temp_range monday_high monday_low
  let tuesday_range := temp_range tuesday_high tuesday_low
  let wednesday_range := temp_range wednesday_high wednesday_low
  let thursday_range := temp_range thursday_high thursday_low
  let friday_range := temp_range friday_high friday_low
  (monday_range > tuesday_range) ∧
  (monday_range > wednesday_range) ∧
  (monday_range > thursday_range) ∧
  (monday_range > friday_range) :=
by sorry

end monday_greatest_range_l625_62595


namespace union_when_t_is_two_B_subset_A_iff_l625_62592

-- Define sets A and B
def A (t : ℝ) : Set ℝ := {x | x^2 + (1-t)*x - t ≤ 0}
def B : Set ℝ := {x | |x-2| < 1}

-- Statement 1
theorem union_when_t_is_two :
  A 2 ∪ B = {x | -1 ≤ x ∧ x < 3} := by sorry

-- Statement 2
theorem B_subset_A_iff (t : ℝ) :
  B ⊆ A t ↔ t ≥ 3 := by sorry

end union_when_t_is_two_B_subset_A_iff_l625_62592


namespace flea_treatment_effectiveness_l625_62574

theorem flea_treatment_effectiveness (F : ℕ) : 
  (F : ℝ) * 0.4 * 0.55 * 0.7 * 0.8 = 20 → F - 20 = 142 := by
  sorry

end flea_treatment_effectiveness_l625_62574


namespace sqrt_sum_squares_equality_l625_62514

theorem sqrt_sum_squares_equality (a b c : ℝ) :
  Real.sqrt (a^2 + b^2 + c^2) = a + b - c ↔ a * b = c * (a + b) ∧ a + b - c ≥ 0 := by
  sorry

end sqrt_sum_squares_equality_l625_62514


namespace mk_97_check_one_l625_62501

theorem mk_97_check_one (x : ℝ) : x = 1 ↔ x ≠ 2 * x ∧ ∃! y : ℝ, y ^ 2 + 2 * x * y + x = 0 := by sorry

end mk_97_check_one_l625_62501


namespace office_meeting_reduction_l625_62568

theorem office_meeting_reduction (total_people : ℕ) (women_in_meeting : ℕ) : 
  total_people = 60 → 
  women_in_meeting = 6 → 
  (women_in_meeting : ℚ) / (total_people / 2 : ℚ) * 100 = 20 := by
  sorry

end office_meeting_reduction_l625_62568


namespace lucas_cycling_speed_l625_62567

theorem lucas_cycling_speed 
  (philippe_speed : ℝ) 
  (marta_ratio : ℝ) 
  (lucas_ratio : ℝ) 
  (h1 : philippe_speed = 10)
  (h2 : marta_ratio = 3/4)
  (h3 : lucas_ratio = 4/3) : 
  lucas_ratio * (marta_ratio * philippe_speed) = 10 :=
by
  sorry

end lucas_cycling_speed_l625_62567


namespace garden_area_l625_62589

/-- The area of a garden with square cutouts -/
theorem garden_area (garden_length : ℝ) (garden_width : ℝ) 
  (cutout1_side : ℝ) (cutout2_side : ℝ) : 
  garden_length = 20 ∧ garden_width = 18 ∧ 
  cutout1_side = 4 ∧ cutout2_side = 5 →
  garden_length * garden_width - cutout1_side^2 - cutout2_side^2 = 319 := by
  sorry

end garden_area_l625_62589


namespace not_always_same_digit_sum_l625_62566

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem not_always_same_digit_sum :
  ∃ (N M : ℕ), 
    (sum_of_digits (N + M) = sum_of_digits N) ∧
    (∃ (k : ℕ), k > 1 ∧ sum_of_digits (N + k * M) ≠ sum_of_digits N) :=
sorry

end not_always_same_digit_sum_l625_62566


namespace isabel_ds_games_left_l625_62523

/-- Given that Isabel initially had 90 DS games and gave 87 to her friend,
    prove that she has 3 DS games left. -/
theorem isabel_ds_games_left (initial_games : ℕ) (games_given : ℕ) (games_left : ℕ) : 
  initial_games = 90 → games_given = 87 → games_left = initial_games - games_given → games_left = 3 := by
  sorry

end isabel_ds_games_left_l625_62523


namespace old_belt_time_correct_l625_62552

/-- The time it takes for the old conveyor belt to move one day's coal output -/
def old_belt_time : ℝ := 21

/-- The time it takes for the new conveyor belt to move one day's coal output -/
def new_belt_time : ℝ := 15

/-- The time it takes for both belts together to move one day's coal output -/
def combined_time : ℝ := 8.75

/-- Theorem stating that the old conveyor belt time is correct given the conditions -/
theorem old_belt_time_correct :
  1 / old_belt_time + 1 / new_belt_time = 1 / combined_time :=
by sorry

end old_belt_time_correct_l625_62552


namespace fourth_roll_six_prob_l625_62502

/-- Represents a six-sided die --/
structure Die where
  prob_six : ℚ
  prob_other : ℚ
  sum_probs : prob_six + 5 * prob_other = 1

/-- The fair die --/
def fair_die : Die where
  prob_six := 1/6
  prob_other := 1/6
  sum_probs := by norm_num

/-- The biased die --/
def biased_die : Die where
  prob_six := 3/4
  prob_other := 1/20
  sum_probs := by norm_num

/-- The probability of choosing each die --/
def prob_choose_die : ℚ := 1/2

/-- The number of initial rolls that are sixes --/
def num_initial_sixes : ℕ := 3

/-- Theorem: Given the conditions, the probability of rolling a six on the fourth roll is 2187/982 --/
theorem fourth_roll_six_prob :
  let prob_fair := prob_choose_die * fair_die.prob_six^num_initial_sixes
  let prob_biased := prob_choose_die * biased_die.prob_six^num_initial_sixes
  let total_prob := prob_fair + prob_biased
  let cond_prob_fair := prob_fair / total_prob
  let cond_prob_biased := prob_biased / total_prob
  cond_prob_fair * fair_die.prob_six + cond_prob_biased * biased_die.prob_six = 2187 / 982 := by
  sorry

end fourth_roll_six_prob_l625_62502


namespace perfect_square_units_mod_16_l625_62508

theorem perfect_square_units_mod_16 : 
  ∃ (S : Finset ℕ), (∀ n : ℕ, ∃ m : ℕ, n ^ 2 % 16 ∈ S) ∧ S.card = 4 := by
  sorry

end perfect_square_units_mod_16_l625_62508


namespace bianca_birthday_money_l625_62546

/-- The amount of money Bianca received for her birthday -/
def birthday_money (num_friends : ℕ) (dollars_per_friend : ℕ) : ℕ :=
  num_friends * dollars_per_friend

/-- Theorem stating that Bianca received 30 dollars for her birthday -/
theorem bianca_birthday_money :
  birthday_money 5 6 = 30 := by
  sorry

end bianca_birthday_money_l625_62546


namespace absolute_value_equation_sum_l625_62532

theorem absolute_value_equation_sum (n : ℝ) : 
  (∃ n₁ n₂ : ℝ, |3 * n₁ - 8| = 5 ∧ |3 * n₂ - 8| = 5 ∧ n₁ ≠ n₂ ∧ n₁ + n₂ = 16/3) :=
by sorry

end absolute_value_equation_sum_l625_62532


namespace dwarf_system_stabilizes_l625_62582

-- Define the color of a dwarf's house
inductive Color
| Red
| White

-- Define the state of the dwarf system
structure DwarfSystem :=
  (houses : Fin 12 → Color)
  (friends : Fin 12 → Set (Fin 12))

-- Define a single step in the system
def step (sys : DwarfSystem) (i : Fin 12) : DwarfSystem := sorry

-- Define the relation between two states
def reaches (initial final : DwarfSystem) : Prop := sorry

-- Theorem statement
theorem dwarf_system_stabilizes (initial : DwarfSystem) :
  ∃ (final : DwarfSystem), reaches initial final ∧ ∀ i, step final i = final :=
sorry

end dwarf_system_stabilizes_l625_62582


namespace help_sign_white_area_l625_62593

theorem help_sign_white_area :
  let sign_width : ℕ := 18
  let sign_height : ℕ := 7
  let h_area : ℕ := 13
  let e_area : ℕ := 11
  let l_area : ℕ := 8
  let p_area : ℕ := 11
  let total_black_area : ℕ := h_area + e_area + l_area + p_area
  let total_sign_area : ℕ := sign_width * sign_height
  total_sign_area - total_black_area = 83 := by
  sorry

end help_sign_white_area_l625_62593


namespace parabola_c_value_l625_62591

/-- A parabola with equation x = ay^2 + by + c, vertex at (4, 1), and passing through (1, 3) -/
def Parabola (a b c : ℝ) : Prop :=
  ∀ y : ℝ, 4 = a * 1^2 + b * 1 + c ∧
            1 = a * 3^2 + b * 3 + c

theorem parabola_c_value :
  ∀ a b c : ℝ, Parabola a b c → c = 13/4 := by
  sorry

end parabola_c_value_l625_62591


namespace sine_sum_simplification_l625_62576

theorem sine_sum_simplification (x y : ℝ) :
  Real.sin (x - y) * Real.cos y + Real.cos (x - y) * Real.sin y = Real.sin x := by
  sorry

end sine_sum_simplification_l625_62576


namespace fish_pond_problem_l625_62547

/-- Calculates the number of fish in the second catch given the total number of fish in the pond,
    the number of tagged fish, and the number of tagged fish caught in the second catch. -/
def second_catch_size (total_fish : ℕ) (tagged_fish : ℕ) (tagged_caught : ℕ) : ℕ :=
  tagged_fish * total_fish / tagged_caught

/-- Theorem stating that given a pond with approximately 1000 fish, where 40 fish were initially tagged
    and released, and 2 tagged fish were found in a subsequent catch, the number of fish in the
    subsequent catch is 50. -/
theorem fish_pond_problem (total_fish : ℕ) (tagged_fish : ℕ) (tagged_caught : ℕ) :
  total_fish = 1000 → tagged_fish = 40 → tagged_caught = 2 →
  second_catch_size total_fish tagged_fish tagged_caught = 50 := by
  sorry

#eval second_catch_size 1000 40 2

end fish_pond_problem_l625_62547


namespace ball_count_l625_62590

theorem ball_count (red green blue total : ℕ) 
  (ratio : red = 15 ∧ green = 13 ∧ blue = 17)
  (red_count : red = 907) :
  total = 2721 :=
by sorry

end ball_count_l625_62590


namespace hospital_employee_arrangements_l625_62503

theorem hospital_employee_arrangements (n : ℕ) (h : n = 6) :
  (Nat.factorial n = 720) ∧
  (Nat.factorial (n - 1) = 120) ∧
  (n * (n - 1) * (n - 2) = 120) := by
  sorry

#check hospital_employee_arrangements

end hospital_employee_arrangements_l625_62503


namespace fraction_of_juniors_l625_62579

theorem fraction_of_juniors (J S : ℕ) : 
  J > 0 → -- There is at least one junior
  S > 0 → -- There is at least one senior
  (J : ℚ) / 2 = (S : ℚ) * 2 / 3 → -- Half the number of juniors equals two-thirds the number of seniors
  (J : ℚ) / (J + S) = 4 / 7 := by
sorry

end fraction_of_juniors_l625_62579


namespace river_width_l625_62587

/-- Given a river with specified depth, flow rate, and volume per minute, prove its width. -/
theorem river_width (depth : ℝ) (flow_rate : ℝ) (volume_per_minute : ℝ) :
  depth = 2 →
  flow_rate = 3 →
  volume_per_minute = 4500 →
  (flow_rate * 1000 / 60) * depth * (volume_per_minute / (flow_rate * 1000 / 60) / depth) = 45 := by
  sorry

end river_width_l625_62587


namespace largest_factor_is_large_barrel_capacity_l625_62512

def total_oil : ℕ := 95
def small_barrel_capacity : ℕ := 5
def small_barrels_used : ℕ := 1

def remaining_oil : ℕ := total_oil - (small_barrel_capacity * small_barrels_used)

def is_valid_large_barrel_capacity (capacity : ℕ) : Prop :=
  capacity > small_barrel_capacity ∧ 
  remaining_oil % capacity = 0 ∧
  capacity ≤ remaining_oil

theorem largest_factor_is_large_barrel_capacity : 
  ∃ (large_barrel_capacity : ℕ), 
    is_valid_large_barrel_capacity large_barrel_capacity ∧
    ∀ (x : ℕ), is_valid_large_barrel_capacity x → x ≤ large_barrel_capacity := by
  sorry

end largest_factor_is_large_barrel_capacity_l625_62512


namespace geometric_mean_of_one_and_nine_l625_62520

theorem geometric_mean_of_one_and_nine :
  ∃ (c : ℝ), c^2 = 1 * 9 ∧ (c = 3 ∨ c = -3) := by
  sorry

end geometric_mean_of_one_and_nine_l625_62520


namespace smallest_undefined_inverse_l625_62507

theorem smallest_undefined_inverse (a : ℕ) : 
  (∀ k < 3, k > 0 → (Nat.gcd k 63 = 1 ∨ Nat.gcd k 66 = 1)) ∧ 
  Nat.gcd 3 63 > 1 ∧ 
  Nat.gcd 3 66 > 1 := by
  sorry

end smallest_undefined_inverse_l625_62507


namespace cubic_roots_relation_l625_62556

theorem cubic_roots_relation (p q r : ℝ) (u v w : ℝ) : 
  (p^3 + 4*p^2 + 5*p - 13 = 0) →
  (q^3 + 4*q^2 + 5*q - 13 = 0) →
  (r^3 + 4*r^2 + 5*r - 13 = 0) →
  ((p+q)^3 + u*(p+q)^2 + v*(p+q) + w = 0) →
  ((q+r)^3 + u*(q+r)^2 + v*(q+r) + w = 0) →
  ((r+p)^3 + u*(r+p)^2 + v*(r+p) + w = 0) →
  w = 33 := by
sorry

end cubic_roots_relation_l625_62556


namespace softball_players_count_l625_62588

theorem softball_players_count (cricket hockey football total : ℕ) 
  (h1 : cricket = 15)
  (h2 : hockey = 12)
  (h3 : football = 13)
  (h4 : total = 55) :
  total - (cricket + hockey + football) = 15 := by
  sorry

end softball_players_count_l625_62588


namespace range_of_a_l625_62597

open Set

/-- The equation that must have 3 distinct real solutions -/
def equation (a x : ℝ) : ℝ := 2 * x * |x| - (a - 2) * x + |x| - a + 1

/-- The condition that the equation has 3 distinct real solutions -/
def has_three_distinct_solutions (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    equation a x₁ = 0 ∧ equation a x₂ = 0 ∧ equation a x₃ = 0

/-- The theorem stating the range of a -/
theorem range_of_a (a : ℝ) :
  has_three_distinct_solutions a → a ∈ Ioi 9 :=
sorry

end range_of_a_l625_62597


namespace roots_of_equation_l625_62528

theorem roots_of_equation : 
  let f (x : ℝ) := 21 / (x^2 - 9) - 3 / (x - 3) - 1
  ∀ x : ℝ, f x = 0 ↔ x = 3 ∨ x = -7 := by sorry

end roots_of_equation_l625_62528


namespace sphere_volume_l625_62578

theorem sphere_volume (R : ℝ) (h : R = 3) : (4 / 3 : ℝ) * Real.pi * R^3 = 36 * Real.pi := by
  sorry

end sphere_volume_l625_62578


namespace ten_lines_intersections_l625_62534

/-- The number of intersections formed by n straight lines where no two lines are parallel
    and no three lines intersect at a single point. -/
def intersections (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else (n - 1) * (n - 2) / 2

/-- Theorem stating that 10 straight lines under the given conditions form 45 intersections -/
theorem ten_lines_intersections :
  intersections 10 = 45 := by
  sorry

end ten_lines_intersections_l625_62534


namespace unique_number_problem_l625_62561

theorem unique_number_problem : ∃! (x : ℝ), x > 0 ∧ (((x^2 / 3)^3) / 9) = x :=
by sorry

end unique_number_problem_l625_62561


namespace triangle_angle_measure_l625_62518

theorem triangle_angle_measure (b c S_ABC : ℝ) (h1 : b = 8) (h2 : c = 8 * Real.sqrt 3) 
  (h3 : S_ABC = 16 * Real.sqrt 3) :
  ∃ A : ℝ, (A = π / 6 ∨ A = 5 * π / 6) ∧ 
    S_ABC = (1/2) * b * c * Real.sin A ∧ 0 < A ∧ A < π :=
by sorry

end triangle_angle_measure_l625_62518


namespace set_empty_properties_l625_62575

theorem set_empty_properties (A : Set α) :
  let p := A ∩ ∅ = ∅
  let q := A ∪ ∅ = A
  (p ∧ q) ∧ (¬p ∨ q) := by sorry

end set_empty_properties_l625_62575


namespace rationalize_denominator_l625_62541

theorem rationalize_denominator : 
  (Real.sqrt 18 - Real.sqrt 2 + Real.sqrt 27) / (Real.sqrt 3 + Real.sqrt 2) = 5 - Real.sqrt 6 := by
  sorry

end rationalize_denominator_l625_62541


namespace no_x_satisfying_conditions_l625_62544

theorem no_x_satisfying_conditions : ¬∃ x : ℝ, 
  250 ≤ x ∧ x ≤ 350 ∧ 
  ⌊Real.sqrt (x - 50)⌋ = 14 ∧ 
  ⌊Real.sqrt (50 * x)⌋ = 256 := by
  sorry

#check no_x_satisfying_conditions

end no_x_satisfying_conditions_l625_62544


namespace operation_results_in_zero_in_quotient_l625_62538

-- Define the arithmetic operation
def operation : ℕ → ℕ → ℕ := (·+·)

-- Define the property of having a zero in the middle of the quotient
def has_zero_in_middle_of_quotient (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a * 10 + 0 * 1 + b ∧ 0 < b ∧ b < 10

-- Theorem statement
theorem operation_results_in_zero_in_quotient :
  has_zero_in_middle_of_quotient (operation 6 4 / 3) :=
sorry

end operation_results_in_zero_in_quotient_l625_62538


namespace snackies_leftover_l625_62553

theorem snackies_leftover (m : ℕ) (h : m % 8 = 5) : (4 * m) % 8 = 4 := by
  sorry

end snackies_leftover_l625_62553


namespace hunter_saw_twelve_ants_l625_62517

/-- The number of ants Hunter saw in the playground -/
def ants_seen (spiders ladybugs_initial ladybugs_left total_insects : ℕ) : ℕ :=
  total_insects - spiders - ladybugs_left

/-- Theorem stating that Hunter saw 12 ants given the problem conditions -/
theorem hunter_saw_twelve_ants :
  let spiders : ℕ := 3
  let ladybugs_initial : ℕ := 8
  let ladybugs_left : ℕ := ladybugs_initial - 2
  let total_insects : ℕ := 21
  ants_seen spiders ladybugs_initial ladybugs_left total_insects = 12 := by
  sorry

end hunter_saw_twelve_ants_l625_62517


namespace obtuse_triangle_count_l625_62510

/-- A function that checks if a triangle with sides a, b, and c is obtuse -/
def is_obtuse (a b c : ℝ) : Prop :=
  (a^2 > b^2 + c^2) ∨ (b^2 > a^2 + c^2) ∨ (c^2 > a^2 + b^2)

/-- A function that checks if a triangle with sides a, b, and c is valid -/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a

/-- The main theorem stating that there are exactly 13 positive integer values of k
    for which a triangle with sides 12, 16, and k is obtuse -/
theorem obtuse_triangle_count :
  (∃! (s : Finset ℕ), s.card = 13 ∧ 
    (∀ k, k ∈ s ↔ (is_valid_triangle 12 16 k ∧ is_obtuse 12 16 k))) := by
  sorry

end obtuse_triangle_count_l625_62510
