import Mathlib

namespace cakes_left_with_brenda_l1491_149146

def cakes_per_day : ℕ := 20
def days_baking : ℕ := 9
def fraction_sold : ℚ := 1/2

theorem cakes_left_with_brenda : 
  (cakes_per_day * days_baking) * (1 - fraction_sold) = 90 := by
  sorry

end cakes_left_with_brenda_l1491_149146


namespace seating_arrangements_eq_48_num_seating_arrangements_l1491_149152

/- Define the number of teams -/
def num_teams : ℕ := 3

/- Define the number of athletes per team -/
def athletes_per_team : ℕ := 2

/- Define the total number of athletes -/
def total_athletes : ℕ := num_teams * athletes_per_team

/- Function to calculate the number of seating arrangements -/
def seating_arrangements : ℕ :=
  (Nat.factorial num_teams) * (Nat.factorial athletes_per_team)^num_teams

/- Theorem stating that the number of seating arrangements is 48 -/
theorem seating_arrangements_eq_48 :
  seating_arrangements = 48 := by
  sorry

/- Main theorem to prove -/
theorem num_seating_arrangements :
  ∀ (n m : ℕ), n = num_teams → m = athletes_per_team →
  (Nat.factorial n) * (Nat.factorial m)^n = 48 := by
  sorry

end seating_arrangements_eq_48_num_seating_arrangements_l1491_149152


namespace parabola_equilateral_distance_l1491_149190

/-- Parabola structure -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  directrix : ℝ → ℝ → Prop

/-- Point on a parabola -/
def PointOnParabola (p : Parabola) (point : ℝ × ℝ) : Prop :=
  p.equation point.1 point.2

/-- Perpendicular to directrix -/
def PerpendicularToDirectrix (p : Parabola) (point : ℝ × ℝ) (foot : ℝ × ℝ) : Prop :=
  p.directrix foot.1 foot.2 ∧ 
  (point.1 - foot.1) * (p.focus.1 - foot.1) + (point.2 - foot.2) * (p.focus.2 - foot.2) = 0

/-- Equilateral triangle -/
def IsEquilateralTriangle (a b c : ℝ × ℝ) : Prop :=
  let dist := fun (p q : ℝ × ℝ) => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist a b = dist b c ∧ dist b c = dist c a

/-- Main theorem -/
theorem parabola_equilateral_distance (p : Parabola) (point : ℝ × ℝ) (foot : ℝ × ℝ) :
  p.equation = fun x y => y^2 = 6*x →
  PointOnParabola p point →
  PerpendicularToDirectrix p point foot →
  IsEquilateralTriangle point foot p.focus →
  Real.sqrt ((point.1 - p.focus.1)^2 + (point.2 - p.focus.2)^2) = 6 := by
  sorry

end parabola_equilateral_distance_l1491_149190


namespace unique_zero_quadratic_l1491_149147

/-- Given a quadratic function f(x) = 3x^2 + 2x - a with a unique zero in (-1, 1),
    prove that a ∈ (1, 5) ∪ {-1/3} -/
theorem unique_zero_quadratic (a : ℝ) :
  (∃! x : ℝ, x ∈ (Set.Ioo (-1) 1) ∧ 3 * x^2 + 2 * x - a = 0) →
  (a ∈ Set.Ioo 1 5 ∨ a = -1/3) :=
sorry

end unique_zero_quadratic_l1491_149147


namespace truck_filling_time_l1491_149165

/-- Calculates the total time to fill a truck with stone blocks -/
theorem truck_filling_time 
  (truck_capacity : ℕ)
  (rate_per_person : ℕ)
  (initial_workers : ℕ)
  (initial_duration : ℕ)
  (additional_workers : ℕ)
  (h1 : truck_capacity = 6000)
  (h2 : rate_per_person = 250)
  (h3 : initial_workers = 2)
  (h4 : initial_duration = 4)
  (h5 : additional_workers = 6) :
  ∃ (total_time : ℕ), total_time = 6 ∧ 
  (initial_workers * rate_per_person * initial_duration + 
   (initial_workers + additional_workers) * rate_per_person * (total_time - initial_duration) = truck_capacity) :=
by
  sorry


end truck_filling_time_l1491_149165


namespace line_circle_intersection_l1491_149184

/-- Given a point (a,b) outside the circle x^2 + y^2 = r^2, 
    the line ax + by = r^2 intersects the circle and does not pass through the center. -/
theorem line_circle_intersection (a b r : ℝ) (hr : r > 0) (h_outside : a^2 + b^2 > r^2) :
  ∃ (x y : ℝ), x^2 + y^2 = r^2 ∧ a*x + b*y = r^2 ∧ (x ≠ 0 ∨ y ≠ 0) := by
  sorry


end line_circle_intersection_l1491_149184


namespace sum_of_dimensions_for_specific_box_l1491_149123

/-- A rectangular box with dimensions A, B, and C -/
structure RectangularBox where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The sum of dimensions of a rectangular box -/
def sum_of_dimensions (box : RectangularBox) : ℝ :=
  box.A + box.B + box.C

/-- Theorem: For a rectangular box with given surface areas, the sum of its dimensions is 27.67 -/
theorem sum_of_dimensions_for_specific_box :
  ∃ (box : RectangularBox),
    box.A * box.B = 40 ∧
    box.A * box.C = 90 ∧
    box.B * box.C = 100 ∧
    sum_of_dimensions box = 27.67 := by
  sorry

end sum_of_dimensions_for_specific_box_l1491_149123


namespace caterpillar_to_scorpion_ratio_l1491_149128

/-- Represents Calvin's bug collection -/
structure BugCollection where
  roaches : ℕ
  scorpions : ℕ
  crickets : ℕ
  caterpillars : ℕ

/-- Calvin's bug collection satisfies the given conditions -/
def calvins_collection : BugCollection where
  roaches := 12
  scorpions := 3
  crickets := 6  -- half as many crickets as roaches
  caterpillars := 6  -- to be proven

theorem caterpillar_to_scorpion_ratio (c : BugCollection) 
  (h1 : c.roaches = 12)
  (h2 : c.scorpions = 3)
  (h3 : c.crickets = c.roaches / 2)
  (h4 : c.roaches + c.scorpions + c.crickets + c.caterpillars = 27) :
  c.caterpillars / c.scorpions = 2 := by
  sorry

#check caterpillar_to_scorpion_ratio calvins_collection

end caterpillar_to_scorpion_ratio_l1491_149128


namespace children_retaking_test_l1491_149186

theorem children_retaking_test (total : ℝ) (passed : ℝ) (retaking : ℝ) : 
  total = 698.0 → passed = 105.0 → retaking = total - passed → retaking = 593.0 := by
sorry

end children_retaking_test_l1491_149186


namespace bee_count_l1491_149177

theorem bee_count (initial_bees : ℕ) (incoming_bees : ℕ) : 
  initial_bees = 16 → incoming_bees = 8 → initial_bees + incoming_bees = 24 := by
  sorry

end bee_count_l1491_149177


namespace similar_triangles_area_l1491_149133

-- Define the triangles and their properties
def Triangle : Type := Unit

def similar (t1 t2 : Triangle) : Prop := sorry

def similarityRatio (t1 t2 : Triangle) : ℚ := sorry

def area (t : Triangle) : ℝ := sorry

-- State the theorem
theorem similar_triangles_area 
  (ABC DEF : Triangle) 
  (h_similar : similar ABC DEF) 
  (h_ratio : similarityRatio ABC DEF = 1 / 2) 
  (h_area_ABC : area ABC = 3) : 
  area DEF = 12 := by sorry

end similar_triangles_area_l1491_149133


namespace ladybug_dots_count_l1491_149174

/-- The number of ladybugs Andre caught on Monday -/
def monday_ladybugs : ℕ := 8

/-- The number of ladybugs Andre caught on Tuesday -/
def tuesday_ladybugs : ℕ := 5

/-- The number of dots each ladybug has -/
def dots_per_ladybug : ℕ := 6

/-- The total number of dots on all ladybugs caught by Andre -/
def total_dots : ℕ := (monday_ladybugs + tuesday_ladybugs) * dots_per_ladybug

theorem ladybug_dots_count : total_dots = 78 := by
  sorry

end ladybug_dots_count_l1491_149174


namespace system_solutions_l1491_149130

theorem system_solutions :
  ∃! (S : Set (ℝ × ℝ × ℝ)), 
    S = {(1, 1, 1), (-2, -2, -2)} ∧
    ∀ (x y z : ℝ), (x, y, z) ∈ S ↔ 
      (x + y * z = 2 ∧ y + z * x = 2 ∧ z + x * y = 2) :=
by sorry

end system_solutions_l1491_149130


namespace product_of_place_values_l1491_149105

def numeral : ℚ := 8712480.83

theorem product_of_place_values :
  let millions := 8000000
  let thousands := 8000
  let tenths := 0.8
  millions * thousands * tenths = 51200000000 :=
by sorry

end product_of_place_values_l1491_149105


namespace shoes_discount_percentage_l1491_149120

/-- Given the original price and sale price of an item, calculate the discount percentage. -/
def discount_percentage (original_price sale_price : ℚ) : ℚ :=
  (original_price - sale_price) / original_price * 100

/-- Theorem: The discount percentage for shoes with original price $204 and sale price $51 is 75%. -/
theorem shoes_discount_percentage :
  discount_percentage 204 51 = 75 := by sorry

end shoes_discount_percentage_l1491_149120


namespace derivative_of_exp_2x_l1491_149160

theorem derivative_of_exp_2x (x : ℝ) :
  deriv (fun x => Real.exp (2 * x)) x = 2 * Real.exp (2 * x) := by
  sorry

end derivative_of_exp_2x_l1491_149160


namespace incorrect_absolute_value_expression_l1491_149129

theorem incorrect_absolute_value_expression : 
  ((-|5|)^2 = 25) ∧ 
  (|((-5)^2)| = 25) ∧ 
  ((-|5|)^2 = 25) ∧ 
  ¬((|(-5)|)^2 = 25) := by sorry

end incorrect_absolute_value_expression_l1491_149129


namespace hyperbola_asymptote_through_point_implies_b_and_eccentricity_l1491_149157

/-- A hyperbola with equation x^2 - y^2/b^2 = 1 where b > 0 -/
structure Hyperbola where
  b : ℝ
  h_pos : b > 0

/-- The asymptote of the hyperbola -/
def asymptote (h : Hyperbola) (x : ℝ) : ℝ := h.b * x

theorem hyperbola_asymptote_through_point_implies_b_and_eccentricity
  (h : Hyperbola)
  (h_asymptote : asymptote h 1 = 2) :
  h.b = 2 ∧ Real.sqrt ((1 : ℝ)^2 + h.b^2) / 1 = Real.sqrt 5 := by
  sorry

end hyperbola_asymptote_through_point_implies_b_and_eccentricity_l1491_149157


namespace additional_miles_is_33_l1491_149185

/-- Represents the distances between locations in Kona's trip -/
structure TripDistances where
  apartment_to_bakery : ℕ
  bakery_to_grandma : ℕ
  grandma_to_apartment : ℕ

/-- Calculates the additional miles driven with bakery stop compared to without -/
def additional_miles (d : TripDistances) : ℕ :=
  d.apartment_to_bakery + d.bakery_to_grandma + d.grandma_to_apartment - 2 * d.grandma_to_apartment

/-- Theorem stating that the additional miles driven with bakery stop is 33 -/
theorem additional_miles_is_33 (d : TripDistances) 
    (h1 : d.apartment_to_bakery = 9)
    (h2 : d.bakery_to_grandma = 24)
    (h3 : d.grandma_to_apartment = 27) : 
  additional_miles d = 33 := by
  sorry

end additional_miles_is_33_l1491_149185


namespace sequence_arithmetic_progression_l1491_149111

theorem sequence_arithmetic_progression
  (s : ℕ → ℕ)
  (h_increasing : ∀ n, s n < s (n + 1))
  (h_positive : ∀ n, s n > 0)
  (h_subseq1 : ∃ a d : ℕ, ∀ n, s (s n) = a + n * d)
  (h_subseq2 : ∃ b e : ℕ, ∀ n, s (s n + 1) = b + n * e) :
  ∃ c f : ℕ, ∀ n, s n = c + n * f := by
sorry

end sequence_arithmetic_progression_l1491_149111


namespace evaluate_expression_l1491_149107

theorem evaluate_expression : -(16 / 4 * 7 + 25 - 2 * 7) = -39 := by
  sorry

end evaluate_expression_l1491_149107


namespace max_profit_multimedia_devices_l1491_149168

/-- Represents the profit function for multimedia devices -/
def profit_function (x : ℝ) : ℝ := -0.1 * x + 20

/-- Represents the constraint on the quantity of devices -/
def quantity_constraint (x : ℝ) : Prop := 4 * x ≥ 50 - x

/-- Theorem stating the maximum profit and optimal quantity of type A devices -/
theorem max_profit_multimedia_devices :
  ∃ (x : ℝ), 
    quantity_constraint x ∧ 
    profit_function x = 19 ∧ 
    x = 10 ∧
    ∀ (y : ℝ), quantity_constraint y → profit_function y ≤ profit_function x :=
by
  sorry


end max_profit_multimedia_devices_l1491_149168


namespace binomial_sum_odd_terms_l1491_149103

theorem binomial_sum_odd_terms (n : ℕ) (h : n > 0) (h_equal : Nat.choose n 4 = Nat.choose n 6) :
  (Finset.range ((n + 1) / 2)).sum (fun k => Nat.choose n (2 * k)) = 2^(n - 1) :=
sorry

end binomial_sum_odd_terms_l1491_149103


namespace trigonometric_simplification_l1491_149117

theorem trigonometric_simplification (θ : Real) (h : 0 < θ ∧ θ < π) :
  ((1 + Real.sin θ + Real.cos θ) * (Real.sin (θ/2) - Real.cos (θ/2))) / 
  Real.sqrt (2 + 2 * Real.cos θ) = -Real.cos θ := by
  sorry

end trigonometric_simplification_l1491_149117


namespace smallest_number_with_remainders_l1491_149194

theorem smallest_number_with_remainders : ∃ (b : ℕ), b = 87 ∧
  b % 5 = 2 ∧ b % 4 = 3 ∧ b % 7 = 1 ∧
  ∀ (n : ℕ), n % 5 = 2 ∧ n % 4 = 3 ∧ n % 7 = 1 → b ≤ n :=
by sorry

end smallest_number_with_remainders_l1491_149194


namespace sequence_general_term_l1491_149136

theorem sequence_general_term (a : ℕ → ℝ) (h1 : a 1 = 6) 
  (h2 : ∀ n : ℕ, a (n + 1) = 2 * a n + 3 * 5^n) : 
  ∀ n : ℕ, n ≥ 1 → a n = 5^n + 2^(n - 1) := by
  sorry

end sequence_general_term_l1491_149136


namespace five_fridays_in_october_implies_five_mondays_in_november_l1491_149189

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a specific date in a month -/
structure Date :=
  (day : Nat)
  (dayOfWeek : DayOfWeek)

def october_has_five_fridays (year : Nat) : Prop :=
  ∃ dates : List Date,
    dates.length = 5 ∧
    ∀ d ∈ dates, d.dayOfWeek = DayOfWeek.Friday ∧ d.day ≤ 31

def november_has_five_mondays (year : Nat) : Prop :=
  ∃ dates : List Date,
    dates.length = 5 ∧
    ∀ d ∈ dates, d.dayOfWeek = DayOfWeek.Monday ∧ d.day ≤ 30

theorem five_fridays_in_october_implies_five_mondays_in_november (year : Nat) :
  october_has_five_fridays year → november_has_five_mondays year :=
by
  sorry


end five_fridays_in_october_implies_five_mondays_in_november_l1491_149189


namespace infinite_series_sum_l1491_149167

theorem infinite_series_sum : 
  let series := fun n : ℕ => (n : ℝ) / 8^n
  ∑' n, series n = 8 / 49 := by
sorry

end infinite_series_sum_l1491_149167


namespace tank_volume_l1491_149144

/-- Given a cube-shaped tank constructed from metal sheets, calculate its volume in liters -/
theorem tank_volume (sheet_length : ℝ) (sheet_width : ℝ) (num_sheets : ℕ) : 
  sheet_length = 2 →
  sheet_width = 3 →
  num_sheets = 100 →
  (((num_sheets * sheet_length * sheet_width / 6) ^ (1/2 : ℝ)) ^ 3) * 1000 = 1000000 := by
  sorry

#check tank_volume

end tank_volume_l1491_149144


namespace perpendicular_lines_planes_l1491_149112

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation for lines and planes
variable (perp_line_plane : Line → Plane → Prop)
variable (perp_line_line : Line → Line → Prop)
variable (perp_plane_plane : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_lines_planes 
  (a b : Line) (α β : Plane) 
  (h_non_coincident : a ≠ b) 
  (h_a_perp_α : perp_line_plane a α) 
  (h_b_perp_β : perp_line_plane b β) : 
  (perp_line_line a b ↔ perp_plane_plane α β) :=
sorry

end perpendicular_lines_planes_l1491_149112


namespace m_range_theorem_l1491_149134

-- Define the conditions
def p (x : ℝ) : Prop := -2 < x ∧ x < 10
def q (x m : ℝ) : Prop := (x - 1)^2 - m^2 ≤ 0

-- Define the theorem
theorem m_range_theorem :
  (∀ x, p x → q x m) ∧ 
  (∃ x, q x m ∧ ¬p x) ∧ 
  (m > 0) →
  m ≥ 9 :=
sorry

end m_range_theorem_l1491_149134


namespace intersection_tangent_negative_x_l1491_149113

theorem intersection_tangent_negative_x (x₀ y₀ : ℝ) : 
  x₀ > 0 → y₀ = Real.tan x₀ → y₀ = -x₀ → 
  (x₀^2 + 1) * (Real.cos (2 * x₀) + 1) = 2 := by sorry

end intersection_tangent_negative_x_l1491_149113


namespace upward_shift_quadratic_l1491_149187

/-- The original quadratic function -/
def f (x : ℝ) : ℝ := -x^2

/-- The amount of upward shift -/
def shift : ℝ := 2

/-- The shifted function -/
def g (x : ℝ) : ℝ := f x + shift

theorem upward_shift_quadratic :
  ∀ x : ℝ, g x = -(x^2) + 2 := by
  sorry

end upward_shift_quadratic_l1491_149187


namespace parabola_vertex_l1491_149195

/-- The parabola equation -/
def parabola_equation (x : ℝ) : ℝ := x^2 - 4*x + 7

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (2, 3)

/-- Theorem: The vertex of the parabola y = x^2 - 4x + 7 is at the point (2, 3) -/
theorem parabola_vertex :
  let (h, k) := vertex
  (∀ x, parabola_equation x = (x - h)^2 + k) ∧
  (∀ x, parabola_equation x ≥ parabola_equation h) :=
by sorry

end parabola_vertex_l1491_149195


namespace quadratic_max_value_l1491_149108

/-- A quadratic function that takes specific values at consecutive natural numbers -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ n : ℕ, f n = 6 ∧ f (n + 1) = 14 ∧ f (n + 2) = 14

/-- The theorem stating the maximum value of the quadratic function -/
theorem quadratic_max_value (f : ℝ → ℝ) (h : QuadraticFunction f) :
  ∃ c : ℝ, c = 15 ∧ ∀ x : ℝ, f x ≤ c :=
sorry

end quadratic_max_value_l1491_149108


namespace third_number_is_two_l1491_149163

def is_valid_sequence (seq : List Nat) : Prop :=
  seq.length = 37 ∧
  seq.toFinset = Finset.range 37 ∧
  ∀ i j, i < j → j < seq.length → (seq.take j).sum % seq[j]! = 0

theorem third_number_is_two (seq : List Nat) :
  is_valid_sequence seq →
  seq[0]! = 37 →
  seq[1]! = 1 →
  seq[2]! = 2 :=
by sorry

end third_number_is_two_l1491_149163


namespace framing_for_enlarged_picture_l1491_149142

/-- Calculates the minimum number of linear feet of framing needed for an enlarged picture with border -/
def minimum_framing_feet (original_width original_height enlargement_factor border_width : ℕ) : ℕ :=
  let enlarged_width := original_width * enlargement_factor
  let enlarged_height := original_height * enlargement_factor
  let final_width := enlarged_width + 2 * border_width
  let final_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (final_width + final_height)
  (perimeter_inches + 11) / 12  -- Dividing by 12 and rounding up

/-- Theorem stating that the minimum framing needed for the given picture is 10 feet -/
theorem framing_for_enlarged_picture :
  minimum_framing_feet 5 7 4 3 = 10 := by
  sorry

end framing_for_enlarged_picture_l1491_149142


namespace train_platform_crossing_time_l1491_149193

/-- Given a train and platform with specific lengths, and the time taken to cross a post,
    calculate the time taken to cross the platform. -/
theorem train_platform_crossing_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (time_to_cross_post : ℝ) 
  (h1 : train_length = 300)
  (h2 : platform_length = 350)
  (h3 : time_to_cross_post = 18) :
  (train_length + platform_length) / (train_length / time_to_cross_post) = 39 :=
sorry

end train_platform_crossing_time_l1491_149193


namespace bricks_for_wall_l1491_149176

/-- Calculates the number of bricks needed to build a wall -/
def bricks_needed (wall_length wall_height wall_thickness brick_length brick_width brick_height : ℕ) : ℕ :=
  let wall_volume := wall_length * wall_height * wall_thickness
  let brick_volume := brick_length * brick_width * brick_height
  (wall_volume + brick_volume - 1) / brick_volume

/-- Theorem stating the number of bricks needed for the given wall and brick dimensions -/
theorem bricks_for_wall : bricks_needed 800 600 2 5 11 6 = 2910 := by
  sorry

end bricks_for_wall_l1491_149176


namespace house_of_cards_impossible_l1491_149198

theorem house_of_cards_impossible (decks : ℕ) (cards_per_deck : ℕ) (layers : ℕ) : 
  decks = 36 → cards_per_deck = 104 → layers = 64 → 
  ¬ ∃ (cards_per_layer : ℕ), (decks * cards_per_deck) = (layers * cards_per_layer) :=
by
  sorry

end house_of_cards_impossible_l1491_149198


namespace finite_square_solutions_l1491_149118

theorem finite_square_solutions (a b : ℤ) (h : ¬ ∃ k : ℤ, b = k^2) :
  { x : ℤ | ∃ y : ℤ, x^2 + a*x + b = y^2 }.Finite :=
sorry

end finite_square_solutions_l1491_149118


namespace collinear_points_x_value_l1491_149116

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p2.x) = (p3.y - p2.y) * (p2.x - p1.x)

/-- The main theorem -/
theorem collinear_points_x_value :
  let A : Point := ⟨-1, 1⟩
  let B : Point := ⟨2, -4⟩
  let C : Point := ⟨x, -9⟩
  collinear A B C → x = 5 := by
  sorry


end collinear_points_x_value_l1491_149116


namespace drilled_cube_surface_area_l1491_149148

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with a drilled tunnel -/
structure DrilledCube where
  edgeLength : ℝ
  tunnelStartDistance : ℝ

/-- Calculates the surface area of a drilled cube -/
noncomputable def surfaceArea (cube : DrilledCube) : ℝ :=
  sorry

theorem drilled_cube_surface_area :
  let cube : DrilledCube := { edgeLength := 10, tunnelStartDistance := 3 }
  surfaceArea cube = 582 + 42 * Real.sqrt 6 := by
  sorry

end drilled_cube_surface_area_l1491_149148


namespace kennel_cats_dogs_difference_l1491_149161

/-- Proves that in a kennel with a 2:3 ratio of cats to dogs and 18 dogs, there are 6 fewer cats than dogs -/
theorem kennel_cats_dogs_difference :
  ∀ (num_cats num_dogs : ℕ),
  num_dogs = 18 →
  num_cats * 3 = num_dogs * 2 →
  num_cats < num_dogs →
  num_dogs - num_cats = 6 :=
by
  sorry

end kennel_cats_dogs_difference_l1491_149161


namespace initial_students_count_l1491_149188

theorem initial_students_count (initial_avg : ℝ) (new_student_weight : ℝ) (new_avg : ℝ) :
  initial_avg = 28 →
  new_student_weight = 4 →
  new_avg = 27.2 →
  ∃ n : ℕ, n * initial_avg + new_student_weight = (n + 1) * new_avg ∧ n = 29 :=
by sorry

end initial_students_count_l1491_149188


namespace max_profit_at_0_032_l1491_149124

-- Define the bank's profit function
def bankProfit (k : ℝ) (x : ℝ) : ℝ := 0.048 * k * x^2 - k * x^3

-- State the theorem
theorem max_profit_at_0_032 (k : ℝ) (h_k : k > 0) :
  ∃ (x : ℝ), x ∈ Set.Ioo 0 0.048 ∧
  ∀ (y : ℝ), y ∈ Set.Ioo 0 0.048 → bankProfit k x ≥ bankProfit k y :=
sorry

end max_profit_at_0_032_l1491_149124


namespace ratio_arithmetic_sequence_property_l1491_149127

/-- Definition of a ratio arithmetic sequence -/
def is_ratio_arithmetic (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a (n + 2) / a (n + 1) - a (n + 1) / a n = d

/-- Theorem about the specific ratio arithmetic sequence -/
theorem ratio_arithmetic_sequence_property (a : ℕ → ℝ) (d : ℝ) :
  is_ratio_arithmetic a d →
  a 1 = 1 →
  a 2 = 1 →
  a 3 = 2 →
  a 2009 / a 2006 = 2006 := by
sorry

end ratio_arithmetic_sequence_property_l1491_149127


namespace ax5_plus_by5_l1491_149173

theorem ax5_plus_by5 (a b x y : ℝ) 
  (h1 : a * x + b * y = 4)
  (h2 : a * x^2 + b * y^2 = 10)
  (h3 : a * x^3 + b * y^3 = 28)
  (h4 : a * x^4 + b * y^4 = 60) :
  a * x^5 + b * y^5 = 229 + 1/3 := by
sorry

end ax5_plus_by5_l1491_149173


namespace board_numbers_problem_l1491_149191

theorem board_numbers_problem (a b c : ℕ) :
  70 ≤ a ∧ a < 80 ∧
  60 ≤ b ∧ b < 70 ∧
  50 ≤ c ∧ c < 60 ∧
  a + b = 147 ∧
  120 ≤ a + c ∧ a + c < 130 ∧
  120 ≤ b + c ∧ b + c < 130 ∧
  a + c ≠ b + c →
  a = 78 :=
by sorry

end board_numbers_problem_l1491_149191


namespace min_sum_squares_reciprocal_inequality_l1491_149155

-- Define the set D
def D : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2 ∧ p.1 > 0 ∧ p.2 > 0}

-- Theorem 1: Minimum value of x₁² + x₂²
theorem min_sum_squares (p : ℝ × ℝ) (h : p ∈ D) : p.1^2 + p.2^2 ≥ 2 := by
  sorry

-- Theorem 2: Inequality for reciprocals
theorem reciprocal_inequality (p : ℝ × ℝ) (h : p ∈ D) :
  1 / (p.1 + 2*p.2) + 1 / (2*p.1 + p.2) ≥ 2/3 := by
  sorry

end min_sum_squares_reciprocal_inequality_l1491_149155


namespace power_function_property_l1491_149110

/-- A power function is a function of the form f(x) = x^α for some real α -/
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

theorem power_function_property (f : ℝ → ℝ) (h1 : isPowerFunction f) (h2 : f 2 = 4) :
  f 3 = 9 := by
  sorry

end power_function_property_l1491_149110


namespace projection_x_coordinate_l1491_149154

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Theorem: The x-coordinate of the projection of a point on a circle onto the x-axis -/
theorem projection_x_coordinate 
  (circle : Circle)
  (start : Point)
  (B : Point)
  (angle : ℝ) :
  circle.center = Point.mk 0 0 →
  circle.radius = 4 →
  start = Point.mk 4 0 →
  B.x = 4 * Real.cos angle →
  B.y = 4 * Real.sin angle →
  angle ≥ 0 →
  4 * Real.cos angle = (Point.mk (B.x) 0).x :=
by sorry

end projection_x_coordinate_l1491_149154


namespace circle_lattice_point_uniqueness_l1491_149178

theorem circle_lattice_point_uniqueness (r : ℝ) (hr : r > 0) :
  ∃! (x y : ℤ), (↑x - Real.sqrt 2)^2 + (↑y - 1/3)^2 = r^2 :=
by sorry

end circle_lattice_point_uniqueness_l1491_149178


namespace min_pencils_in_box_l1491_149179

theorem min_pencils_in_box (total_boxes : Nat) (total_pencils : Nat) (max_capacity : Nat)
  (h1 : total_boxes = 13)
  (h2 : total_pencils = 74)
  (h3 : max_capacity = 6) :
  ∃ (min_pencils : Nat), min_pencils = 2 ∧
    (∀ (box : Nat), box ≤ total_boxes → ∃ (pencils_in_box : Nat),
      pencils_in_box ≥ min_pencils ∧ pencils_in_box ≤ max_capacity) ∧
    (∃ (box : Nat), box ≤ total_boxes ∧ ∃ (pencils_in_box : Nat), pencils_in_box = min_pencils) :=
by
  sorry

end min_pencils_in_box_l1491_149179


namespace contrapositive_statement_l1491_149170

theorem contrapositive_statement (a b : ℝ) :
  (a > 0 ∧ a + b < 0) → b < 0 := by
  sorry

end contrapositive_statement_l1491_149170


namespace trigonometric_inequality_l1491_149172

theorem trigonometric_inequality (φ : Real) (h : φ ∈ Set.Ioo 0 (Real.pi / 2)) :
  Real.sin (Real.cos φ) < Real.cos φ ∧ Real.cos φ < Real.cos (Real.sin φ) := by
  sorry

end trigonometric_inequality_l1491_149172


namespace max_price_correct_optimal_price_correct_max_profit_correct_l1491_149109

/-- Represents the beverage pricing and sales model for a food company. -/
structure BeverageModel where
  initial_price : ℝ
  initial_cost : ℝ
  initial_sales : ℝ
  price_sensitivity : ℝ
  marketing_cost : ℝ → ℝ
  sales_decrease : ℝ → ℝ

/-- The maximum price that ensures the total profit is not lower than the initial profit. -/
def max_price (model : BeverageModel) : ℝ :=
  model.initial_price + 5

/-- The price that maximizes the total profit under the new marketing strategy. -/
def optimal_price (model : BeverageModel) : ℝ := 19

/-- The maximum total profit under the new marketing strategy. -/
def max_profit (model : BeverageModel) : ℝ := 45.45

/-- Theorem stating the correctness of the maximum price. -/
theorem max_price_correct (model : BeverageModel) 
  (h1 : model.initial_price = 15)
  (h2 : model.initial_cost = 10)
  (h3 : model.initial_sales = 80000)
  (h4 : model.price_sensitivity = 8000) :
  max_price model = 20 := by sorry

/-- Theorem stating the correctness of the optimal price for maximum profit. -/
theorem optimal_price_correct (model : BeverageModel) 
  (h1 : model.initial_cost = 10)
  (h2 : ∀ x, x ≥ 16 → model.marketing_cost x = (33/4) * (x - 16))
  (h3 : ∀ x, model.sales_decrease x = 0.8 / ((x - 15)^2)) :
  optimal_price model = 19 := by sorry

/-- Theorem stating the correctness of the maximum total profit. -/
theorem max_profit_correct (model : BeverageModel) 
  (h1 : model.initial_cost = 10)
  (h2 : ∀ x, x ≥ 16 → model.marketing_cost x = (33/4) * (x - 16))
  (h3 : ∀ x, model.sales_decrease x = 0.8 / ((x - 15)^2)) :
  max_profit model = 45.45 := by sorry

end max_price_correct_optimal_price_correct_max_profit_correct_l1491_149109


namespace fraction_comparison_l1491_149171

theorem fraction_comparison : 
  (14/10 : ℚ) = 7/5 ∧ 
  (1 + 2/5 : ℚ) = 7/5 ∧ 
  (1 + 14/35 : ℚ) = 7/5 ∧ 
  (1 + 4/20 : ℚ) ≠ 7/5 ∧ 
  (1 + 3/15 : ℚ) ≠ 7/5 := by
  sorry

end fraction_comparison_l1491_149171


namespace total_cost_is_correct_l1491_149122

def cement_bags : ℕ := 500
def cement_price_per_bag : ℚ := 10
def cement_discount_rate : ℚ := 5 / 100
def sand_lorries : ℕ := 20
def sand_tons_per_lorry : ℕ := 10
def sand_price_per_ton : ℚ := 40
def tax_rate_first_half : ℚ := 7 / 100
def tax_rate_second_half : ℚ := 5 / 100

def total_cost : ℚ := sorry

theorem total_cost_is_correct : 
  total_cost = 13230 := by sorry

end total_cost_is_correct_l1491_149122


namespace inequality_proof_l1491_149125

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : c < 0) : a * (c - 1) < b * (c - 1) := by
  sorry

end inequality_proof_l1491_149125


namespace one_divides_six_digit_number_l1491_149199

/-- Represents a 6-digit number of the form abacab -/
def SixDigitNumber (a b c : ℕ) : ℕ :=
  100000 * a + 10000 * b + 1000 * a + 100 * c + 10 * a + b

/-- Theorem stating that 1 is a factor of any SixDigitNumber -/
theorem one_divides_six_digit_number (a b c : ℕ) (h1 : a ≠ 0) (h2 : a < 10) (h3 : b < 10) (h4 : c < 10) :
  1 ∣ SixDigitNumber a b c := by
  sorry


end one_divides_six_digit_number_l1491_149199


namespace quadratic_decreasing_before_vertex_l1491_149153

def f (x : ℝ) : ℝ := 2 * (x - 3)^2 + 1

theorem quadratic_decreasing_before_vertex :
  ∀ (x1 x2 : ℝ), x1 < x2 → x2 < 3 → f x1 > f x2 := by
  sorry

end quadratic_decreasing_before_vertex_l1491_149153


namespace buffy_stolen_apples_l1491_149181

theorem buffy_stolen_apples (initial_apples : ℕ) (fallen_apples : ℕ) (remaining_apples : ℕ) 
  (h1 : initial_apples = 79)
  (h2 : fallen_apples = 26)
  (h3 : remaining_apples = 8) :
  initial_apples - fallen_apples - remaining_apples = 45 :=
by sorry

end buffy_stolen_apples_l1491_149181


namespace incircle_identity_l1491_149166

-- Define a triangle with an incircle
structure TriangleWithIncircle where
  -- The sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- The semi-perimeter
  p : ℝ
  -- The inradius
  r : ℝ
  -- The angle APB
  α : ℝ
  -- Conditions
  side_positive : 0 < a ∧ 0 < b ∧ 0 < c
  semi_perimeter : p = (a + b + c) / 2
  inradius_positive : 0 < r
  angle_positive : 0 < α ∧ α < π / 2

-- The theorem to prove
theorem incircle_identity (t : TriangleWithIncircle) :
  1 / (t.p - t.b) + 1 / (t.p - t.c) = 2 / (t.r * Real.tan t.α) := by
  sorry

end incircle_identity_l1491_149166


namespace wages_theorem_l1491_149106

/-- 
Given:
- A sum of money can pay A's wages for 20 days
- The same sum of money can pay B's wages for 30 days

Prove:
The same sum of money can pay both A and B's wages together for 12 days
-/
theorem wages_theorem (A B : ℝ) (h1 : 20 * A = 30 * B) : 
  12 * (A + B) = 20 * A := by sorry

end wages_theorem_l1491_149106


namespace teds_age_l1491_149135

theorem teds_age (t s : ℝ) 
  (h1 : t = 3 * s - 10)  -- Ted's age is 10 years less than three times Sally's age
  (h2 : t + s = 60)      -- The sum of their ages is 60
  : t = 42.5 :=          -- Ted's age is 42.5
by sorry

end teds_age_l1491_149135


namespace retreat_speed_l1491_149180

theorem retreat_speed (total_distance : ℝ) (total_time : ℝ) (return_speed : ℝ) :
  total_distance = 600 →
  total_time = 10 →
  return_speed = 75 →
  ∃ outbound_speed : ℝ,
    outbound_speed = 50 ∧
    total_time = (total_distance / 2) / outbound_speed + (total_distance / 2) / return_speed :=
by sorry

end retreat_speed_l1491_149180


namespace m_salary_percentage_l1491_149182

/-- The percentage of m's salary compared to n's salary -/
def salary_percentage (total_salary n_salary : ℚ) : ℚ :=
  (total_salary - n_salary) / n_salary * 100

/-- Proof that m's salary is 120% of n's salary -/
theorem m_salary_percentage :
  let total_salary : ℚ := 572
  let n_salary : ℚ := 260
  salary_percentage total_salary n_salary = 120 := by
  sorry

end m_salary_percentage_l1491_149182


namespace parallel_line_equation_perpendicular_line_equation_l1491_149126

-- Define the intersection point
def intersection_point : ℝ × ℝ := (3, 2)

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := 3 * x - 2 * y + 4 = 0

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 4 * x - 3 * y - 7 = 0

-- Theorem for the parallel case
theorem parallel_line_equation :
  ∃ (m : ℝ), ∀ (x y : ℝ),
    (x, y) = intersection_point →
    (3 * x - 2 * y + m = 0 ↔ parallel_line x y) :=
sorry

-- Theorem for the perpendicular case
theorem perpendicular_line_equation :
  ∃ (n : ℝ), ∀ (x y : ℝ),
    (x, y) = intersection_point →
    (3 * x + 4 * y + n = 0 ↔ perpendicular_line x y) :=
sorry

end parallel_line_equation_perpendicular_line_equation_l1491_149126


namespace train_passing_time_l1491_149149

theorem train_passing_time 
  (L : ℝ) 
  (v₁ v₂ : ℝ) 
  (h₁ : L > 0) 
  (h₂ : v₁ > 0) 
  (h₃ : v₂ > 0) : 
  (2 * L) / ((v₁ + v₂) * (1000 / 3600)) = 
  (2 * L) / ((v₁ + v₂) * (1000 / 3600)) :=
by sorry

#check train_passing_time

end train_passing_time_l1491_149149


namespace bus_station_arrangement_count_l1491_149143

/-- The number of seats in the bus station -/
def num_seats : ℕ := 10

/-- The number of passengers -/
def num_passengers : ℕ := 4

/-- The number of consecutive empty seats required -/
def consecutive_empty_seats : ℕ := 5

/-- The number of ways to arrange passengers with the required consecutive empty seats -/
def arrangement_count : ℕ := 480

/-- Theorem stating that the number of ways to arrange passengers with the required consecutive empty seats is correct -/
theorem bus_station_arrangement_count :
  (num_seats : ℕ) = 10 →
  (num_passengers : ℕ) = 4 →
  (consecutive_empty_seats : ℕ) = 5 →
  (arrangement_count : ℕ) = 480 := by
  sorry

end bus_station_arrangement_count_l1491_149143


namespace repeating_decimal_47_l1491_149159

theorem repeating_decimal_47 : ∃ (x : ℚ), x = 47 / 99 ∧ ∀ (n : ℕ), (x * 10^(2*n+2) - ⌊x * 10^(2*n+2)⌋ : ℚ) = x := by
  sorry

end repeating_decimal_47_l1491_149159


namespace ambiguous_date_and_longest_periods_l1491_149156

/-- Represents a date in DD/MM format -/
structure Date :=
  (day : Nat)
  (month : Nat)

/-- Checks if a date is valid in both DD/MM and MM/DD formats -/
def Date.isAmbiguous (d : Date) : Prop :=
  d.day ≤ 12 ∧ d.month ≤ 12 ∧ d.day ≠ d.month

/-- Checks if a date is within the range of January 2nd to January 12th or December 2nd to December 12th -/
def Date.isInLongestAmbiguousPeriod (d : Date) : Prop :=
  (d.month = 1 ∧ d.day ≥ 2 ∧ d.day ≤ 12) ∨ (d.month = 12 ∧ d.day ≥ 2 ∧ d.day ≤ 12)

theorem ambiguous_date_and_longest_periods :
  (∃ d : Date, d.day = 3 ∧ d.month = 12 ∧ d.isAmbiguous) ∧
  (∀ d : Date, d.isAmbiguous → d.isInLongestAmbiguousPeriod ∨ ¬d.isInLongestAmbiguousPeriod) ∧
  (∀ d : Date, d.isInLongestAmbiguousPeriod → d.isAmbiguous) :=
sorry

end ambiguous_date_and_longest_periods_l1491_149156


namespace roof_area_l1491_149145

theorem roof_area (width length : ℝ) : 
  width > 0 →
  length > 0 →
  length = 4 * width →
  length - width = 48 →
  width * length = 1024 := by
  sorry

end roof_area_l1491_149145


namespace baby_age_at_weight_7200_l1491_149139

/-- The relationship between a baby's weight and age -/
def weight_age_relation (a : ℝ) (x : ℝ) : ℝ := a + 800 * x

/-- The theorem stating the age of the baby when their weight is 7200 grams -/
theorem baby_age_at_weight_7200 (a : ℝ) (x : ℝ) 
  (h1 : a = 3200) -- The baby's weight at birth is 3200 grams
  (h2 : weight_age_relation a x = 7200) -- The baby's weight is 7200 grams
  : x = 5 := by
  sorry

#check baby_age_at_weight_7200

end baby_age_at_weight_7200_l1491_149139


namespace special_sequence_has_repeats_l1491_149131

/-- A sequence of rational numbers satisfying the given property -/
def SpecialSequence := ℕ → ℚ

/-- The property that defines our special sequence -/
def HasSpecialProperty (a : SpecialSequence) : Prop :=
  ∀ m n : ℕ, a m + a n = a (m * n)

/-- The theorem stating that a sequence with the special property has repeated elements -/
theorem special_sequence_has_repeats (a : SpecialSequence) (h : HasSpecialProperty a) :
  ∃ i j : ℕ, i ≠ j ∧ a i = a j := by sorry

end special_sequence_has_repeats_l1491_149131


namespace polygon_exterior_angles_l1491_149175

theorem polygon_exterior_angles (n : ℕ) (exterior_angle : ℝ) : 
  (n > 2) → (exterior_angle = 45) → (n * exterior_angle = 360) → n = 8 := by
  sorry

end polygon_exterior_angles_l1491_149175


namespace dollar_symmetric_sum_l1491_149114

def dollar (a b : ℝ) : ℝ := (a - b)^2

theorem dollar_symmetric_sum (x y : ℝ) : dollar (x + y) (y + x) = 0 := by
  sorry

end dollar_symmetric_sum_l1491_149114


namespace johns_remaining_money_is_135_l1491_149138

/-- Calculates John's remaining money after dog walking and expenses in April --/
def johns_remaining_money : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ :=
  fun (total_days : ℕ) (sundays : ℕ) (weekday_rate : ℕ) (weekend_rate : ℕ)
      (mark_help_days : ℕ) (book_cost : ℕ) (book_discount : ℕ)
      (sister_percentage : ℕ) (gift_cost : ℕ) =>
    let working_days := total_days - sundays
    let weekends := sundays
    let weekdays := working_days - weekends
    let weekday_earnings := weekdays * weekday_rate
    let weekend_earnings := weekends * weekend_rate
    let mark_split_earnings := (mark_help_days * weekday_rate) / 2
    let total_earnings := weekday_earnings + weekend_earnings + mark_split_earnings
    let discounted_book_cost := book_cost - (book_cost * book_discount / 100)
    let after_books := total_earnings - discounted_book_cost
    let sister_share := after_books * sister_percentage / 100
    let after_sister := after_books - sister_share
    let after_gift := after_sister - gift_cost
    let food_cost := weekends * 10
    after_gift - food_cost

theorem johns_remaining_money_is_135 :
  johns_remaining_money 30 4 10 15 3 50 10 20 25 = 135 := by
  sorry

end johns_remaining_money_is_135_l1491_149138


namespace complex_power_one_minus_i_six_l1491_149101

theorem complex_power_one_minus_i_six :
  (1 - Complex.I) ^ 6 = 8 * Complex.I := by sorry

end complex_power_one_minus_i_six_l1491_149101


namespace max_m_value_max_m_is_maximum_l1491_149141

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem max_m_value (m : ℝ) : 
  (∃ t : ℝ, ∀ x ∈ Set.Icc 2 m, f (x + t) ≤ 2 * x) → 
  m ≤ 8 ∧ ∃ t : ℝ, ∀ x ∈ Set.Icc 2 8, f (x + t) ≤ 2 * x :=
sorry

-- Define the maximum value of m
def max_m : ℝ := 8

-- Prove that max_m is indeed the maximum value
theorem max_m_is_maximum :
  (∃ t : ℝ, ∀ x ∈ Set.Icc 2 max_m, f (x + t) ≤ 2 * x) ∧
  ∀ m > max_m, ¬(∃ t : ℝ, ∀ x ∈ Set.Icc 2 m, f (x + t) ≤ 2 * x) :=
sorry

end max_m_value_max_m_is_maximum_l1491_149141


namespace milk_water_mixture_l1491_149132

/-- Given a mixture of milk and water with an initial ratio of 6:3 and a final ratio of 6:5 after
    adding 10 liters of water, the original quantity of milk is 30 liters. -/
theorem milk_water_mixture (milk : ℝ) (water : ℝ) : 
  milk / water = 6 / 3 →
  milk / (water + 10) = 6 / 5 →
  milk = 30 :=
by sorry

end milk_water_mixture_l1491_149132


namespace tile_area_calculation_l1491_149197

/-- Given a rectangular room and tiles covering a fraction of it, calculate the area of each tile. -/
theorem tile_area_calculation (room_length room_width : ℝ) (num_tiles : ℕ) (fraction_covered : ℚ) :
  room_length = 12 →
  room_width = 20 →
  num_tiles = 40 →
  fraction_covered = 1/6 →
  (room_length * room_width * fraction_covered) / num_tiles = 1 := by
  sorry

end tile_area_calculation_l1491_149197


namespace greatest_common_divisor_of_sum_l1491_149169

/-- An arithmetic sequence with positive integer terms and perfect square common difference -/
structure ArithmeticSequence where
  first_term : ℕ+
  common_difference : ℕ
  is_perfect_square : ∃ (n : ℕ), n^2 = common_difference

/-- The sum of the first 15 terms of an arithmetic sequence -/
def sum_first_15_terms (seq : ArithmeticSequence) : ℕ :=
  15 * seq.first_term + 105 * seq.common_difference

/-- 15 is the greatest positive integer that always divides the sum of the first 15 terms -/
theorem greatest_common_divisor_of_sum (seq : ArithmeticSequence) :
  (∃ (m : ℕ+), m > 15 ∧ (m : ℕ) ∣ sum_first_15_terms seq) → False := by sorry

end greatest_common_divisor_of_sum_l1491_149169


namespace cookies_per_box_l1491_149151

/-- The number of cookies in each box, given the collection amounts of Abigail, Grayson, and Olivia, and the total number of cookies. -/
theorem cookies_per_box (abigail_boxes : ℚ) (grayson_boxes : ℚ) (olivia_boxes : ℚ) (total_cookies : ℕ) :
  abigail_boxes = 2 →
  grayson_boxes = 3 / 4 →
  olivia_boxes = 3 →
  total_cookies = 276 →
  total_cookies / (abigail_boxes + grayson_boxes + olivia_boxes) = 48 := by
sorry

end cookies_per_box_l1491_149151


namespace min_coach_handshakes_l1491_149196

/-- Represents the number of handshakes involving coaches -/
def coach_handshakes (nA nB : ℕ) : ℕ := 
  620 - (nA.choose 2 + nB.choose 2 + nA * nB)

/-- The main theorem to prove -/
theorem min_coach_handshakes : 
  ∃ (nA nB : ℕ), 
    nA = nB + 2 ∧ 
    nA > 0 ∧ 
    nB > 0 ∧
    ∀ (mA mB : ℕ), 
      mA = mB + 2 → 
      mA > 0 → 
      mB > 0 → 
      coach_handshakes nA nB ≤ coach_handshakes mA mB ∧
      coach_handshakes nA nB = 189 :=
sorry

end min_coach_handshakes_l1491_149196


namespace escalator_speed_l1491_149121

theorem escalator_speed (escalator_speed : ℝ) (escalator_length : ℝ) (time_taken : ℝ) 
  (h1 : escalator_speed = 12)
  (h2 : escalator_length = 150)
  (h3 : time_taken = 10) :
  let person_speed := (escalator_length / time_taken) - escalator_speed
  person_speed = 3 := by
  sorry

end escalator_speed_l1491_149121


namespace triangle_abc_properties_l1491_149158

/-- Theorem about a triangle ABC with specific properties -/
theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < C → C < Real.pi / 2 →
  a * c * Real.cos B - b * c * Real.cos A = 3 * b^2 →
  c = Real.sqrt 11 →
  Real.sin C = 2 * Real.sqrt 2 / 3 →
  (a / b = 2) ∧ (1/2 * a * b * Real.sin C = 2 * Real.sqrt 2) := by
  sorry

end triangle_abc_properties_l1491_149158


namespace hyperbola_asymptote_l1491_149119

/-- Given a hyperbola with equation x²/m² - y² = 1 where m > 0,
    if one of its asymptote equations is x + √3 * y = 0, then m = √3 -/
theorem hyperbola_asymptote (m : ℝ) (hm : m > 0) :
  (∃ x y : ℝ, x^2 / m^2 - y^2 = 1) →
  (∃ x y : ℝ, x + Real.sqrt 3 * y = 0) →
  m = Real.sqrt 3 :=
by sorry

end hyperbola_asymptote_l1491_149119


namespace copper_percentage_in_second_alloy_l1491_149192

/-- Given a mixture of two alloys, prove the copper percentage in the second alloy -/
theorem copper_percentage_in_second_alloy
  (total_mixture : ℝ)
  (desired_copper_percentage : ℝ)
  (first_alloy_amount : ℝ)
  (first_alloy_copper_percentage : ℝ)
  (h_total_mixture : total_mixture = 100)
  (h_desired_copper_percentage : desired_copper_percentage = 24.9)
  (h_first_alloy_amount : first_alloy_amount = 30)
  (h_first_alloy_copper_percentage : first_alloy_copper_percentage = 20)
  : ∃ (second_alloy_copper_percentage : ℝ),
    second_alloy_copper_percentage = 27 ∧
    (first_alloy_amount * first_alloy_copper_percentage / 100 +
     (total_mixture - first_alloy_amount) * second_alloy_copper_percentage / 100 =
     total_mixture * desired_copper_percentage / 100) :=
by sorry

end copper_percentage_in_second_alloy_l1491_149192


namespace cookie_pattern_proof_l1491_149115

def cookie_sequence (n : ℕ) : ℕ := 
  match n with
  | 1 => 5
  | 2 => 5  -- This is what we want to prove
  | 3 => 10
  | 4 => 14
  | 5 => 19
  | 6 => 25
  | _ => 0  -- For other values, we don't care in this problem

theorem cookie_pattern_proof : 
  (cookie_sequence 1 = 5) ∧ 
  (cookie_sequence 3 = 10) ∧ 
  (cookie_sequence 4 = 14) ∧ 
  (cookie_sequence 5 = 19) ∧ 
  (cookie_sequence 6 = 25) ∧ 
  (∀ n : ℕ, n > 2 → cookie_sequence n - cookie_sequence (n-1) = 
    if n % 2 = 0 then 4 else 5) →
  cookie_sequence 2 = 5 := by
sorry

end cookie_pattern_proof_l1491_149115


namespace distance_to_line_l1491_149162

/-- Given a triangle ABC with sides AB = 3, BC = 4, and CA = 5,
    the distance from point B to line AC is 12/5 -/
theorem distance_to_line (A B C : ℝ × ℝ) : 
  let d := (λ P Q : ℝ × ℝ => Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2))
  d A B = 3 ∧ d B C = 4 ∧ d C A = 5 → 
  (let area := (1/2) * d A B * d B C
   area / d C A) = 12/5 := by
  sorry

end distance_to_line_l1491_149162


namespace product_of_numbers_with_given_sum_and_difference_l1491_149150

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 30 ∧ x - y = 10 → x * y = 200 := by
  sorry

end product_of_numbers_with_given_sum_and_difference_l1491_149150


namespace cube_root_increasing_l1491_149140

/-- The cube root function is increasing on the real numbers. -/
theorem cube_root_increasing :
  ∀ x y : ℝ, x < y → x^(1/3) < y^(1/3) := by sorry

end cube_root_increasing_l1491_149140


namespace g_behavior_l1491_149183

def g (x : ℝ) : ℝ := 3 * x^3 - 5 * x^2 + 1

theorem g_behavior :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → g x > M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < N → g x < M) := by
sorry

end g_behavior_l1491_149183


namespace number_with_specific_remainders_l1491_149100

theorem number_with_specific_remainders : ∃! (N : ℕ), N < 221 ∧ N % 13 = 11 ∧ N % 17 = 9 := by
  sorry

end number_with_specific_remainders_l1491_149100


namespace sin_double_plus_sin_squared_l1491_149104

theorem sin_double_plus_sin_squared (α : Real) (h : Real.tan α = 1/2) :
  Real.sin (2 * α) + Real.sin α ^ 2 = 1 := by sorry

end sin_double_plus_sin_squared_l1491_149104


namespace complex_modulus_l1491_149164

theorem complex_modulus (z : ℂ) : (2 - I) * z = 3 + I → Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_modulus_l1491_149164


namespace power_product_equals_two_l1491_149137

theorem power_product_equals_two :
  (1/2)^2016 * (-2)^2017 * (-1)^2017 = 2 := by
  sorry

end power_product_equals_two_l1491_149137


namespace cartons_packed_l1491_149102

theorem cartons_packed (total_cups : ℕ) (cups_per_box : ℕ) (boxes_per_carton : ℕ) 
  (h1 : total_cups = 768) 
  (h2 : cups_per_box = 12) 
  (h3 : boxes_per_carton = 8) : 
  total_cups / (cups_per_box * boxes_per_carton) = 8 := by
  sorry

end cartons_packed_l1491_149102
