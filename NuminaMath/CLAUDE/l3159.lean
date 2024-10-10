import Mathlib

namespace interest_rate_proof_l3159_315947

/-- The rate of interest given specific investment conditions -/
theorem interest_rate_proof (principal : ℝ) (time : ℝ) (simple_interest : ℝ) (compound_interest : ℝ) 
  (h1 : principal = 4000)
  (h2 : time = 2)
  (h3 : simple_interest = 400)
  (h4 : compound_interest = 410) :
  ∃ (rate : ℝ), 
    rate = 5 ∧
    simple_interest = (principal * rate * time) / 100 ∧
    compound_interest = principal * ((1 + rate / 100) ^ time - 1) := by
  sorry

end interest_rate_proof_l3159_315947


namespace greatest_product_prime_factorization_sum_l3159_315909

/-- The greatest product of positive integers summing to 2014 -/
def A : ℕ := 3^670 * 2^2

/-- The sum of all positive integers that produce A -/
def sum_of_factors : ℕ := 2014

/-- Function to calculate the sum of bases and exponents in prime factorization -/
def sum_bases_and_exponents (n : ℕ) : ℕ := sorry

theorem greatest_product_prime_factorization_sum :
  sum_bases_and_exponents A = 677 :=
sorry

end greatest_product_prime_factorization_sum_l3159_315909


namespace product_equality_proof_l3159_315903

theorem product_equality_proof : ∃! X : ℕ, 865 * 48 = X * 240 ∧ X = 173 := by sorry

end product_equality_proof_l3159_315903


namespace a_equals_two_l3159_315950

theorem a_equals_two (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x + 1 > 0) → a = 2 := by
  sorry

end a_equals_two_l3159_315950


namespace compute_expression_l3159_315914

theorem compute_expression : 
  20 * (150 / 3 + 50 / 6 + 16 / 25 + 2) = 90460 / 75 := by
  sorry

end compute_expression_l3159_315914


namespace trivia_team_distribution_l3159_315931

theorem trivia_team_distribution (total : ℕ) (not_picked : ℕ) (groups : ℕ) : 
  total = 36 → not_picked = 9 → groups = 3 → 
  (total - not_picked) / groups = 9 :=
by
  sorry

end trivia_team_distribution_l3159_315931


namespace sequence_value_l3159_315967

theorem sequence_value (a : ℕ → ℕ) (h1 : a 1 = 2) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + 2 * n) : a 100 = 9902 := by
  sorry

end sequence_value_l3159_315967


namespace odd_function_sum_l3159_315945

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_sum (f : ℝ → ℝ) (h_odd : is_odd_function f) (h_domain : ∀ x ∈ [-3, 3], f x = f x) (h_value : f 3 = -2) :
  f (-3) + f 0 = 2 :=
by sorry

end odd_function_sum_l3159_315945


namespace rectangle_perimeter_l3159_315993

theorem rectangle_perimeter (h w : ℝ) : 
  h > 0 ∧ w > 0 ∧              -- positive dimensions
  h * w = 40 ∧                 -- area of rectangle is 40
  w > 2 * h ∧                  -- width more than twice the height
  h * (w - h) = 24 →           -- area of parallelogram after folding
  2 * h + 2 * w = 28 :=        -- perimeter of original rectangle
by sorry

end rectangle_perimeter_l3159_315993


namespace tangent_line_at_origin_l3159_315907

/-- Given a real number a and a function f(x) = x^3 + ax^2 + (a - 2)x whose derivative
    is an even function, the tangent line to f(x) at the origin has equation y = -2x. -/
theorem tangent_line_at_origin (a : ℝ) :
  let f := fun x : ℝ => x^3 + a*x^2 + (a - 2)*x
  let f' := fun x : ℝ => 3*x^2 + 2*a*x + (a - 2)
  (∀ x, f' x = f' (-x)) →
  (fun x => -2*x) = fun x => (f' 0) * x :=
by sorry

end tangent_line_at_origin_l3159_315907


namespace binomial_12_choose_3_l3159_315987

theorem binomial_12_choose_3 : Nat.choose 12 3 = 220 := by
  sorry

end binomial_12_choose_3_l3159_315987


namespace inequality_proof_l3159_315911

theorem inequality_proof (a b c : ℝ) (h1 : a < b) (h2 : b < 0) (h3 : c > 0) :
  b / a < (c - b) / (c - a) := by
  sorry

end inequality_proof_l3159_315911


namespace bob_corn_rows_l3159_315934

/-- Represents the number of corn stalks in each row -/
def stalks_per_row : ℕ := 80

/-- Represents the number of corn stalks needed to produce one bushel -/
def stalks_per_bushel : ℕ := 8

/-- Represents the total number of bushels Bob will harvest -/
def total_bushels : ℕ := 50

/-- Calculates the number of rows of corn Bob has -/
def number_of_rows : ℕ := (total_bushels * stalks_per_bushel) / stalks_per_row

theorem bob_corn_rows :
  number_of_rows = 5 :=
sorry

end bob_corn_rows_l3159_315934


namespace locus_of_point_M_l3159_315973

/-- The locus of points M(x,y) forming triangles with fixed points A(-1,0) and B(1,0),
    where the sum of slopes of AM and BM is 2. -/
theorem locus_of_point_M (x y : ℝ) (h : x ≠ 1 ∧ x ≠ -1) :
  (y / (x + 1) + y / (x - 1) = 2) → (x^2 - x*y - 1 = 0) := by
  sorry

end locus_of_point_M_l3159_315973


namespace distance_sum_between_19_and_20_l3159_315954

/-- Given points A, B, and D in a coordinate plane, prove that the sum of distances AD and BD is between 19 and 20 -/
theorem distance_sum_between_19_and_20 (A B D : ℝ × ℝ) : 
  A = (15, 0) → B = (0, 0) → D = (8, 6) → 
  19 < Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) + Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) ∧
  Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) + Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) < 20 :=
by
  sorry

end distance_sum_between_19_and_20_l3159_315954


namespace coal_cost_equilibrium_point_verify_equilibrium_point_l3159_315925

/-- Represents the cost of coal at a point on the line segment AB -/
def coal_cost (x : ℝ) (from_a : Bool) : ℝ :=
  if from_a then
    3.75 + 0.008 * x
  else
    4.25 + 0.008 * (225 - x)

/-- Theorem stating the existence and uniqueness of point C -/
theorem coal_cost_equilibrium_point :
  ∃! x : ℝ, 0 ≤ x ∧ x ≤ 225 ∧
    coal_cost x true = coal_cost x false ∧
    ∀ y : ℝ, 0 ≤ y ∧ y ≤ 225 → coal_cost y true ≤ coal_cost x true ∧ coal_cost y false ≤ coal_cost x false :=
by
  sorry

/-- The actual equilibrium point -/
def equilibrium_point : ℝ := 143.75

/-- The cost of coal at the equilibrium point -/
def equilibrium_cost : ℝ := 4.90

/-- Theorem verifying the equilibrium point and cost -/
theorem verify_equilibrium_point :
  coal_cost equilibrium_point true = equilibrium_cost ∧
  coal_cost equilibrium_point false = equilibrium_cost :=
by
  sorry

end coal_cost_equilibrium_point_verify_equilibrium_point_l3159_315925


namespace flowerbed_perimeter_l3159_315948

/-- A rectangular flowerbed with given dimensions --/
structure Flowerbed where
  width : ℝ
  length : ℝ

/-- The perimeter of a rectangular flowerbed --/
def perimeter (f : Flowerbed) : ℝ := 2 * (f.length + f.width)

/-- Theorem: The perimeter of the specific flowerbed is 22 meters --/
theorem flowerbed_perimeter :
  ∃ (f : Flowerbed), f.width = 4 ∧ f.length = 2 * f.width - 1 ∧ perimeter f = 22 := by
  sorry

end flowerbed_perimeter_l3159_315948


namespace upstream_downstream_time_ratio_l3159_315959

/-- Proves that the ratio of the time taken to row upstream to the time taken to row downstream is 2:1 -/
theorem upstream_downstream_time_ratio 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (h1 : boat_speed = 78) 
  (h2 : stream_speed = 26) : 
  (boat_speed - stream_speed) / (boat_speed + stream_speed) = 1 / 2 := by
  sorry

end upstream_downstream_time_ratio_l3159_315959


namespace max_angle_MPN_at_x_equals_one_l3159_315900

structure Point where
  x : ℝ
  y : ℝ

def M : Point := ⟨-1, 2⟩
def N : Point := ⟨1, 4⟩

def angle_MPN (P : Point) : ℝ :=
  sorry  -- Definition of angle MPN

theorem max_angle_MPN_at_x_equals_one :
  ∃ (P : Point), P.y = 0 ∧ 
    (∀ (Q : Point), Q.y = 0 → angle_MPN P ≥ angle_MPN Q) ∧
    P.x = 1 := by
  sorry

#check max_angle_MPN_at_x_equals_one

end max_angle_MPN_at_x_equals_one_l3159_315900


namespace erased_number_l3159_315937

/-- Represents a quadratic polynomial ax^2 + bx + c with roots m and n -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  m : ℤ
  n : ℤ

/-- Checks if the given QuadraticPolynomial satisfies Vieta's formulas -/
def satisfiesVieta (p : QuadraticPolynomial) : Prop :=
  p.c = p.a * p.m * p.n ∧ p.b = -p.a * (p.m + p.n)

/-- Checks if four of the five numbers in the QuadraticPolynomial are 2, 3, 4, -5 -/
def hasFourOf (p : QuadraticPolynomial) : Prop :=
  (p.a = 2 ∧ p.b = 4 ∧ p.m = 3 ∧ p.n = -5) ∨
  (p.a = 2 ∧ p.b = 4 ∧ p.m = 3 ∧ p.c = -5) ∨
  (p.a = 2 ∧ p.b = 4 ∧ p.n = -5 ∧ p.c = 3) ∨
  (p.a = 2 ∧ p.m = 3 ∧ p.n = -5 ∧ p.c = 4) ∨
  (p.b = 4 ∧ p.m = 3 ∧ p.n = -5 ∧ p.c = 2)

theorem erased_number (p : QuadraticPolynomial) :
  satisfiesVieta p → hasFourOf p → 
  p.a = -30 ∨ p.b = -30 ∨ p.c = -30 ∨ p.m = -30 ∨ p.n = -30 := by
  sorry


end erased_number_l3159_315937


namespace two_thirds_cubed_l3159_315970

theorem two_thirds_cubed : (2 / 3 : ℚ) ^ 3 = 8 / 27 := by
  sorry

end two_thirds_cubed_l3159_315970


namespace equation_solution_l3159_315951

theorem equation_solution : ∃! x : ℚ, x + 2/5 = 8/15 + 1/3 ∧ x = 7/15 := by sorry

end equation_solution_l3159_315951


namespace investment_rate_problem_l3159_315901

/-- Proves that given the conditions of the investment problem, the lower interest rate is 12% --/
theorem investment_rate_problem (sum : ℝ) (time : ℝ) (high_rate : ℝ) (interest_diff : ℝ) :
  sum = 14000 →
  time = 2 →
  high_rate = 15 →
  interest_diff = 840 →
  sum * high_rate * time / 100 - sum * time * (sum * high_rate * time / 100 - interest_diff) / (sum * time) = 12 := by
  sorry


end investment_rate_problem_l3159_315901


namespace molecular_weight_BaBr2_l3159_315915

/-- The atomic weight of barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of bromine in g/mol -/
def atomic_weight_Br : ℝ := 79.90

/-- The number of bromine atoms in a barium bromide molecule -/
def Br_count : ℕ := 2

/-- The number of moles of barium bromide -/
def moles_BaBr2 : ℕ := 4

/-- Theorem: The molecular weight of 4 moles of Barium bromide (BaBr2) is 1188.52 grams -/
theorem molecular_weight_BaBr2 : 
  moles_BaBr2 * (atomic_weight_Ba + Br_count * atomic_weight_Br) = 1188.52 := by
  sorry

end molecular_weight_BaBr2_l3159_315915


namespace quadratic_inequality_range_l3159_315952

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * x + 1 > 0) ↔ a > 1 :=
sorry

end quadratic_inequality_range_l3159_315952


namespace intersection_when_a_is_two_proper_subset_condition_l3159_315930

def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2 * a + 3}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4}

theorem intersection_when_a_is_two :
  A 2 ∩ B = {x | 1 < x ∧ x ≤ 4} := by sorry

theorem proper_subset_condition (a : ℝ) :
  A a ⊂ B ↔ a ≤ -4 ∨ (-1 ≤ a ∧ a ≤ 1/2) := by sorry

end intersection_when_a_is_two_proper_subset_condition_l3159_315930


namespace fourth_term_of_geometric_sequence_l3159_315985

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

theorem fourth_term_of_geometric_sequence :
  let a₁ : ℝ := 6
  let a₈ : ℝ := 186624
  let r : ℝ := (a₈ / a₁) ^ (1 / 7)
  geometric_sequence a₁ r 4 = 1296 := by
sorry

end fourth_term_of_geometric_sequence_l3159_315985


namespace line_no_dot_count_l3159_315982

/-- Represents the number of letters in the alphabet -/
def total_letters : ℕ := 40

/-- Represents the number of letters containing both a dot and a straight line -/
def dot_and_line : ℕ := 13

/-- Represents the number of letters containing a dot but not a straight line -/
def dot_no_line : ℕ := 3

/-- Theorem stating that the number of letters containing a straight line but not a dot is 24 -/
theorem line_no_dot_count : 
  total_letters - (dot_and_line + dot_no_line) = 24 := by sorry

end line_no_dot_count_l3159_315982


namespace power_sum_equals_two_l3159_315921

theorem power_sum_equals_two : (1 : ℤ)^10 + (-1 : ℤ)^8 + (-1 : ℤ)^7 + (1 : ℤ)^5 = 2 := by
  sorry

end power_sum_equals_two_l3159_315921


namespace nonagon_diagonals_l3159_315969

/-- The number of distinct diagonals in a convex nonagon -/
def diagonals_in_nonagon : ℕ := 27

/-- A convex nonagon has 27 distinct diagonals -/
theorem nonagon_diagonals : diagonals_in_nonagon = 27 := by
  sorry

end nonagon_diagonals_l3159_315969


namespace ellipse_intersecting_lines_l3159_315926

/-- Represents an ellipse with center at the origin and foci on the x-axis -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  h : a > b ∧ b > 0

/-- The standard equation of an ellipse -/
def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- A line in the form y = kx + m -/
structure Line where
  k : ℝ
  m : ℝ
  h : k ≠ 0

/-- Theorem about a specific ellipse and lines intersecting it -/
theorem ellipse_intersecting_lines (e : Ellipse) 
  (h1 : e.a + (e.a^2 - e.b^2).sqrt = 3)
  (h2 : e.a - (e.a^2 - e.b^2).sqrt = 1) :
  (∀ x y, e.equation x y ↔ x^2/4 + y^2/3 = 1) ∧
  (∀ l : Line, ∃ M N : ℝ × ℝ,
    M ≠ N ∧
    e.equation M.1 M.2 ∧
    e.equation N.1 N.2 ∧
    M.2 = l.k * M.1 + l.m ∧
    N.2 = l.k * N.1 + l.m ∧
    M ≠ (2, 0) ∧ N ≠ (2, 0) ∧ M ≠ (-2, 0) ∧ N ≠ (-2, 0) ∧
    (M.1 - 2) * (N.1 - 2) + M.2 * N.2 = 0 →
    l.k * (2/7) + l.m = 0) :=
sorry

end ellipse_intersecting_lines_l3159_315926


namespace original_buckets_count_l3159_315953

/-- The number of buckets needed to fill a tank with reduced capacity buckets -/
def reduced_buckets : ℕ := 105

/-- The ratio of the reduced bucket capacity to the original bucket capacity -/
def capacity_ratio : ℚ := 2 / 5

/-- The volume of the tank in terms of original bucket capacity -/
def tank_volume (original_buckets : ℕ) : ℚ := original_buckets

/-- The volume of the tank in terms of reduced bucket capacity -/
def tank_volume_reduced : ℚ := reduced_buckets * capacity_ratio

theorem original_buckets_count : 
  ∃ (original_buckets : ℕ), 
    tank_volume original_buckets = tank_volume_reduced ∧ 
    original_buckets = 42 :=
sorry

end original_buckets_count_l3159_315953


namespace some_number_value_l3159_315924

theorem some_number_value (x y N : ℝ) 
  (eq1 : 2 * x + y = N) 
  (eq2 : x + 2 * y = 5) 
  (eq3 : (x + y) / 3 = 1) : 
  N = 4 := by
  sorry

end some_number_value_l3159_315924


namespace evie_shells_left_l3159_315979

/-- The number of shells Evie has left after collecting for 6 days and giving some to her brother -/
def shells_left (days : ℕ) (shells_per_day : ℕ) (shells_given : ℕ) : ℕ :=
  days * shells_per_day - shells_given

/-- Theorem stating that Evie has 58 shells left -/
theorem evie_shells_left : shells_left 6 10 2 = 58 := by
  sorry

end evie_shells_left_l3159_315979


namespace initial_deposit_proof_l3159_315904

/-- Represents the initial deposit amount in dollars -/
def initial_deposit : ℝ := 500

/-- Represents the interest earned in the first year in dollars -/
def first_year_interest : ℝ := 100

/-- Represents the balance at the end of the first year in dollars -/
def first_year_balance : ℝ := 600

/-- Represents the percentage increase in the second year -/
def second_year_increase_rate : ℝ := 0.1

/-- Represents the total percentage increase over two years -/
def total_increase_rate : ℝ := 0.32

theorem initial_deposit_proof :
  initial_deposit + first_year_interest = first_year_balance ∧
  first_year_balance * (1 + second_year_increase_rate) = initial_deposit * (1 + total_increase_rate) := by
  sorry

#check initial_deposit_proof

end initial_deposit_proof_l3159_315904


namespace g_properties_l3159_315949

noncomputable def f (a x : ℝ) : ℝ := a * Real.sqrt (1 - x^2) + Real.sqrt (1 + x) + Real.sqrt (1 - x)

noncomputable def g (a : ℝ) : ℝ := ⨆ (x : ℝ), f a x

theorem g_properties (a : ℝ) :
  (a > -1/2 → g a = a + 2) ∧
  (-Real.sqrt 2 / 2 < a ∧ a ≤ -1/2 → g a = -a - 1/(2*a)) ∧
  (a ≤ -Real.sqrt 2 / 2 → g a = Real.sqrt 2) ∧
  (g a = g (1/a) ↔ a = 1 ∨ (-Real.sqrt 2 ≤ a ∧ a ≤ -Real.sqrt 2 / 2)) :=
sorry

end g_properties_l3159_315949


namespace smallest_positive_integer_ending_in_9_divisible_by_11_l3159_315984

theorem smallest_positive_integer_ending_in_9_divisible_by_11 :
  ∃ (n : ℕ), n > 0 ∧ n % 10 = 9 ∧ n % 11 = 0 ∧
  ∀ (m : ℕ), m > 0 → m % 10 = 9 → m % 11 = 0 → m ≥ n :=
by
  sorry

end smallest_positive_integer_ending_in_9_divisible_by_11_l3159_315984


namespace matinee_attendance_difference_l3159_315955

theorem matinee_attendance_difference (child_price adult_price total_receipts num_children : ℚ)
  (h1 : child_price = 4.5)
  (h2 : adult_price = 6.75)
  (h3 : total_receipts = 405)
  (h4 : num_children = 48) :
  num_children - (total_receipts - num_children * child_price) / adult_price = 20 := by
  sorry

end matinee_attendance_difference_l3159_315955


namespace probability_marked_standard_deck_l3159_315968

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (total_ranks : ℕ)
  (total_suits : ℕ)
  (marked_ranks : ℕ)

/-- A standard deck with 52 cards, 13 ranks, 4 suits, and 4 marked ranks -/
def standard_deck : Deck :=
  { total_cards := 52,
    total_ranks := 13,
    total_suits := 4,
    marked_ranks := 4 }

/-- The probability of drawing a card with a special symbol -/
def probability_marked (d : Deck) : ℚ :=
  (d.marked_ranks * d.total_suits) / d.total_cards

/-- Theorem: The probability of drawing a card with a special symbol from a standard deck is 4/13 -/
theorem probability_marked_standard_deck :
  probability_marked standard_deck = 4 / 13 := by
  sorry

end probability_marked_standard_deck_l3159_315968


namespace horse_division_l3159_315999

theorem horse_division (total_horses : ℕ) (eldest_share middle_share youngest_share : ℕ) : 
  total_horses = 7 →
  eldest_share = 4 →
  middle_share = 2 →
  youngest_share = 1 →
  eldest_share + middle_share + youngest_share = total_horses →
  eldest_share = (total_horses + 1) / 2 →
  middle_share = (total_horses + 1) / 4 →
  youngest_share = (total_horses + 1) / 8 :=
by sorry

end horse_division_l3159_315999


namespace perpendicular_planes_l3159_315936

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines
variable (perpLine : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpLinePlane : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (perpPlane : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes 
  (m n : Line) (α β : Plane) :
  perpLine m n → 
  perpLinePlane m α → 
  perpLinePlane n β → 
  perpPlane α β :=
sorry

end perpendicular_planes_l3159_315936


namespace playground_area_l3159_315971

theorem playground_area (total_posts : ℕ) (post_spacing : ℕ) 
  (h1 : total_posts = 28)
  (h2 : post_spacing = 6)
  (h3 : ∃ (short_side long_side : ℕ), 
    short_side + 1 + long_side + 1 = total_posts ∧ 
    long_side + 1 = 3 * (short_side + 1)) :
  ∃ (width length : ℕ), 
    width * length = 1188 ∧ 
    width = post_spacing * short_side ∧ 
    length = post_spacing * long_side :=
sorry

end playground_area_l3159_315971


namespace call_center_team_b_fraction_l3159_315998

/-- The fraction of total calls processed by team B in a call center with two teams -/
theorem call_center_team_b_fraction (team_a team_b : ℕ) (calls_a calls_b : ℚ) :
  team_a = (5 : ℚ) / 8 * team_b →
  calls_a = (7 : ℚ) / 5 * calls_b →
  (team_b * calls_b) / (team_a * calls_a + team_b * calls_b) = (8 : ℚ) / 15 := by
sorry


end call_center_team_b_fraction_l3159_315998


namespace escalator_time_theorem_l3159_315912

/-- The time taken for a person to cover the length of an escalator -/
theorem escalator_time_theorem (escalator_speed : ℝ) (person_speed : ℝ) (escalator_length : ℝ) :
  escalator_speed = 12 →
  person_speed = 8 →
  escalator_length = 160 →
  escalator_length / (escalator_speed + person_speed) = 8 := by
  sorry

end escalator_time_theorem_l3159_315912


namespace b_spend_percent_calculation_l3159_315990

def combined_salary : ℝ := 3000
def a_salary : ℝ := 2250
def a_spend_percent : ℝ := 0.95

theorem b_spend_percent_calculation :
  let b_salary := combined_salary - a_salary
  let a_savings := a_salary * (1 - a_spend_percent)
  let b_spend_percent := 1 - (a_savings / b_salary)
  b_spend_percent = 0.85 := by sorry

end b_spend_percent_calculation_l3159_315990


namespace tulip_price_is_two_l3159_315977

/-- Represents the price of a tulip in dollars -/
def tulip_price : ℝ := 2

/-- Represents the price of a rose in dollars -/
def rose_price : ℝ := 3

/-- Calculates the total revenue for the three days -/
def total_revenue (tulip_price : ℝ) : ℝ :=
  -- First day
  (30 * tulip_price + 20 * rose_price) +
  -- Second day
  (60 * tulip_price + 40 * rose_price) +
  -- Third day
  (6 * tulip_price + 16 * rose_price)

theorem tulip_price_is_two :
  total_revenue tulip_price = 420 :=
by sorry

end tulip_price_is_two_l3159_315977


namespace x_range_l3159_315932

theorem x_range (x : ℝ) : (|x + 1| + |x - 1| = 2) ↔ (-1 ≤ x ∧ x ≤ 1) := by sorry

end x_range_l3159_315932


namespace older_brother_allowance_l3159_315992

theorem older_brother_allowance (younger_allowance older_allowance : ℕ) : 
  younger_allowance + older_allowance = 12000 →
  older_allowance = younger_allowance + 1000 →
  older_allowance = 6500 := by
sorry

end older_brother_allowance_l3159_315992


namespace largest_n_satisfying_equation_l3159_315994

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating that 397 is the largest positive integer satisfying the equation -/
theorem largest_n_satisfying_equation :
  ∀ n : ℕ, n > 0 → n = (sum_of_digits n)^2 + 2*(sum_of_digits n) - 2 → n ≤ 397 :=
by sorry

end largest_n_satisfying_equation_l3159_315994


namespace evaluate_expression_l3159_315935

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 6560 := by
  sorry

end evaluate_expression_l3159_315935


namespace large_box_125_times_small_box_l3159_315986

-- Define the dimensions of the large box
def large_width : ℝ := 30
def large_length : ℝ := 20
def large_height : ℝ := 5

-- Define the dimensions of the small box
def small_width : ℝ := 6
def small_length : ℝ := 4
def small_height : ℝ := 1

-- Define the volume calculation function for a cuboid
def cuboid_volume (width length height : ℝ) : ℝ := width * length * height

-- Theorem statement
theorem large_box_125_times_small_box :
  cuboid_volume large_width large_length large_height =
  125 * cuboid_volume small_width small_length small_height := by
  sorry

end large_box_125_times_small_box_l3159_315986


namespace decagon_diagonal_intersections_l3159_315957

-- Define a regular polygon
def RegularPolygon (n : ℕ) := {p : ℕ | p ≥ 3}

-- Define a decagon
def Decagon := RegularPolygon 10

-- Define the number of interior intersection points of diagonals
def InteriorIntersectionPoints (p : RegularPolygon 10) : ℕ := sorry

-- Define the number of ways to choose 4 vertices from 10
def Choose4From10 : ℕ := Nat.choose 10 4

-- Theorem statement
theorem decagon_diagonal_intersections (d : Decagon) : 
  InteriorIntersectionPoints d = Choose4From10 := by sorry

end decagon_diagonal_intersections_l3159_315957


namespace new_person_weight_l3159_315913

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 20 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 40 :=
by sorry

end new_person_weight_l3159_315913


namespace optimal_mask_pricing_l3159_315961

/-- Represents the cost and pricing model for masks during an epidemic --/
structure MaskPricing where
  costA : ℝ  -- Cost of type A masks
  costB : ℝ  -- Cost of type B masks
  sellingPrice : ℝ  -- Selling price of type B masks
  profit : ℝ  -- Daily average total profit

/-- Conditions for the mask pricing problem --/
def MaskPricingConditions (m : MaskPricing) : Prop :=
  m.costB = 2 * m.costA - 10 ∧  -- Condition 1
  6000 / m.costA = 10000 / m.costB ∧  -- Condition 2
  m.profit = (m.sellingPrice - m.costB) * (100 - 5 * (m.sellingPrice - 60))  -- Conditions 3 and 4 combined

/-- Theorem stating the optimal solution for the mask pricing problem --/
theorem optimal_mask_pricing :
  ∃ m : MaskPricing,
    MaskPricingConditions m ∧
    m.costA = 30 ∧
    m.costB = 50 ∧
    m.sellingPrice = 65 ∧
    m.profit = 1125 ∧
    ∀ m' : MaskPricing, MaskPricingConditions m' → m'.profit ≤ m.profit :=
by
  sorry

end optimal_mask_pricing_l3159_315961


namespace smallest_constant_inequality_l3159_315941

theorem smallest_constant_inequality (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) :
  Real.sqrt (x / (y + z + w)) + Real.sqrt (y / (x + z + w)) +
  Real.sqrt (z / (x + y + w)) + Real.sqrt (w / (x + y + z)) ≤ 2 ∧
  ∀ M : ℝ, M < 2 → ∃ a b c d : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    Real.sqrt (a / (b + c + d)) + Real.sqrt (b / (a + c + d)) +
    Real.sqrt (c / (a + b + d)) + Real.sqrt (d / (a + b + c)) > M :=
by sorry

end smallest_constant_inequality_l3159_315941


namespace probability_13_or_more_points_l3159_315963

/-- Represents the face cards in a deck --/
inductive FaceCard
  | A
  | K
  | Q
  | J

/-- Assigns points to a face card --/
def point_value (card : FaceCard) : ℕ :=
  match card with
  | FaceCard.A => 4
  | FaceCard.K => 3
  | FaceCard.Q => 2
  | FaceCard.J => 1

/-- Calculates the total points for a hand of face cards --/
def hand_points (hand : List FaceCard) : ℕ :=
  hand.map point_value |>.sum

/-- Represents all possible 4-card hands of face cards --/
def all_hands : List (List FaceCard) :=
  sorry

/-- Checks if a hand has 13 or more points --/
def has_13_or_more_points (hand : List FaceCard) : Bool :=
  hand_points hand ≥ 13

/-- Counts the number of hands with 13 or more points --/
def count_13_or_more : ℕ :=
  all_hands.filter has_13_or_more_points |>.length

theorem probability_13_or_more_points :
  count_13_or_more / all_hands.length = 197 / 1820 := by
  sorry

end probability_13_or_more_points_l3159_315963


namespace line_only_count_l3159_315916

/-- Represents the alphabet with its properties -/
structure Alphabet where
  total : ℕ
  dot_and_line : ℕ
  dot_only : ℕ
  has_dot_or_line : total = dot_and_line + dot_only + (total - (dot_and_line + dot_only))

/-- The specific alphabet from the problem -/
def problem_alphabet : Alphabet := {
  total := 50
  dot_and_line := 16
  dot_only := 4
  has_dot_or_line := by sorry
}

/-- The number of letters with a straight line but no dot -/
def line_only (a : Alphabet) : ℕ := a.total - (a.dot_and_line + a.dot_only)

/-- Theorem stating the result for the problem alphabet -/
theorem line_only_count : line_only problem_alphabet = 30 := by sorry

end line_only_count_l3159_315916


namespace egg_distribution_l3159_315991

theorem egg_distribution (a : ℚ) : a = 7 ↔
  (a / 2 - 1 / 2) / 2 - 1 / 2 - ((a / 4 - 3 / 4) / 2 + 1 / 2) = 0 :=
by sorry

end egg_distribution_l3159_315991


namespace range_of_t_l3159_315966

theorem range_of_t (x t : ℝ) : 
  (∀ x, (1 < x ∧ x ≤ 4) → |x - t| < 1) →
  (2 ≤ t ∧ t ≤ 3) := by
sorry

end range_of_t_l3159_315966


namespace perfect_square_pairs_l3159_315995

theorem perfect_square_pairs (x y : ℕ) :
  (∃ a : ℕ, x^2 + 8*y = a^2) ∧ (∃ b : ℕ, y^2 - 8*x = b^2) →
  (∃ n : ℕ, x = n ∧ y = n + 2) ∨
  ((x = 7 ∧ y = 15) ∨ (x = 33 ∧ y = 17) ∨ (x = 45 ∧ y = 23)) :=
by sorry

end perfect_square_pairs_l3159_315995


namespace arithmetic_sequence_problem_l3159_315964

def arithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
  (h1 : arithmeticSequence a)
  (h2 : a 2 + a 9 + a 12 - a 14 + a 20 - a 7 = 8) :
  a 9 - (1/4) * a 3 = 3 := by
  sorry

end arithmetic_sequence_problem_l3159_315964


namespace binomial_variance_example_l3159_315920

/-- The variance of a binomial distribution with 100 trials and 0.02 probability of success is 1.96 -/
theorem binomial_variance_example :
  let n : ℕ := 100
  let p : ℝ := 0.02
  let q : ℝ := 1 - p
  let variance : ℝ := n * p * q
  variance = 1.96 := by
  sorry

end binomial_variance_example_l3159_315920


namespace parallel_vectors_x_value_l3159_315933

/-- Two vectors are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given vectors a and b, prove that if they are parallel, then x = 6 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (4, 2)
  let b : ℝ × ℝ := (x, 3)
  parallel a b → x = 6 := by
  sorry

end parallel_vectors_x_value_l3159_315933


namespace largest_five_digit_sum_20_l3159_315908

def is_five_digit (n : ℕ) : Prop := n ≥ 10000 ∧ n < 100000

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem largest_five_digit_sum_20 :
  ∀ n : ℕ, is_five_digit n → digit_sum n = 20 → n ≤ 99200 :=
by sorry

end largest_five_digit_sum_20_l3159_315908


namespace no_three_subset_partition_of_positive_integers_l3159_315976

theorem no_three_subset_partition_of_positive_integers :
  ¬ ∃ (A B C : Set ℕ),
    (A ∪ B ∪ C = {n : ℕ | n > 0}) ∧
    (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (C ∩ A = ∅) ∧
    (A ≠ ∅) ∧ (B ≠ ∅) ∧ (C ≠ ∅) ∧
    (∀ x y : ℕ, x > 0 → y > 0 →
      ((x ∈ A ∧ y ∈ B) ∨ (x ∈ B ∧ y ∈ A) → x^2 - x*y + y^2 ∈ C) ∧
      ((x ∈ B ∧ y ∈ C) ∨ (x ∈ C ∧ y ∈ B) → x^2 - x*y + y^2 ∈ A) ∧
      ((x ∈ C ∧ y ∈ A) ∨ (x ∈ A ∧ y ∈ C) → x^2 - x*y + y^2 ∈ B)) :=
sorry

end no_three_subset_partition_of_positive_integers_l3159_315976


namespace mary_lambs_traded_l3159_315905

def lambs_traded_for_goat (initial_lambs : ℕ) (lambs_with_babies : ℕ) (babies_per_lamb : ℕ) (extra_lambs : ℕ) (final_lambs : ℕ) : ℕ :=
  initial_lambs + lambs_with_babies * babies_per_lamb + extra_lambs - final_lambs

theorem mary_lambs_traded :
  lambs_traded_for_goat 6 2 2 7 14 = 3 := by
  sorry

end mary_lambs_traded_l3159_315905


namespace percentage_written_second_week_l3159_315960

/-- Proves that the percentage of remaining pages written in the second week is 30% --/
theorem percentage_written_second_week :
  ∀ (total_pages : ℕ) 
    (first_week_pages : ℕ) 
    (damaged_percentage : ℚ) 
    (final_empty_pages : ℕ),
  total_pages = 500 →
  first_week_pages = 150 →
  damaged_percentage = 20 / 100 →
  final_empty_pages = 196 →
  ∃ (second_week_percentage : ℚ),
    second_week_percentage = 30 / 100 ∧
    final_empty_pages = 
      (1 - damaged_percentage) * 
      (total_pages - first_week_pages - 
       (second_week_percentage * (total_pages - first_week_pages))) :=
by sorry

end percentage_written_second_week_l3159_315960


namespace heart_stickers_count_l3159_315928

/-- Represents the number of stickers needed to decorate a single page -/
def stickers_per_page (total_stickers : ℕ) (num_pages : ℕ) : ℕ :=
  total_stickers / num_pages

/-- Checks if the total number of stickers can be evenly distributed among the pages -/
def can_distribute_evenly (total_stickers : ℕ) (num_pages : ℕ) : Prop :=
  total_stickers % num_pages = 0

theorem heart_stickers_count (star_stickers : ℕ) (num_pages : ℕ) (heart_stickers : ℕ) : 
  star_stickers = 27 →
  num_pages = 9 →
  can_distribute_evenly (heart_stickers + star_stickers) num_pages →
  heart_stickers = 9 := by
  sorry

end heart_stickers_count_l3159_315928


namespace magazine_choice_count_l3159_315975

theorem magazine_choice_count : 
  let science_count : Nat := 4
  let digest_count : Nat := 3
  let entertainment_count : Nat := 2
  science_count + digest_count + entertainment_count = 9 :=
by sorry

end magazine_choice_count_l3159_315975


namespace shaded_area_between_circles_l3159_315974

theorem shaded_area_between_circles (r₁ r₂ : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 5) : 
  let R := (r₁ + r₂) / 2
  π * R^2 - (π * r₁^2 + π * r₂^2) = 40 * π :=
by sorry

end shaded_area_between_circles_l3159_315974


namespace adam_final_money_l3159_315978

/-- Calculates the final amount of money Adam has after a series of transactions --/
theorem adam_final_money (initial : ℚ) (game_cost : ℚ) (snack_cost : ℚ) (found : ℚ) (allowance : ℚ) :
  initial = 5.25 →
  game_cost = 2.30 →
  snack_cost = 1.75 →
  found = 1.00 →
  allowance = 5.50 →
  initial - game_cost - snack_cost + found + allowance = 7.70 := by
  sorry

end adam_final_money_l3159_315978


namespace answer_A_first_is_better_l3159_315919

-- Define the probabilities and point values
def prob_A : ℝ := 0.7
def prob_B : ℝ := 0.5
def points_A : ℝ := 40
def points_B : ℝ := 60

-- Define the expected score when answering A first
def E_A : ℝ := (1 - prob_A) * 0 + prob_A * (1 - prob_B) * points_A + prob_A * prob_B * (points_A + points_B)

-- Define the expected score when answering B first
def E_B : ℝ := (1 - prob_B) * 0 + prob_B * (1 - prob_A) * points_B + prob_B * prob_A * (points_A + points_B)

-- Theorem: Answering A first yields a higher expected score
theorem answer_A_first_is_better : E_A > E_B := by
  sorry

end answer_A_first_is_better_l3159_315919


namespace add_five_sixteen_base7_l3159_315981

/-- Converts a base 7 number to decimal --/
def toDecimal (b₇ : ℕ) : ℕ := sorry

/-- Converts a decimal number to base 7 --/
def toBase7 (d : ℕ) : ℕ := sorry

/-- Addition in base 7 --/
def addBase7 (a₇ b₇ : ℕ) : ℕ := 
  toBase7 (toDecimal a₇ + toDecimal b₇)

theorem add_five_sixteen_base7 : 
  addBase7 5 16 = 24 := by sorry

end add_five_sixteen_base7_l3159_315981


namespace modular_inverse_of_5_mod_23_l3159_315980

theorem modular_inverse_of_5_mod_23 : ∃ x : ℕ, x < 23 ∧ (5 * x) % 23 = 1 :=
by
  use 14
  constructor
  · norm_num
  · norm_num

#eval (5 * 14) % 23  -- This should output 1

end modular_inverse_of_5_mod_23_l3159_315980


namespace count_solution_pairs_l3159_315943

/-- The number of pairs of positive integers (x, y) satisfying x^2 - y^2 = 72 -/
def solution_count : Nat :=
  (Finset.filter (fun p : Nat × Nat =>
    let (x, y) := p
    x > 0 ∧ y > 0 ∧ x^2 - y^2 = 72
  ) (Finset.product (Finset.range 100) (Finset.range 100))).card

theorem count_solution_pairs : solution_count = 3 := by
  sorry

end count_solution_pairs_l3159_315943


namespace system_solutions_l3159_315906

theorem system_solutions (a : ℝ) (x y : ℝ) 
  (h1 : x - 2*y = 3 - a) 
  (h2 : x + y = 2*a) 
  (h3 : -2 ≤ a ∧ a ≤ 0) : 
  (a = 0 → x = -y) ∧ 
  (a = -1 → 2*x - y = 1 - a) := by
sorry

end system_solutions_l3159_315906


namespace a_10_equals_512_l3159_315938

/-- The sequence {aₙ} where Sₙ = 2aₙ - 1 for all n ∈ ℕ⁺, and Sₙ is the sum of the first n terms of {aₙ} -/
def sequence_a (n : ℕ+) : ℝ :=
  sorry

/-- The sum of the first n terms of the sequence {aₙ} -/
def S (n : ℕ+) : ℝ :=
  sorry

/-- The main theorem stating that a₁₀ = 512 -/
theorem a_10_equals_512 (h : ∀ n : ℕ+, S n = 2 * sequence_a n - 1) : sequence_a 10 = 512 := by
  sorry

end a_10_equals_512_l3159_315938


namespace complex_square_simplification_l3159_315972

theorem complex_square_simplification :
  let z : ℂ := 4 - 3 * I
  z^2 = 7 - 24 * I := by sorry

end complex_square_simplification_l3159_315972


namespace student_number_problem_l3159_315927

theorem student_number_problem (x : ℝ) : 6 * x - 138 = 102 → x = 40 := by
  sorry

end student_number_problem_l3159_315927


namespace ice_cream_flavors_l3159_315942

/-- The number of ways to distribute n indistinguishable objects into k distinguishable boxes -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The binomial coefficient C(n,k) -/
def binomial_coefficient (n k : ℕ) : ℕ := sorry

theorem ice_cream_flavors :
  distribute 5 4 = binomial_coefficient 8 3 := by sorry

end ice_cream_flavors_l3159_315942


namespace circle_center_l3159_315962

/-- The equation of a circle in the x-y plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 + 4*y = -4

/-- The center of a circle -/
def is_center (h k : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = 16

/-- Theorem: The center of the circle with equation x^2 - 8x + y^2 + 4y = -4 is (4, -2) -/
theorem circle_center : is_center 4 (-2) := by sorry

end circle_center_l3159_315962


namespace sweets_expenditure_l3159_315917

theorem sweets_expenditure (initial_amount : ℝ) (amount_per_friend : ℝ) (num_friends : ℕ) :
  initial_amount = 10.50 →
  amount_per_friend = 3.40 →
  num_friends = 2 →
  initial_amount - (amount_per_friend * num_friends) = 3.70 :=
by sorry

end sweets_expenditure_l3159_315917


namespace optimal_strategy_highest_hunter_l3159_315922

/-- Represents a hunter in the treasure division game -/
structure Hunter :=
  (id : Nat)
  (coins : Nat)

/-- Represents the state of the game -/
structure GameState :=
  (n : Nat)  -- Total number of hunters
  (m : Nat)  -- Total number of coins
  (hunters : List Hunter)

/-- Checks if a proposal is accepted by majority vote -/
def isProposalAccepted (state : GameState) (proposal : List Hunter) : Prop :=
  2 * (proposal.filter (fun h => h.coins > 0)).length > state.hunters.length

/-- Generates the optimal proposal for a given hunter -/
def optimalProposal (state : GameState) (hunterId : Nat) : List Hunter :=
  sorry

/-- Theorem: The optimal strategy for the highest-numbered hunter is to propose
    m - (n ÷ 2) coins for themselves and 1 coin each for the even-numbered
    hunters below them, until they secure a majority vote -/
theorem optimal_strategy_highest_hunter (state : GameState) :
  let proposal := optimalProposal state state.n
  isProposalAccepted state proposal ∧
  (proposal.head?.map Hunter.coins).getD 0 = state.m - (state.n / 2) ∧
  (proposal.tail.filter (fun h => h.coins > 0)).all (fun h => h.coins = 1 ∧ h.id % 2 = 0) :=
  sorry


end optimal_strategy_highest_hunter_l3159_315922


namespace inclination_angle_expression_l3159_315958

theorem inclination_angle_expression (θ : Real) : 
  (2 : Real) * Real.tan θ = -1 → 
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = 1 / 3 := by
sorry

end inclination_angle_expression_l3159_315958


namespace fraction_irreducible_l3159_315988

theorem fraction_irreducible (n : ℕ+) : Nat.gcd (n^2 + n - 1) (n^2 + 2*n) = 1 := by
  sorry

end fraction_irreducible_l3159_315988


namespace problem_solution_l3159_315940

theorem problem_solution (a b c : ℝ) (m n : ℕ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq1 : (a + b) * (a + c) = b * c + 2)
  (h_eq2 : (b + c) * (b + a) = c * a + 5)
  (h_eq3 : (c + a) * (c + b) = a * b + 9)
  (h_abc : a * b * c = m / n)
  (h_coprime : Nat.Coprime m n) : 
  100 * m + n = 4532 := by
sorry

end problem_solution_l3159_315940


namespace class_gathering_problem_l3159_315946

theorem class_gathering_problem (male_students : ℕ) (female_students : ℕ) :
  female_students = male_students + 6 →
  (female_students : ℚ) / (male_students + female_students) = 2 / 3 →
  male_students + female_students = 18 :=
by
  sorry

end class_gathering_problem_l3159_315946


namespace liquid_rise_ratio_l3159_315996

-- Define the cones and their properties
structure Cone where
  radius : ℝ
  height : ℝ
  volume : ℝ

-- Define the marble
def marbleRadius : ℝ := 1

-- Define the cones
def narrowCone : Cone := { radius := 3, height := 0, volume := 0 }
def wideCone : Cone := { radius := 6, height := 0, volume := 0 }

-- State that both cones contain the same amount of liquid
axiom equal_volume : narrowCone.volume = wideCone.volume

-- Define the rise in liquid level after dropping the marble
def liquidRise (c : Cone) : ℝ := sorry

-- Theorem to prove
theorem liquid_rise_ratio :
  liquidRise narrowCone / liquidRise wideCone = 4 := by sorry

end liquid_rise_ratio_l3159_315996


namespace roof_dimension_difference_l3159_315997

/-- Represents the dimensions of a rectangular base pyramid roof -/
structure RoofDimensions where
  width : ℝ
  length : ℝ
  height : ℝ
  area : ℝ

/-- Conditions for the roof dimensions -/
def roof_conditions (r : RoofDimensions) : Prop :=
  r.length = 4 * r.width ∧
  r.area = 1024 ∧
  r.height = 50 ∧
  r.area = r.length * r.width

/-- Theorem stating the difference between length and width -/
theorem roof_dimension_difference (r : RoofDimensions) 
  (h : roof_conditions r) : r.length - r.width = 48 := by
  sorry

#check roof_dimension_difference

end roof_dimension_difference_l3159_315997


namespace artichokey_invested_seven_l3159_315923

/-- Represents the investment and payout of earthworms -/
structure EarthwormInvestment where
  total_earthworms : ℕ
  okeydokey_apples : ℕ
  okeydokey_earthworms : ℕ

/-- Calculates the number of apples Artichokey invested -/
def artichokey_investment (e : EarthwormInvestment) : ℕ :=
  sorry

/-- Theorem stating that Artichokey invested 7 apples -/
theorem artichokey_invested_seven (e : EarthwormInvestment)
  (h1 : e.total_earthworms = 60)
  (h2 : e.okeydokey_apples = 5)
  (h3 : e.okeydokey_earthworms = 25)
  (h4 : e.okeydokey_earthworms * e.total_earthworms = e.okeydokey_apples * (e.total_earthworms + e.okeydokey_earthworms)) :
  artichokey_investment e = 7 :=
sorry

end artichokey_invested_seven_l3159_315923


namespace xy_value_l3159_315910

theorem xy_value (x y : ℝ) (h1 : x - y = 5) (h2 : x^3 - y^3 = 40) : x * y = 85 / 6 := by
  sorry

end xy_value_l3159_315910


namespace average_weight_problem_l3159_315939

/-- Given the average weight of three people and some additional information,
    prove that the average weight of two of them is 43 kg. -/
theorem average_weight_problem (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →   -- average weight of a, b, and c
  (a + b) / 2 = 40 →       -- average weight of a and b
  b = 31 →                 -- weight of b
  (b + c) / 2 = 43         -- average weight of b and c to be proved
  := by sorry

end average_weight_problem_l3159_315939


namespace product_trailing_zeros_l3159_315902

/-- The number of trailing zeros in a natural number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The product of 40, 360, and 125 -/
def product : ℕ := 40 * 360 * 125

theorem product_trailing_zeros :
  trailingZeros product = 5 := by sorry

end product_trailing_zeros_l3159_315902


namespace line_mb_less_than_neg_one_l3159_315983

/-- A line passing through two points (0, 3) and (2, -1) -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept
  point1 : m * 0 + b = 3
  point2 : m * 2 + b = -1

/-- Theorem stating that for a line passing through (0, 3) and (2, -1), mb < -1 -/
theorem line_mb_less_than_neg_one (l : Line) : l.m * l.b < -1 := by
  sorry


end line_mb_less_than_neg_one_l3159_315983


namespace tenth_term_is_110_l3159_315965

/-- Define the sequence of small stars -/
def smallStars (n : ℕ) : ℕ := n * (n + 1)

/-- Theorem: The 10th term of the sequence is 110 -/
theorem tenth_term_is_110 : smallStars 10 = 110 := by
  sorry

end tenth_term_is_110_l3159_315965


namespace certain_number_equation_l3159_315989

theorem certain_number_equation (x : ℝ) : 7 * x = 4 * x + 12 + 6 ↔ x = 6 := by
  sorry

end certain_number_equation_l3159_315989


namespace hyperbola_vertex_distance_l3159_315929

/-- The distance between the vertices of a hyperbola with equation y²/48 - x²/16 = 1 is 8√3 -/
theorem hyperbola_vertex_distance :
  let hyperbola := {(x, y) : ℝ × ℝ | y^2 / 48 - x^2 / 16 = 1}
  ∃ v₁ v₂ : ℝ × ℝ, v₁ ∈ hyperbola ∧ v₂ ∈ hyperbola ∧ 
    ∀ p ∈ hyperbola, (p.1 = v₁.1 ∨ p.1 = v₂.1) → p.2 = 0 ∧
    ‖v₁ - v₂‖ = 8 * Real.sqrt 3 :=
by sorry

end hyperbola_vertex_distance_l3159_315929


namespace fairview_population_l3159_315956

/-- The number of cities in the District of Fairview -/
def num_cities : ℕ := 25

/-- The average population of cities in the District of Fairview -/
def avg_population : ℕ := 3800

/-- The total population of the District of Fairview -/
def total_population : ℕ := num_cities * avg_population

theorem fairview_population :
  total_population = 95000 := by
  sorry

end fairview_population_l3159_315956


namespace fraction_equality_implies_numerator_equality_l3159_315918

theorem fraction_equality_implies_numerator_equality 
  (a b c : ℝ) (h1 : c ≠ 0) (h2 : a / c = b / c) : a = b := by
  sorry

end fraction_equality_implies_numerator_equality_l3159_315918


namespace afternoon_shells_l3159_315944

/-- Given that Lino picked up 292 shells in the morning and a total of 616 shells,
    prove that he picked up 324 shells in the afternoon. -/
theorem afternoon_shells (morning_shells : ℕ) (total_shells : ℕ) (h1 : morning_shells = 292) (h2 : total_shells = 616) :
  total_shells - morning_shells = 324 := by
  sorry

end afternoon_shells_l3159_315944
